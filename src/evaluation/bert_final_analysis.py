"""Verify saved BERT results, compare paired bootstrap and select a private audit.

No training, checkpoint inference, reselection or changes to historical artifacts.
Run from root: python -m src.evaluation.bert_final_analysis --output reports/bert_final_analysis
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
from pathlib import Path
import random
import statistics

import numpy as np

from src.evaluation.metrics import LABELS, scores

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"Majority": "Majority", "POS": "POS rule", "LR_03": "LR Full", "Tree_03": "Tree Full"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def metric(y, pred):
    result = scores(y, pred)
    result["macro_precision"] = statistics.mean(v["precision"] for v in result["per_label"].values())
    result["macro_recall"] = statistics.mean(v["recall"] for v in result["per_label"].values())
    return result


def mean_std(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def group_bootstrap(metadata, predictions, groups, weights):
    """Fixed five-class F1, using the exact same group resamples for all models."""
    group_index = {name: i for i, name in enumerate(groups)}
    label_index = {name: i for i, name in enumerate(LABELS)}
    if set(group_index) != {row["group_id"] for row in metadata}:
        raise ValueError("Group set differs from historical bootstrap")
    if weights.ndim != 2 or weights.shape[1] != len(groups):
        raise ValueError("Invalid bootstrap weight shape")
    distributions = {}
    for name, pred in predictions.items():
        if len(pred) != len(metadata):
            raise ValueError("Predictions do not align with metadata")
        matrices = np.zeros((len(groups), len(LABELS), len(LABELS)), dtype=np.int64)
        for row, label in zip(metadata, pred, strict=True):
            matrices[group_index[row["group_id"]], label_index[row["error_type"]], label_index[label]] += 1
        boot = (weights @ matrices.reshape(len(groups), -1)).reshape(-1, len(LABELS), len(LABELS))
        diagonal = np.diagonal(boot, axis1=1, axis2=2)
        denominator = boot.sum(1) + boot.sum(2)
        distributions[name] = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                                         where=denominator > 0).mean(1)
    return distributions


def canonical_csv_hash(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()


def balanced_take(rows, count, rng):
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        buckets[(row["error_type"], row["BERT"], row["LR_03"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    if len(selected) != count:
        raise ValueError("Insufficient audit candidates")
    return selected


def choose_audit(rows, seed=20261028):
    rng = random.Random(seed)
    pools = {
        "BERT_wrong_LR_correct": [r for r in rows if r["BERT"] != r["error_type"] == r["LR_03"]],
        "BERT_correct_LR_wrong": [r for r in rows if r["BERT"] == r["error_type"] != r["LR_03"]],
        "both_wrong": [r for r in rows if r["BERT"] != r["error_type"] and r["LR_03"] != r["error_type"]],
        "both_correct_control": [r for r in rows if r["BERT"] == r["error_type"] == r["LR_03"]],
    }
    sva = balanced_take([r for r in pools["BERT_wrong_LR_correct"] if r["error_type"] == "SVA"], 6, rng)
    used = {r["sample_id"] for r in sva}
    first = sva + balanced_take([r for r in pools["BERT_wrong_LR_correct"] if r["sample_id"] not in used], 6, rng)
    selected = [("BERT_wrong_LR_correct", row) for row in first]
    for name, count in (("BERT_correct_LR_wrong", 12), ("both_wrong", 10), ("both_correct_control", 6)):
        selected += [(name, row) for row in balanced_take(pools[name], count, rng)]
    if len({r["sample_id"] for _, r in selected}) != 40:
        raise ValueError("Duplicate audit IDs")
    return selected


def create_audit(output, metadata, predictions, config):
    rows = [dict(row, BERT=predictions["BERT"][i]) for i, row in enumerate(metadata)]
    selected = choose_audit(rows)
    selection = [{"sample_id": row["sample_id"], "annotation_id": row["annotation_id"],
                  "stratum": category, "gold": row["error_type"], "BERT": row["BERT"], "LR": row["LR_03"]}
                 for category, row in selected]
    private = output / "private"
    private.mkdir()
    # Freeze IDs before looking up any sentence, correction or source interpretation.
    dump(private / "audit_selection.json", {"seed": 20261028, "selected_before_source_review": True,
        "method": "12 BERT-wrong/LR-correct (six prioritized gold SVA); 12 reverse; 10 both wrong; six controls; round-robin gold/pred buckets",
        "rows": selection})
    wanted = {r["sample_id"] for r in selection}
    samples = {}
    with Path(config["samples"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] in wanted:
                samples[row["sample_id"]] = row
    from src.data.constants import SGML_FILE
    from src.data.nucle_sgml import load_nucle
    from src.data.parse_nucle import documents_to_dataframe
    source = documents_to_dataframe(load_nucle(SGML_FILE)).set_index("annotation_id")
    if any(sid.split(":")[0] != sha(SGML_FILE) for sid in wanted):
        raise ValueError("Source SGML hash differs from sample provenance")
    packet, review_rows = [], []
    for item in selection:
        sample = samples[item["sample_id"]]
        original = source.loc[int(item["annotation_id"])]
        if sample["sentence"][sample["error_start"]:sample["error_end"]] != sample["error_span"]:
            raise ValueError("Audit span reconstruction differs")
        if original["error_type"] != item["gold"] or original["error_span"] != sample["error_span"]:
            raise ValueError("Audit gold/span differs from raw annotation")
        correction = "" if original["correction"] is None else str(original["correction"])
        before, after = sample["error_span"].casefold().split(), correction.casefold().split()
        operations = [op for op, *_ in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes() if op != "equal"]
        record = {**item, "sentence": sample["sentence"], "span": sample["error_span"],
                  "correction": correction, "missing_correction": bool(original["missing_correction"]),
                  "start": sample["error_start"], "end": sample["error_end"],
                  "edit_operations_casefold_whitespace": operations,
                  "source_gold_span_exact": True}
        packet.append(record)
        review_rows.append({**{k: record[k] for k in ("annotation_id", "sample_id", "stratum", "sentence", "span", "correction", "gold", "BERT", "LR")},
                            "review": "", "contribution": "", "reviewer": "", "confidence": ""})
    dump(private / "audit_packet.json", packet)
    csv_write(private / "error_review_template.csv", review_rows)
    summary = {"cases": len(packet), "strata": dict(Counter(r["stratum"] for r in packet)),
               "source_gold_span_exact": all(r["source_gold_span_exact"] for r in packet),
               "review_scope": "Purposeful post-prediction sample; source correction only for audit; no gold changes; no causal attribution",
               "human_review_status": "pending", "corpus_text_location": "private/ (local only)"}
    dump(output / "audit_summary.json", summary)
    return packet


def create_figures(output, overall, metrics, history, differences):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figures = output / "figures"
    figures.mkdir()
    labels = [row["model"] for row in overall]
    values = [row["test_macro_f1"] for row in overall]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=["#94a3b8", "#64748b", "#087f8c", "#5073b8", "#c45b39"])
    ax.errorbar(values, labels, xerr=np.array([[r["test_macro_f1"] - r["ci95_low"] for r in overall],
                                             [r["ci95_high"] - r["test_macro_f1"] for r in overall]]), fmt="none", color="#1e293b", capsize=4)
    ax.set(xlim=(0, 1), xlabel="Test macro-F1 (paired group-bootstrap 95% CI)")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(figures / "benchmark_ci.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(LABELS))
    for i, name in enumerate(("POS", "LR_03", "Tree_03", "BERT")):
        ax.bar(x + (i - 1.5) * .19, [metrics[name]["per_label"][l]["f1"] for l in LABELS], .19,
               label=NAMES.get(name, "BERT-small primary"))
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 1), ylabel="Test F1")
    ax.legend(ncol=2, loc="lower left"); fig.tight_layout(); fig.savefig(figures / "per_class_f1.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("LR_03", "BERT"), strict=True):
        c = np.array(metrics[name]["confusion_matrix_true_rows_predicted_columns"])
        ax.imshow(c, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(c[i, j]), ha="center", va="center", color="white" if c[i, j] > c.max() * .55 else "black", fontsize=8)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Predicted", ylabel="Gold", title=NAMES.get(name, "BERT-small primary"))
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(figures / "confusion_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, title in zip(axes, ("loss", "macro_f1"), ("Unweighted cross-entropy", "Macro-F1"), strict=True):
        for part in ("train", "validation"):
            ax.plot([r["epoch"] for r in history], [r[part][key] for r in history], label=part)
        ax.axvline(18, color="gray", linestyle=":", label="selected epoch 18")
        ax.set(xlabel="Epoch", ylabel=title); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "primary_learning_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    ys = np.arange(len(differences))
    for y, row in zip(ys, differences, strict=True):
        ax.plot(row["ci95"], [y, y], color="#087f8c", linewidth=3)
        ax.scatter(row["difference"], y, color="#c45b39", zorder=3)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(yticks=ys, yticklabels=[r["contrast"] for r in differences], xlabel="Paired difference in test macro-F1 (95% CI)")
    fig.tight_layout(); fig.savefig(figures / "paired_differences.png", dpi=180); plt.close(fig)


def run(experiment, output):
    if output.exists():
        raise FileExistsError("Use a new analysis output directory; do not overwrite recorded evidence")
    selection = read(experiment / "selection.json")
    manifest = read(experiment / "manifest.json")
    summary = read(experiment / "test/summary.json")
    config = manifest["config"]
    checks = {"selection_hash": sha(experiment / "selection.json") == (experiment / "selection.sha256").read_text().strip(),
              "manifest_hash": sha(experiment / "manifest.json") == selection["manifest_sha256"],
              "test_selection_hash": summary["selection_sha256"] == sha(experiment / "selection.json"),
              "code": all((ROOT / p).exists() and sha(ROOT / p) == h for p, h in selection["code"].items()),
              "inputs": all(sha(Path(config[k])) == manifest["inputs"][k] for k in ("samples", "split", "model_source"))}
    sources = [experiment / "selection.json", experiment / "manifest.json", experiment / "test/summary.json",
               ROOT / "reports/final_evaluation/test_predictions.csv", ROOT / "reports/final_evaluation/bootstrap.json",
               ROOT / "reports/final_evaluation/bootstrap_group_weights.npy"]
    metadata = csv_read(sources[3])
    ids = [r["sample_id"] for r in metadata]
    y = [r["error_type"] for r in metadata]
    if len(ids) != 2368 or len(set(ids)) != len(ids):
        raise ValueError("Expected 2368 unique test IDs")
    split = {r["sample_id"]: r for r in csv_read(Path(config["split"]))}
    checks["test_partitions"] = all(split[sid]["partition"] == "test" for sid in ids)
    checks["test_gold_groups"] = all(split[r["sample_id"]]["error_type"] == r["error_type"] and
                                     split[r["sample_id"]]["group_id"] == r["group_id"] for r in metadata)
    original_hash = read(ROOT / "reports/final_evaluation/test_scoring_complete.json")["predictions_sha256"]
    checks["classical_csv_content"] = canonical_csv_hash(metadata) == original_hash
    predictions = {name: [row[name] for row in metadata] for name in NAMES}
    validation_rows = []
    all_runs = {}
    for variant in selection["ranking"]:
        values = []
        for name in variant["runs"]:
            folder = experiment / name
            record = read(folder / "summary.json")
            checks[name + "_checkpoint"] = tree_hashes(folder / "best") == record["checkpoint_hashes"]
            checks[name + "_validation_hash"] = sha(folder / "validation_predictions.csv") == record["validation_prediction_sha256"]
            saved = csv_read(folder / "validation_predictions.csv")
            measured = metric([r["error_type"] for r in saved], [r["prediction"] for r in saved])
            checks[name + "_validation_metric"] = abs(measured["macro_f1"] - record["best"]["validation"]["macro_f1"]) < 1e-12
            values.append(measured["macro_f1"])
            all_runs[name] = record
            sources += [folder / "summary.json", folder / "history.json", folder / "validation_predictions.csv"]
        aggregate = mean_std(values)
        checks[str(variant["variant"]) + "_mean"] = abs(aggregate["mean"] - variant["validation_macro_f1"]["mean"]) < 1e-12
        validation_rows.append({**variant["variant"], "seeds": len(values), "validation_macro_f1_mean": aggregate["mean"],
                                "validation_macro_f1_sample_std": aggregate["sample_std"]})
    winner = max(selection["ranking"], key=lambda r: r["validation_macro_f1"]["mean"])
    checks["selected_by_validation_mean"] = winner == selection["selected"]
    checks["primary_seed_predeclared"] = selection["primary_seed"] == config["seeds"][0]
    test_seeds = []
    for saved in summary["runs"]:
        name = saved["run"]
        checks[name + "_locked_checkpoint"] = tree_hashes(experiment / name / "best") == selection["checkpoints"][name]
        path = experiment / "test" / f"{name}_test_predictions.csv"
        sources.append(path)
        rows = csv_read(path)
        by_id = {row["sample_id"]: row for row in rows}
        if len(rows) != len(by_id) or set(by_id) != set(ids):
            raise ValueError("Test prediction IDs differ")
        checks[name + "_test_gold"] = all(by_id[s]["error_type"] == truth for s, truth in zip(ids, y, strict=True))
        pred = [by_id[s]["prediction"] for s in ids]
        measured = metric(y, pred)
        checks[name + "_test_metric"] = all(measured[k] == saved["metrics"][k] for k in (
            "macro_f1", "accuracy", "per_label", "confusion_matrix_true_rows_predicted_columns"))
        predictions[name] = pred
        test_seeds.append({"seed": saved["seed"], "run": name, "validation_macro_f1": all_runs[name]["best"]["validation"]["macro_f1"],
                           "test_macro_f1": measured["macro_f1"], "accuracy": measured["accuracy"],
                           "best_epoch": all_runs[name]["best_epoch"], "primary": name == selection["primary_run"]})
    checks["test_seeds_complete"] = [r["run"] for r in summary["runs"]] == selection["selected"]["runs"]
    aggregate = mean_std([r["test_macro_f1"] for r in test_seeds])
    checks["test_mean_std"] = all(abs(aggregate[k] - summary["test_macro_f1"][k]) < 1e-12 for k in aggregate)
    if not all(checks.values()):
        raise ValueError(f"Failed checks: {[k for k, v in checks.items() if not v]}")
    predictions["BERT"] = predictions[selection["primary_run"]]
    metrics = {name: metric(y, pred) for name, pred in predictions.items()}
    bootstrap = read(sources[4])
    weights = np.load(sources[5])
    groups = bootstrap["groups"]
    expected = np.random.default_rng(bootstrap["seed"]).multinomial(len(groups), np.ones(len(groups)) / len(groups), size=1000)
    if not np.array_equal(expected, weights):
        raise ValueError("Historical bootstrap weights differ from seed")
    main = {name: predictions[name] for name in (*NAMES, "BERT")}
    distributions = group_bootstrap(metadata, main, groups, weights)
    differences = []
    for other in ("LR_03", "Tree_03", "POS", "Majority"):
        differences.append({"contrast": "BERT primary - " + NAMES[other],
                            "difference": metrics["BERT"]["macro_f1"] - metrics[other]["macro_f1"],
                            "ci95": np.quantile(distributions["BERT"] - distributions[other], [.025, .975]).tolist()})
    old_ci = bootstrap["slices"]["all"]["models"]
    for name in NAMES:
        if not np.allclose(np.quantile(distributions[name], [.025, .975]), old_ci[name]["ci95"], rtol=0, atol=1e-12):
            raise ValueError("Recomputed classical CI differs")
    val_old = {r["pipeline"]: float(r["validation_macro_f1"]) for r in csv_read(ROOT / "reports/evaluation_followup/reviewed_overall_results.csv")}
    overall = []
    for name in main:
        m = metrics[name]
        lo, hi = np.quantile(distributions[name], [.025, .975]).tolist()
        overall.append({"pipeline": name, "model": NAMES.get(name, "BERT-small primary"),
                        "validation_macro_f1": all_runs[selection["primary_run"]]["best"]["validation"]["macro_f1"] if name == "BERT" else val_old[name],
                        "test_macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                        "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"], "ci95_low": lo, "ci95_high": hi})
    overlaps = Counter((predictions["BERT"][i] == y[i], predictions["LR_03"][i] == y[i]) for i in range(len(y)))
    error_counts = {"both_correct": overlaps[True, True], "BERT_only_correct": overlaps[True, False],
                    "LR_only_correct": overlaps[False, True], "both_wrong": overlaps[False, False]}
    confusion_rows = []
    for name in main:
        c = metrics[name]["confusion_matrix_true_rows_predicted_columns"]
        for i, gold in enumerate(LABELS):
            for j, pred in enumerate(LABELS):
                if i != j:
                    confusion_rows.append({"model": NAMES.get(name, "BERT-small primary"), "gold": gold, "prediction": pred, "count": c[i][j]})
    output.mkdir(parents=True)
    dump(output / "verification.json", {"all_passed": True, "checks": checks, "scope": "Saved predictions and hashes only; no checkpoint inference/retraining",
                                        "legacy_classical_csv": "strip padding reconstructs original hash; file left unchanged"})
    dump(output / "bootstrap.json", {"method": "paired connected-group percentile; fixed five labels; pointwise 95% CI",
                                      "seed": bootstrap["seed"], "replicates": len(weights), "groups": len(groups),
                                      "primary_seed": selection["primary_seed"], "differences": differences,
                                      "model_ci": {r["pipeline"]: [r["ci95_low"], r["ci95_high"]] for r in overall},
                                      "seed_std_is_not_bootstrap_ci": True})
    dump(output / "summary.json", {"date": "2026-10-28", "selected": selection["selected"], "primary_run": selection["primary_run"],
                                    "seed_test_macro_f1": aggregate, "test_seeds": test_seeds, "metrics": {name: metrics[name] for name in main},
                                    "error_overlap": error_counts, "test_samples": len(ids), "test_groups": len(groups),
                                    "limitations": "One locked split; post-classical-test extension disclosed; no independent new test; no post-test tuning; fixed primary seed; pointwise CI"})
    csv_write(output / "overall_results.csv", overall)
    csv_write(output / "validation_grid.csv", validation_rows)
    csv_write(output / "seed_results.csv", test_seeds)
    csv_write(output / "paired_differences.csv", [{"contrast": r["contrast"], "difference": r["difference"], "ci95_low": r["ci95"][0], "ci95_high": r["ci95"][1]} for r in differences])
    csv_write(output / "per_class_metrics.csv", [{"model": NAMES.get(name, "BERT-small primary"), "label": label, **values}
                                               for name in main for label, values in metrics[name]["per_label"].items()])
    csv_write(output / "confusions.csv", confusion_rows)
    create_figures(output, overall, metrics, read(experiment / selection["primary_run"] / "history.json"), differences)
    create_audit(output, metadata, predictions, config)
    dump(output / "source_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__))},
        "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "outputs": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()},
        "scope": "New evidence synthesis; historical inputs unchanged; source corpus text only under private/"})
    print(json.dumps({"checks": len(checks), "all_passed": True, "primary_test_f1": metrics["BERT"]["macro_f1"],
                      "test_mean_std": aggregate, "error_overlap": error_counts, "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "reports/bert_small_experiment")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bert_final_analysis")
    args = parser.parse_args()
    run(args.experiment.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
