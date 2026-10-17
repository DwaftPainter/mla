"""Build a self-contained, pre-rendered notebook from an ablation summary."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.data.constants import PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.constants import PROJECT_ROOT

SUMMARY = PROJECT_ROOT / "reports" / "nucle_feature_ablation_v1" / "summary.json"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "04_feature_ablation_report.ipynb"
ORDER = ("S+L", "S+L+P", "S+L+P+M", "S+L+P+M+D", "Full_mask_span_lexical", "S+P+M+D_nonlexical")
SHORT = ("S+L", "S+L+P", "S+L+P+M", "Full", "Masked\nspan", "No\nlexical")


def fig_output(fig) -> dict:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"output_type": "display_data", "metadata": {},
            "data": {"image/png": base64.b64encode(stream.getvalue()).decode("ascii"),
                     "text/plain": "<Figure>"}}


def render(summary: dict) -> list[dict]:
    results = pd.DataFrame(summary["results"])
    selected_rows = []
    for variant, models in summary["selected_by_validation"].items():
        for model, row in models.items():
            selected_rows.append({"Variant": variant, "Features": summary["variant_feature_counts"][variant],
                                  "Model": model, "Selected config": str(row["configuration"]),
                                  "Val Macro-F1": row["validation_macro_f1"],
                                  "Val accuracy": row["validation_accuracy"],
                                  "Train accuracy": row["train_accuracy"]})
    selected = pd.DataFrame(selected_rows)
    weight_rows = []
    for variant in ORDER:
        for model in ("LogisticRegression", "DecisionTree"):
            for weight in (None, "balanced"):
                class_weights = results.configuration.map(lambda c: c["class_weight"])
                weight_mask = class_weights.isna() if weight is None else class_weights == weight
                subset = results[(results.variant == variant) & (results.model == model) & weight_mask]
                if model == "LogisticRegression":
                    subset = subset[subset.convergence.map(lambda c: c["converged"])]
                if subset.empty:
                    weight_rows.append({"Variant": variant, "Model": model,
                                        "class_weight": "balanced" if weight else "None",
                                        "Best Val Macro-F1": float("nan"), "Config": "No converged config"})
                    continue
                best = subset.loc[subset["macro_f1"].idxmax()]
                weight_rows.append({"Variant": variant, "Model": model,
                                    "class_weight": "balanced" if weight else "None",
                                    "Best Val Macro-F1": best.macro_f1,
                                    "Config": str(best.configuration)})
    weight_table = pd.DataFrame(weight_rows)
    outputs = [{"output_type": "stream", "name": "stdout", "text":
                f"{summary['fit_count']} fits | {summary['train_samples']:,} train | "
                f"{summary['validation_samples']:,} validation | {summary['runtime_seconds']/60:.1f} min | "
                f"test scored: {summary['test_scored']}\n"}]
    outputs.append({"output_type": "display_data", "metadata": {},
                    "data": {"text/html": selected.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
                             "text/plain": selected.to_string(index=False) }})
    outputs.append({"output_type": "display_data", "metadata": {},
                    "data": {"text/html": weight_table.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
                             "text/plain": weight_table.to_string(index=False) }})

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, model in enumerate(("LogisticRegression", "DecisionTree")):
        values = [summary["selected_by_validation"][v][model]["validation_macro_f1"] for v in ORDER]
        ax.bar([i + (offset - .5) * .34 for i in range(len(ORDER))], values, width=.34,
               label="LR" if offset == 0 else "Tree")
    ax.set_xticks(range(len(ORDER)), SHORT); ax.set_ylim(.5, .85)
    ax.set_ylabel("Validation Macro-F1"); ax.set_title("Best validation Macro-F1 by feature variant")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
    outputs.append(fig_output(fig))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model, title in zip(axes, ("LogisticRegression", "DecisionTree"),
                                ("Logistic Regression", "Decision Tree")):
        for offset, weight in enumerate(("None", "balanced")):
            rows = weight_table[(weight_table.Model == model) & (weight_table.class_weight == weight)]
            rows = rows.set_index("Variant").loc[list(ORDER)]
            ax.bar([i + (offset - .5) * .34 for i in range(len(ORDER))], rows["Best Val Macro-F1"],
                   width=.34, label=weight)
        ax.set_xticks(range(len(ORDER)), SHORT); ax.set_title(title)
        ax.grid(axis="y", alpha=.25); ax.set_ylim(.5, .85)
    axes[0].set_ylabel("Best validation Macro-F1"); axes[1].legend(title="class_weight")
    fig.suptitle("Best score within each class-weight setting"); fig.tight_layout()
    outputs.append(fig_output(fig))

    class_rows = []
    for model in ("LogisticRegression", "DecisionTree"):
        cfg = summary["selected_by_validation"]["S+L+P+M+D"][model]["configuration"]
        row = next(x for x in summary["results"] if x["variant"] == "S+L+P+M+D" and
                   x["model"] == model and x["configuration"] == cfg)
        for label, metrics in row["per_label"].items():
            class_rows.append({"Label": label, "Model": "LR" if model == "LogisticRegression" else "Tree",
                               "F1": metrics["f1"], "Support": metrics["support"]})
    per_class = pd.DataFrame(class_rows)
    pivot = per_class.pivot(index="Label", columns="Model", values="F1").reset_index()
    outputs.append({"output_type": "display_data", "metadata": {},
                    "data": {"text/html": pivot.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
                             "text/plain": pivot.to_string(index=False) }})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ("ArtOrDet", "Prep", "Nn", "Vform", "SVA")
    for offset, model in enumerate(("LR", "Tree")):
        values = per_class[per_class.Model == model].set_index("Label").loc[list(labels)]["F1"]
        ax.bar([i + (offset - .5) * .36 for i in range(5)], values, width=.36, label=model)
    ax.set_xticks(range(5), labels); ax.set_ylim(0, 1); ax.set_ylabel("F1")
    ax.set_title("Per-class F1 for selected full feature models")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout()
    outputs.append(fig_output(fig))
    return outputs


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    compact = {key: summary[key] for key in (
        "feature_schema", "feature_manifest_sha256", "split_manifest_sha256", "fit_partition",
        "selection_partition", "test_scored", "train_samples", "validation_samples",
        "variant_feature_counts", "selected_by_validation", "fit_count", "runtime_seconds",
        "started_at_utc", "finished_at_utc", "masking_specification")}
    compact["results"] = [{key: row[key] for key in (
        "variant", "description", "model", "configuration", "feature_count", "fit_seconds",
        "train_accuracy", "macro_f1", "accuracy", "per_label", "convergence") if key in row}
        for row in summary["results"]]

    score = lambda variant, model: summary["selected_by_validation"][variant][model]["validation_macro_f1"]
    non_converged = sum(row["model"] == "LogisticRegression" and not row["convergence"]["converged"]
                        for row in summary["results"])
    rows = []
    for variant in ORDER:
        for model, title in (("LogisticRegression", "LR"), ("DecisionTree", "Tree")):
            selection = summary["selected_by_validation"][variant][model]
            rows.append(f"| {variant} | {summary['variant_feature_counts'][variant]:,} | {title} | "
                        f"`{selection['configuration']}` | {selection['validation_macro_f1']:.4f} | "
                        f"{selection['validation_accuracy']:.4f} | {selection['train_accuracy']:.4f} |")
    lr, tree = score("S+L+P+M+D", "LogisticRegression"), score("S+L+P+M+D", "DecisionTree")
    findings = f"""# NUCLE feature ablation and masking — validation report

**Run date:** 2026-10-17  
**Scope:** models fit on train; validation selects configurations; the held-out test partition was not scored.

This run compares the incremental sequence S+L → S+L+P → S+L+P+M → S+L+P+M+D, plus target-span lexical masking and the non-lexical S+P+M+D condition. Every condition tunes the existing LR C/class-weight grid and Tree depth/leaf/class-weight grid. The masking encoder was rebuilt after masking on train rows; lexical context outside the span remains. The run contains {summary['fit_count']} fits ({summary['train_samples']:,} train, {summary['validation_samples']:,} validation) and took {summary['runtime_seconds']/60:.1f} minutes.

The first masked-feature pass was discarded after auditing which encoded fields contained span lexical identity; only the corrected masking run is included in the results and runtime. The other five variants are retained from the original full benchmark run.

Feature manifest SHA-256: `{summary['feature_manifest_sha256']}`  
Split manifest SHA-256: `{summary['split_manifest_sha256']}`  
LR configurations that did not converge and were excluded from selection: {non_converged}/48.

## Selected validation results

| Variant | Features | Model | Selected configuration | Macro-F1 | Accuracy | Train accuracy |
|---|---:|---|---|---:|---:|---:|
{chr(10).join(rows)}

The best full-feature results are LR Macro-F1 **{lr:.4f}** and Tree **{tree:.4f}**. Both selected unweighted configurations.

## Findings

- Adding P to S+L changes Macro-F1 by **{score('S+L+P','LogisticRegression')-score('S+L','LogisticRegression'):+.4f}** for LR and **{score('S+L+P','DecisionTree')-score('S+L','DecisionTree'):+.4f}** for Tree. It gives the largest gain in the incremental sequence for both models.
- Adding M after P changes Macro-F1 by **{score('S+L+P+M','LogisticRegression')-score('S+L+P','LogisticRegression'):+.4f}** for LR and **{score('S+L+P+M','DecisionTree')-score('S+L+P','DecisionTree'):+.4f}** for Tree.
- Adding D after M changes Macro-F1 by **{score('S+L+P+M+D','LogisticRegression')-score('S+L+P+M','LogisticRegression'):+.4f}** for LR and **{score('S+L+P+M+D','DecisionTree')-score('S+L+P+M','DecisionTree'):+.4f}** for Tree. These are sequential tuned comparisons, not isolated causal effects.
- Masking target-span lexical identity scores **{score('Full_mask_span_lexical','LogisticRegression'):.4f}** LR and **{score('Full_mask_span_lexical','DecisionTree'):.4f}** Tree. The difference from full features is **{score('Full_mask_span_lexical','LogisticRegression')-lr:+.4f}** and **{score('Full_mask_span_lexical','DecisionTree')-tree:+.4f}** respectively.
- Removing all lexical features scores **{score('S+P+M+D_nonlexical','LogisticRegression'):.4f}** LR and **{score('S+P+M+D_nonlexical','DecisionTree'):.4f}** Tree. The full-to-nonlexical difference is **{score('S+P+M+D_nonlexical','LogisticRegression')-lr:+.4f}** and **{score('S+P+M+D_nonlexical','DecisionTree')-tree:+.4f}**.
- The class-weight table below compares the best converged configuration under `None` and `balanced` for every variant/model. Balanced weights help selected Tree variants, but not the selected full-feature model; LR selects unweighted in all six variants.

## Detailed tables and charts

The notebook displays the selected-config table, per-weight comparison, per-class F1 for the full models, and three charts. Embedded values are aggregate metrics only; no source sentences or sample IDs are included.

## Limits and next step

Each feature condition selects hyperparameters on the same validation set, so the differences are validation model-selection evidence rather than an unbiased final estimate. Test remains unopened. Next, freeze the feature/model protocol using validation, then score test once.

To rerun the benchmark and regenerate this notebook from its local aggregate report, run `python src/training/nucle_ablation.py`, then `python src/reporting/build_ablation_notebook.py` from the project root.
"""
    source = "import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nREPORT = json.loads(" + repr(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) + ")\n" + '''results = pd.DataFrame(REPORT['results'])
selected_rows = []
for variant, models in REPORT['selected_by_validation'].items():
    for model, row in models.items():
        selected_rows.append({'Variant': variant, 'Features': REPORT['variant_feature_counts'][variant], 'Model': model, 'Selected config': row['configuration'], 'Val Macro-F1': row['validation_macro_f1'], 'Val accuracy': row['validation_accuracy'], 'Train accuracy': row['train_accuracy']})
selected = pd.DataFrame(selected_rows)
weight_rows = []
for variant in REPORT['variant_feature_counts']:
    for model in ('LogisticRegression', 'DecisionTree'):
        for weight in (None, 'balanced'):
            class_weights = results.configuration.map(lambda c: c['class_weight'])
            weight_mask = class_weights.isna() if weight is None else class_weights == weight
            subset = results[(results.variant == variant) & (results.model == model) & weight_mask]
            if model == 'LogisticRegression': subset = subset[subset.convergence.map(lambda c: c['converged'])]
            if subset.empty:
                weight_rows.append({'Variant': variant, 'Model': model, 'class_weight': 'balanced' if weight else 'None', 'Best Val Macro-F1': float('nan'), 'Config': 'No converged config'})
                continue
            best = subset.loc[subset['macro_f1'].idxmax()]
            weight_rows.append({'Variant': variant, 'Model': model, 'class_weight': 'balanced' if weight else 'None', 'Best Val Macro-F1': best['macro_f1'], 'Config': best['configuration']})
weight_comparison = pd.DataFrame(weight_rows)
print(f"{REPORT['fit_count']} fits | {REPORT['train_samples']:,} train | {REPORT['validation_samples']:,} validation | {REPORT['runtime_seconds']/60:.1f} min | test scored: {REPORT['test_scored']}")
print(selected.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(weight_comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

order = list(REPORT['variant_feature_counts'])
labels = ['S+L','S+L+P','S+L+P+M','Full','Masked\\nspan','No\\nlexical']
fig, ax = plt.subplots(figsize=(11,5))
for offset, model in enumerate(('LogisticRegression','DecisionTree')):
    vals = [REPORT['selected_by_validation'][v][model]['validation_macro_f1'] for v in order]
    ax.bar([i+(offset-.5)*.34 for i in range(len(order))], vals, width=.34, label='LR' if offset == 0 else 'Tree')
ax.set_xticks(range(len(order)), labels); ax.set_ylim(.5,.85); ax.set_ylabel('Validation Macro-F1'); ax.set_title('Best validation Macro-F1 by feature variant'); ax.grid(axis='y',alpha=.25); ax.legend(); fig.tight_layout(); plt.show()

fig, axes = plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax, model, title in zip(axes, ('LogisticRegression','DecisionTree'), ('Logistic Regression','Decision Tree')):
    for offset, weight in enumerate(('None','balanced')):
        rows = weight_comparison[(weight_comparison.Model == model) & (weight_comparison.class_weight == weight)].set_index('Variant').loc[order]
        ax.bar([i+(offset-.5)*.34 for i in range(len(order))], rows['Best Val Macro-F1'], width=.34, label=weight)
    ax.set_xticks(range(len(order)), labels); ax.set_title(title); ax.grid(axis='y',alpha=.25); ax.set_ylim(.5,.85)
axes[0].set_ylabel('Best validation Macro-F1'); axes[1].legend(title='class_weight'); fig.suptitle('Best score within each class-weight setting'); fig.tight_layout(); plt.show()

class_rows = []
for model in ('LogisticRegression','DecisionTree'):
    cfg = REPORT['selected_by_validation']['S+L+P+M+D'][model]['configuration']
    row = results[(results.variant=='S+L+P+M+D') & (results.model==model) & (results.configuration==cfg)].iloc[0]
    for label, metrics in row.per_label.items(): class_rows.append({'Label':label,'Model':'LR' if model=='LogisticRegression' else 'Tree','F1':metrics['f1'],'Support':metrics['support']})
per_class = pd.DataFrame(class_rows)
print(per_class.pivot(index='Label',columns='Model',values='F1').to_string(float_format=lambda x: f'{x:.4f}'))
fig, ax = plt.subplots(figsize=(8,4.5))
for offset, model in enumerate(('LR','Tree')):
    vals = per_class[per_class.Model==model].set_index('Label').loc[['ArtOrDet','Prep','Nn','Vform','SVA']]['F1']
    ax.bar([i+(offset-.5)*.36 for i in range(5)], vals, width=.36, label=model)
ax.set_xticks(range(5), ['ArtOrDet','Prep','Nn','Vform','SVA']); ax.set_ylim(0,1); ax.set_ylabel('F1'); ax.set_title('Per-class F1 for selected full feature models'); ax.grid(axis='y',alpha=.25); ax.legend(); fig.tight_layout(); plt.show()
'''.replace("\n+", "\n")

    notebook = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": findings.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": render(summary),
         "source": source.splitlines(keepends=True)},
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11.2"}},
        "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
