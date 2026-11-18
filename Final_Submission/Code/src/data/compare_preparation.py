"""Read-only v1/v2 comparison and a fixed fresh review packet (no decisions)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

try:  # Support direct-file execution as well as package execution.
    from .constants import DATA_ROOT, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from .nucle_sgml import load_nucle
    from .prepare_nucle import DEFAULT_OUTPUT, V1_OUTPUT, canonical, digest, file_hash, require, write_json
except ImportError:
    from constants import DATA_ROOT, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from nucle_sgml import load_nucle
    from prepare_nucle import DEFAULT_OUTPUT, V1_OUTPUT, canonical, digest, file_hash, require, write_json

DEFAULT_REPORT = REPORTS_DIR / "nucle_preparation_v2_comparison"
FRESH_SEED = 2026101202


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def check_alignment(sample, sentence, record, paragraph):
    text = sentence["sentence"]
    require(sample["sentence"] == text == paragraph[record["sentence_start_par"]:record["sentence_end_par"]],
            "Sentence/source mismatch")
    start, end = sample["error_start"], sample["error_end"]
    require(0 <= start <= end <= len(text), "Invalid sample offsets")
    require(record["start_off"] == record["sentence_start_par"] + start
            and record["end_off"] == record["sentence_start_par"] + end, "Offset conversion mismatch")
    require(text[start:end] == sample["error_span"] == paragraph[record["start_off"]:record["end_off"]],
            "Span mismatch")
    tokens = sentence["tokens"]
    last = 0
    for index, token in enumerate(tokens):
        token_start, token_end = token["start"], token["end"]
        require(token["index"] == index and last <= token_start < token_end <= len(text),
                "Token ordering mismatch")
        require(text[token_start:token_end] == token["text"] and not token["text"].isspace(),
                "Token slice mismatch")
        require(not text[last:token_start].strip(), "Unrepresented non-whitespace text")
        last = token_end
    require(not text[last:].strip(), "Unrepresented sentence tail")
    expected_span = [] if start == end else [i for i, token in enumerate(tokens)
                                               if max(start, token["start"]) < min(end, token["end"])]
    expected_left = [i for i, token in enumerate(tokens) if token["end"] <= start][-2:]
    expected_right = [i for i, token in enumerate(tokens) if token["start"] >= end][:2]
    require(sample["span_token_indices"] == expected_span and sample["left_token_indices"] == expected_left
            and sample["right_token_indices"] == expected_right, "Token assignment mismatch")
    cuts = [any(token["start"] < point < token["end"] for token in tokens) for point in (start, end)]
    require(cuts == [sample["cuts_start_token"], sample["cuts_end_token"]], "Partial-token flags mismatch")
    if start == end:
        require(not any(cuts), "Insertion inside token")
        expected_alignment = "insertion_boundary"
    else:
        require(bool(expected_span), "No token overlap")
        expected_alignment = ("partial_token" if any(cuts) else "exact" if
                              start == tokens[expected_span[0]]["start"]
                              and end == tokens[expected_span[-1]]["end"] else "whitespace_edge")
    require(sample["alignment"] == expected_alignment, "Alignment category mismatch")


def load_checked(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    version = (folder / "manifest.sha256").read_text().strip()
    require(digest(canonical(manifest).encode()) == version, "Manifest identity mismatch")
    for name, expected in manifest["outputs"].items():
        require(file_hash(folder / name) == expected, f"Artifact checksum mismatch: {name}")
    return {"manifest": manifest, "version": version,
            "samples": {r["annotation_id"]: r for r in read_rows(folder / "samples.jsonl")},
            "provenance": {r["annotation_id"]: r for r in read_rows(folder / "provenance.jsonl")},
            "sentences": {r["sentence_id"]: r for r in read_rows(folder / "sentences.jsonl")}}


def fresh_selection(samples, changed_ids, previously_reviewed, seed=FRESH_SEED):
    """5/class from unseen samples, plus 5/class from unseen changed inputs."""
    eligible = [r for r in samples if r["annotation_id"] not in previously_reviewed]
    memberships, strata = {}, {}
    for group in ("A_fresh", "B_changed"):
        for label in TARGET_LABELS:
            name = f"{group}:{label}"
            ids = [r["annotation_id"] for r in eligible if r["error_type"] == label
                   and (group == "A_fresh" or r["annotation_id"] in changed_ids)]
            picked = sorted(ids, key=lambda aid: (digest(f"{seed}:{name}:{aid}".encode()), aid))[:5]
            strata[name] = {"eligible": len(ids), "selected": picked}
            for aid in picked:
                memberships.setdefault(aid, []).append(name)
    return dict(sorted(memberships.items())), strata


def compare(old=V1_OUTPUT, new=DEFAULT_OUTPUT, output=DEFAULT_REPORT, source=SGML_FILE):
    output = Path(output)
    require(not output.exists(), "Comparison output exists; use a new directory")
    v1, v2 = load_checked(old), load_checked(new)
    source_hash = file_hash(source)
    require(source_hash == v1["manifest"]["inputs"]["source"]["sha256"]
            == v2["manifest"]["inputs"]["source"]["sha256"], "Different source snapshots")
    require(set(v1["provenance"]) == set(v2["provenance"]), "Annotation inventory changed")
    docs = load_nucle(source)
    for sample in v2["samples"].values():
        record = v2["provenance"][sample["annotation_id"]]
        paragraph = docs[record["essay_index"]].paragraphs[record["start_par"]]
        check_alignment(sample, v2["sentences"][sample["sentence_id"]], record, paragraph)
    diffs, changed, retention = [], set(), []
    unchanged_fields = ("essay_id", "essay_index", "mistake_index", "start_par", "start_off", "end_par", "end_off", "original_label", "quality_flags", "review_status")
    for aid in sorted(v1["provenance"]):
        p1, p2 = v1["provenance"][aid], v2["provenance"][aid]
        require(all(p1[k] == p2[k] for k in unchanged_fields), f"Source metadata changed for {aid}")
        s1, s2 = v1["samples"].get(aid), v2["samples"].get(aid)
        row = {"annotation_id": aid, "label": p2["original_label"], "old_status": p1["status"], "new_status": p2["status"],
               "old_reason": p1["reason"], "new_reason": p2["reason"],
               "old_sentence_id": p1["sentence_id"], "new_sentence_id": p2["sentence_id"],
               "input_changed": False, "sentence_id_changed": p1["sentence_id"] != p2["sentence_id"]}
        if p1["status"] != p2["status"] or p1["reason"] != p2["reason"]:
            retention.append(aid)
        if s1 and s2:
            require(s1["error_span"] == s2["error_span"] and s1["error_type"] == s2["error_type"], "Source span or label changed")
            tokens1 = v1["sentences"][s1["sentence_id"]]["tokens"]
            tokens2 = v2["sentences"][s2["sentence_id"]]["tokens"]
            span1 = [(tokens1[i]["text"], tokens1[i]["start"] + p1["sentence_start_par"], tokens1[i]["end"] + p1["sentence_start_par"]) for i in s1["span_token_indices"]]
            span2 = [(tokens2[i]["text"], tokens2[i]["start"] + p2["sentence_start_par"], tokens2[i]["end"] + p2["sentence_start_par"]) for i in s2["span_token_indices"]]
            absolute_span1 = (p1["sentence_start_par"] + s1["error_start"], p1["sentence_start_par"] + s1["error_end"])
            absolute_span2 = (p2["sentence_start_par"] + s2["error_start"], p2["sentence_start_par"] + s2["error_end"])
            require(absolute_span1 == absolute_span2 and s1["error_span"] == s2["error_span"], f"Source span identity changed: {aid}")
            context1 = [[tokens1[i]["text"] for i in s1[k]] for k in ("left_token_indices", "right_token_indices")]
            context2 = [[tokens2[i]["text"] for i in s2[k]] for k in ("left_token_indices", "right_token_indices")]
            row.update(input_changed=s1["input_sha256"] != s2["input_sha256"],
                       context_changed=context1 != context2, old_sentence=s1["sentence"], new_sentence=s2["sentence"],
                       old_error_start=s1["error_start"], old_error_end=s1["error_end"],
                       new_error_start=s2["error_start"], new_error_end=s2["error_end"],
                       old_token_count=len(tokens1), new_token_count=len(tokens2), span_tokens_unchanged=(span1 == span2))
            if row["input_changed"]:
                changed.add(aid)
        if row["input_changed"] or row["sentence_id_changed"] or aid in retention:
            diffs.append(row)
    old_review_path = DATA_ROOT / "interim" / "nucle_boundary_decisions.jsonl"
    old_reviews = read_rows(old_review_path)
    previous = {r["annotation_id"] for r in old_reviews}
    memberships, strata = fresh_selection(list(v2["samples"].values()), changed, previous)
    packet = []
    for aid, groups in memberships.items():
        sample, record = v2["samples"][aid], v2["provenance"][aid]
        sentence = v2["sentences"][sample["sentence_id"]]
        paragraph = docs[record["essay_index"]].paragraphs[record["start_par"]]
        packet.append({**sample, "memberships": groups, "paragraph": paragraph,
                       "sentence_start_par": record["sentence_start_par"], "sentence_end_par": record["sentence_end_par"],
                       "old_sentence": v1["samples"].get(aid, {}).get("sentence"), "input_changed": aid in changed,
                       "span_tokens": [sentence["tokens"][i]["text"] for i in sample["span_token_indices"]],
                       "left_tokens": [sentence["tokens"][i]["text"] for i in sample["left_token_indices"]],
                       "right_tokens": [sentence["tokens"][i]["text"] for i in sample["right_token_indices"]],
                       "source_sha256": source_hash, "dataset_version": v2["version"]})
    regressions = [{"annotation_id": r["annotation_id"], "previous_boundary_decision": r["boundary"],
                    "issue_type": r["issue_type"], "input_changed": r["annotation_id"] in changed,
                    "old_sentence": v1["samples"][r["annotation_id"]]["sentence"],
                    "new_sentence": v2["samples"].get(r["annotation_id"], {}).get("sentence")}
                   for r in old_reviews]
    summary = {"source_sha256": source_hash, "v1_version": v1["version"], "v2_version": v2["version"],
               "old_samples": len(v1["samples"]), "new_samples": len(v2["samples"]),
               "old_sentences": len(v1["sentences"]), "new_sentences": len(v2["sentences"]),
               "input_changed": len(changed), "input_unchanged": len(set(v1["samples"]) & set(v2["samples"])) - len(changed),
               "changed_per_label": dict(Counter(v2["samples"][aid]["error_type"] for aid in changed)),
               "context_changed": sum(r.get("context_changed", False) for r in diffs),
               "sentence_id_changed": sum(r["sentence_id_changed"] for r in diffs),
               "retention_changed_ids": retention, "mechanical_checked": len(v2["samples"]),
               "span_source_offsets_unchanged": True,
               "span_tokenization_changed": sum(not r.get("span_tokens_unchanged", True) for r in diffs),
               "source_metadata_unchanged": True,
               "fresh_review_count": len(packet)}
    require(file_hash(source) == source_hash, "Source changed during comparison")
    output.mkdir(parents=True)
    for name, rows in (("diff", diffs), ("regression_cases", regressions), ("fresh_packet", packet)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    write_json(output / "comparison_summary.json", summary)
    write_json(output / "fresh_selection.json", {"seed": FRESH_SEED, "strata": strata,
                                                "excluded_reviewed_ids": sorted(previous), "packet_sha256": file_hash(output / "fresh_packet.jsonl"),
                                                "v1_manifest_sha256": file_hash(Path(old) / "manifest.json"),
                                                "v2_manifest_sha256": file_hash(Path(new) / "manifest.json"),
                                                "old_decisions_sha256": file_hash(old_review_path)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=V1_OUTPUT)
    parser.add_argument("--new", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(compare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
