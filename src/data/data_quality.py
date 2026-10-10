"""Step 2: audit NUCLE quality without cleaning, feature extraction or splitting."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

if __package__:
    from .constants import ANNOTATIONS_FILE, DEFAULT_RELEASE, DOCS_DIR, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from .nucle_sgml import load_nucle
    from .parse_nucle import documents_to_dataframe
else:
    from constants import ANNOTATIONS_FILE, DEFAULT_RELEASE, DOCS_DIR, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from nucle_sgml import load_nucle
    from parse_nucle import documents_to_dataframe


def annotation_review(frame):
    """Keep one row per annotation; all relationships below are review candidates."""
    # Use source document position, so duplicate/missing essay IDs do not join documents.
    location = ["essay_index", "start_par", "start_off", "end_par", "end_off"]
    identity = location + ["error_type", "correction", "missing_correction", "comment"]
    labels = frame.groupby(location).error_type.transform("nunique").gt(1)
    duplicate_candidates = frame.duplicated(identity, keep=False)
    overlaps = {int(value): [] for value in frame.annotation_id}
    single = frame[frame.valid_offset & ~frame.cross_paragraph]
    for _, group in single.groupby(["essay_index", "start_par"]):
        active = []
        for row in group.sort_values(["start_off", "end_off", "annotation_id"]).itertuples():
            active = [previous for previous in active if previous.end_off >= row.start_off]
            for previous in active:
                intersection = max(previous.start_off, row.start_off) < min(previous.end_off, row.end_off)
                same_point = previous.start_off == previous.end_off == row.start_off == row.end_off
                point_inside = (
                    previous.start_off < row.start_off < previous.end_off
                    and row.start_off == row.end_off
                ) or (
                    row.start_off < previous.start_off < row.end_off
                    and previous.start_off == previous.end_off
                )
                if intersection or same_point or point_inside:
                    overlaps[int(row.annotation_id)].append(int(previous.annotation_id))
                    overlaps[int(previous.annotation_id)].append(int(row.annotation_id))
            active.append(row)

    columns = ["annotation_id", "essay_index", "essay_id", "start_par", "start_off",
               "end_par", "end_off", "error_type", "is_target_label"]
    audit = frame[columns].copy()
    audit["invalid_offset"] = ~frame.valid_offset
    audit["cross_paragraph"] = frame.cross_paragraph
    audit["missing_label"] = frame.error_type.str.strip().eq("")
    audit["missing_correction"] = frame.missing_correction
    audit["no_op"] = frame.valid_offset & ~frame.cross_paragraph & ~frame.missing_correction & frame.error_span.eq(frame.correction)
    audit["empty_edit"] = frame.edit_type.eq("empty")
    audit["long_span"] = frame.error_span.fillna("").str.len().gt(100)
    audit["same_span_different_label"] = labels
    audit["duplicate_candidate"] = duplicate_candidates
    audit["overlap_annotation_ids"] = [json.dumps(sorted(overlaps[int(value)])) for value in frame.annotation_id]
    audit["overlapping_span"] = [bool(overlaps[int(value)]) for value in frame.annotation_id]
    flag_columns = list(audit.columns[len(columns):-2]) + ["overlapping_span"]
    audit["review_reasons"] = [
        ";".join(name for name in flag_columns if row[name])
        for _, row in audit.iterrows()
    ]
    audit["review_status"] = audit.review_reasons.map(lambda value: "pending" if value else "not_flagged")
    return audit, flag_columns


def duplicate_content(documents):
    """Report exact/normalized essay and paragraph duplicates; never merge groups."""
    buckets = defaultdict(list)
    for index, doc in enumerate(documents):
        units = [("essay", -1, "\n".join(doc.paragraphs))]
        units += [("paragraph", number, text) for number, text in enumerate(doc.paragraphs)]
        for scope, paragraph_index, text in units:
            if not text.strip():
                continue
            normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
            for kind, content in (("exact", text), ("normalized", normalized)):
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                buckets[(scope, kind, digest)].append({
                    "essay_index": index, "essay_id": doc.essay_id,
                    "paragraph_index": paragraph_index,
                })
    return [
        {"scope": scope, "kind": kind, "sha256": digest, "locations": locations}
        for (scope, kind, digest), locations in sorted(buckets.items())
        if len({location["essay_index"] for location in locations}) > 1
    ]


def compare_csv(frame, path):
    if not Path(path).is_file():
        return {"status": "missing", "message": "Run data_understanding to export the annotation CSV."}
    try:
        cached = pd.read_csv(path, keep_default_na=False, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        return {"status": "unreadable", "message": str(exc)}
    fresh = frame.fillna("").astype(str)
    missing = sorted(set(fresh.columns) - set(cached.columns))
    extra = sorted(set(cached.columns) - set(fresh.columns))
    common = sorted(set(fresh.columns) & set(cached.columns))
    mismatch = None
    if len(cached) == len(fresh):
        mismatch = int(cached[common].ne(fresh[common]).to_numpy().sum())
    return {
        "status": "match" if not missing and not extra and len(cached) == len(fresh) and mismatch == 0 else "mismatch",
        "expected_rows": len(fresh), "actual_rows": len(cached),
        "missing_columns": missing, "extra_columns": extra,
        "column_order_matches": list(cached.columns) == list(fresh.columns),
        "different_cells": mismatch,
    }


def audit_quality(documents, frame, csv_path):
    audit, flags = annotation_review(frame)
    duplicates = duplicate_content(documents)
    ids = [doc.essay_id for doc in documents]
    paragraphs = [text for doc in documents for text in doc.paragraphs]
    counts = frame.loc[frame.is_target_label, "error_type"].value_counts().reindex(TARGET_LABELS, fill_value=0)
    positive = counts[counts.gt(0)]
    summary = {
        "annotations": len(frame), "target_annotations": int(frame.is_target_label.sum()),
        "documents": len(documents), "text_blocks": len(paragraphs),
        "csv_comparison": compare_csv(frame, csv_path),
        "missing_essay_ids": sum(not value.strip() for value in ids),
        "duplicate_essay_id_rows": sum(count for value, count in Counter(ids).items() if value and count > 1),
        "documents_without_text": sum(not doc.paragraphs for doc in documents),
        "documents_without_annotations": sum(not doc.mistakes for doc in documents),
        "empty_text_blocks": sum(not text.strip() for text in paragraphs),
        "text_blocks_with_replacement_character": sum("\ufffd" in text for text in paragraphs),
        "text_blocks_with_non_ascii": sum(not text.isascii() for text in paragraphs),
        "text_blocks_with_control_characters": sum(bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text)) for text in paragraphs),
        "flag_counts": {name: int(audit[name].sum()) for name in flags},
        "target_flag_counts": {name: int(audit.loc[audit.is_target_label, name].sum()) for name in flags},
        "overlap_pairs": sum(len(json.loads(value)) for value in audit.overlap_annotation_ids) // 2,
        "duplicate_content_groups": dict(Counter(item["scope"] + "_" + item["kind"] for item in duplicates)),
        "target_class_counts": {label: int(count) for label, count in counts.items()},
        "missing_target_classes": counts[counts.eq(0)].index.tolist(),
        "class_imbalance_ratio": float(positive.max() / positive.min()) if len(positive) else None,
        "policy": "Audit only: keep every annotation and original label; no automatic removal, merging, features or splits.",
        "limitations": "Flags require manual annotation review; hashes do not detect paraphrase; structure checks do not validate linguistic correctness.",
    }
    return summary, audit, duplicates


def build_report(summary):
    rows = ["| Quality flag | All annotations | Target annotations |",
            "| --- | ---: | ---: |"]
    rows += [f"| {name} | {count:,} | {summary['target_flag_counts'][name]:,} |"
             for name, count in summary["flag_counts"].items()]
    return "\n".join([
        "# NUCLE Data Quality", "",
        f"- Documents: **{summary['documents']:,}**; annotations: **{summary['annotations']:,}**; target: **{summary['target_annotations']:,}**.",
        f"- CSV vs SGML: **{summary['csv_comparison']['status']}**.",
        f"- Overlap pairs: **{summary['overlap_pairs']:,}** (pairs, not rows).",
        f"- Duplicate content groups: {json.dumps(summary['duplicate_content_groups'], sort_keys=True)}.",
        "", *rows, "",
        "Each annotation remains a separate sample with one label. Overlap, same-span different labels",
        "and duplicate candidates require review; no rows are removed or merged by this audit.",
        "Overlap uses half-open character spans in a valid single paragraph; touching spans do not overlap.",
        "Insertion points strictly inside a span or at the same insertion position are flagged for review.",
        "The CSV audit preserves annotation order. Source text and offsets are not normalized.",
        "Detailed flags and related annotation IDs are in reports/nucle_annotation_audit.csv.",
        "Content duplicates are in reports/nucle_duplicate_content.json; these are not split groups.",
        "Missing labels/offset errors and no-op corrections need source/context review.",
        "Short boilerplate, near duplicates and semantic label correctness need manual review.",
        "There is no feature engineering, POS baseline, sentence splitting or train/validation/test split.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SGML_FILE)
    parser.add_argument("--csv", type=Path, default=ANNOTATIONS_FILE)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--report", type=Path, default=DOCS_DIR / "nucle_data_quality.md")
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    args = parser.parse_args()
    documents = load_nucle(args.input)
    frame = documents_to_dataframe(documents)
    summary, audit, duplicates = audit_quality(documents, frame, args.csv)
    raw = args.input.read_text(encoding="utf-8")
    summary["source_file"] = args.input.name
    summary["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    summary["release"] = args.release
    summary["source_tags"] = {
        tag: {"open": len(re.findall(r"<" + tag + r"\b", raw, re.I)),
              "close": len(re.findall(r"</" + tag + r"\s*>", raw, re.I))}
        for tag in ("DOC", "MISTAKE")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nucle_data_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit.to_csv(args.output_dir / "nucle_annotation_audit.csv", index=False)
    (args.output_dir / "nucle_duplicate_content.json").write_text(json.dumps(duplicates, indent=2), encoding="utf-8")
    args.report.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
