from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


if __package__:
    from .constants import ANNOTATIONS_FILE, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from .nucle_sgml import load_nucle
    from .parse_nucle import documents_to_dataframe
else:
    from constants import ANNOTATIONS_FILE, REPORTS_DIR, SGML_FILE, TARGET_LABELS
    from nucle_sgml import load_nucle
    from parse_nucle import documents_to_dataframe

DEFAULT_INPUT = SGML_FILE
DEFAULT_OUTPUT = ANNOTATIONS_FILE
DEFAULT_REPORT = REPORTS_DIR / "nucle_data_understanding.md"


def markdown_table(df: pd.DataFrame) -> str:
    # Avoid requiring the optional `tabulate` dependency used by DataFrame.to_markdown().
    columns = list(df.columns)

    rows = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join(str(row[c]) for c in columns)
            + " |"
        )

    return "\n".join(rows)


def build_report(df: pd.DataFrame) -> str:
    target_df = df[df["error_type"].isin(TARGET_LABELS)].copy()

    class_counts = (
        target_df["error_type"]
        .value_counts()
        .reindex(TARGET_LABELS)
        .fillna(0)
        .astype(int)
    )

    class_pct = (
        target_df["error_type"]
        .value_counts(normalize=True)
        .reindex(TARGET_LABELS)
        .fillna(0)
        .mul(100)
        .round(2)
    )

    essays_per_class = (
        target_df.groupby("error_type")["essay_id"]
        .nunique()
        .reindex(TARGET_LABELS)
        .fillna(0)
        .astype(int)
    )

    class_summary = pd.DataFrame(
        {
            "error_type": TARGET_LABELS,
            "count": [class_counts[x] for x in TARGET_LABELS],
            "percentage": [class_pct[x] for x in TARGET_LABELS],
            "essays_with_class": [essays_per_class[x] for x in TARGET_LABELS],
        }
    )

    edit_counts = (
        target_df["edit_type"]
        .value_counts()
        .rename_axis("edit_type")
        .reset_index(name="count")
    )

    errors_per_essay = target_df.groupby("essay_id").size()

    reconstructed = target_df[
        target_df["valid_offset"] & ~target_df["cross_paragraph"]
        & target_df["error_span"].notna()
    ]
    span_lengths = (
        reconstructed["error_span"]
        .astype(str)
        .str.len()
    )

    report = f"""# NUCLE Data Understanding Report

## 1. Dataset overview

- Total annotation rows: **{len(df):,}**
- Unique essays represented in annotations: **{df["essay_id"].nunique():,}**
- Unique error types: **{df["error_type"].nunique():,}**
- Target-label rows: **{len(target_df):,}**
- Invalid offsets: **{int((~df["valid_offset"]).sum()):,}**
- Cross-paragraph annotations: **{int(df["cross_paragraph"].sum()):,}**

## 2. Five target classes

{markdown_table(class_summary)}

## 3. Edit-type distribution inside the five-class subset

{markdown_table(edit_counts)}

## 4. Target errors per essay

- Essays containing at least one target error: **{target_df["essay_id"].nunique():,}**
- Mean errors per essay: **{errors_per_essay.mean():.2f}**
- Median errors per essay: **{errors_per_essay.median():.2f}**
- Minimum: **{int(errors_per_essay.min()) if len(errors_per_essay) else 0}**
- Maximum: **{int(errors_per_essay.max()) if len(errors_per_essay) else 0}**

## 5. Span-length diagnostics

Character-length statistics for valid, single-paragraph target spans only.
Excluded target rows: **{len(target_df) - len(reconstructed):,}**.
Empty strings are measured only when reconstruction succeeded:

- Mean: **{span_lengths.mean():.2f}**
- Median: **{span_lengths.median():.2f}**
- Minimum: **{int(span_lengths.min()) if len(span_lengths) else 0}**
- Maximum: **{int(span_lengths.max()) if len(span_lengths) else 0}**
- Empty spans: **{int((span_lengths == 0).sum()):,}**

## 6. Scope

Each annotation is one observation with its own error label; counts are not independent sentences.
This step preserves every annotation and original offsets, including rows requiring quality review.
Run data_quality next to audit schema, missing data, overlaps and duplicate content.
No sentence splitting, POS, feature extraction, cleaning or dataset splits are performed.
"""

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: inspect NUCLE and export all annotations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    documents = load_nucle(args.input)
    if not documents:
        raise ValueError("No NUCLE documents parsed")
    frame = documents_to_dataframe(documents)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv, index=False)
    report = build_report(frame)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved all {len(frame):,} annotations to {args.csv}")


if __name__ == "__main__":
    main()
