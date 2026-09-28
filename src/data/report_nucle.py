from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/nucle_errors.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports/nucle_data_understanding.md"

TARGET_LABELS = [
    "ArtOrDet",
    "Prep",
    "Nn",
    "Vform",
    "SVA",
]


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

    duplicate_mask = target_df.duplicated(
        subset=[
            "essay_id",
            "start_par",
            "start_off",
            "end_par",
            "end_off",
            "error_type",
            "correction",
        ],
        keep=False,
    )

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

## 6. Duplicate diagnostics

Potential duplicate target annotations using:
`essay_id + offsets + error_type + correction`

- Rows participating in a duplicate group: **{int(duplicate_mask.sum()):,}**

## 7. Required manual checks before Data Preparation

1. Inspect examples from every target class.
2. Confirm reconstructed `error_span` matches the original paragraph.
3. Verify the exact NUCLE release being used.
4. Confirm that `Prep` already exists in this release before attempting any historical Wcip/Rloc mapping.
5. Review empty spans separately from failed reconstruction; zero-length spans can represent insertions.
   Edit types describe span/replacement text, not a linguistic missing-word classification.
6. Do not use `correction` as a prediction feature.
7. Do not create train/validation/test rows yet; split later by `essay_id`.
8. Do not normalize or sentence-tokenize paragraph text before offset validation.
"""

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown Data Understanding report from parsed NUCLE annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Parsed CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Markdown report path (default: {DEFAULT_REPORT})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Parsed CSV not found: {args.input}\n"
            "Run `uv run python -m src.data.parse_nucle` first."
        )

    df = pd.read_csv(
        args.input,
        keep_default_na=False,
    )

    # CSV round-trip can turn booleans into strings depending on tooling.
    for col in [
        "is_target_label",
        "is_insertion",
        "cross_paragraph",
        "valid_offset",
    ]:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )

    report = build_report(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
