from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .nucle_sgml import load_nucle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/nucle3.2.sgml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the raw NUCLE SGML structure before preprocessing."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to NUCLE SGML (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of annotation examples to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    docs = load_nucle(args.input)

    if not docs:
        raise RuntimeError("No <DOC> elements were parsed from the SGML file.")

    total_paragraphs = sum(len(doc.paragraphs) for doc in docs)
    total_annotations = sum(len(doc.mistakes) for doc in docs)

    label_counts = Counter(
        mistake.error_type
        for doc in docs
        for mistake in doc.mistakes
    )

    print("=" * 80)
    print("NUCLE RAW DATA INSPECTION")
    print("=" * 80)
    print(f"Input file          : {args.input}")
    print(f"Documents / essays  : {len(docs):,}")
    print(f"Text blocks (incl. titles): {total_paragraphs:,}")
    print(f"Annotations          : {total_annotations:,}")
    print(f"Unique error labels  : {len(label_counts):,}")

    first_doc = docs[0]

    print("\nFIRST DOCUMENT")
    print("-" * 80)
    print(f"essay_id             : {first_doc.essay_id}")
    print(f"paragraphs            : {len(first_doc.paragraphs)}")
    print(f"mistakes              : {len(first_doc.mistakes)}")

    if first_doc.paragraphs:
        print("\nFirst text block preview (title when present):")
        print(repr(first_doc.paragraphs[0][:500]))

    print("\nTOP ERROR LABELS")
    print("-" * 80)
    for label, count in label_counts.most_common(20):
        print(f"{label:15s} {count:>8,}")

    print("\nANNOTATION EXAMPLES")
    print("-" * 80)

    shown = 0

    for doc in docs:
        for mistake in doc.mistakes:
            if shown >= args.examples:
                break

            print(f"\nExample {shown + 1}")
            print(f"essay_id   : {doc.essay_id}")
            print(f"type       : {mistake.error_type}")
            print(
                "location   : "
                f"par {mistake.start_par}:{mistake.start_off}"
                f" -> par {mistake.end_par}:{mistake.end_off}"
            )
            print(f"correction : {repr(mistake.correction)}")

            if (
                mistake.start_par == mistake.end_par
                and 0 <= mistake.start_par < len(doc.paragraphs)
            ):
                paragraph = doc.paragraphs[mistake.start_par]

                if (
                    0 <= mistake.start_off <= mistake.end_off <= len(paragraph)
                ):
                    error_span = paragraph[
                        mistake.start_off:mistake.end_off
                    ]
                    print(f"error_span : {repr(error_span)}")
                    print(f"paragraph  : {paragraph[:500]}")
                else:
                    print("error_span : <INVALID OFFSET>")
            else:
                print("error_span : <CROSS-PARAGRAPH ANNOTATION>")

            shown += 1

        if shown >= args.examples:
            break


if __name__ == "__main__":
    main()
