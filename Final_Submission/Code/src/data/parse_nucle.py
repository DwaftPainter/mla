from __future__ import annotations

import pandas as pd

if __package__:
    from .nucle_sgml import NucleDocument, NucleMistake, load_nucle
    from .constants import ANNOTATIONS_FILE, SGML_FILE, TARGET_LABELS
else:
    # Direct execution: python src/data/parse_nucle.py.
    from nucle_sgml import NucleDocument, NucleMistake, load_nucle
    from constants import ANNOTATIONS_FILE, SGML_FILE, TARGET_LABELS

DEFAULT_INPUT = SGML_FILE
DEFAULT_OUTPUT = ANNOTATIONS_FILE

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value) # delete strip()

def classify_edit(error_span: str | None, correction: str, missing_correction: bool = False) -> str:
    if pd.isna(error_span):
        return "invalid_offset"

    if missing_correction:
        return "missing_correction"

    original = normalize_text(error_span)
    replacement = normalize_text(correction)

    if original == "" and replacement != "":
        return "insertion"

    if original != "" and replacement == "":
        return "deletion"

    if original != "" and replacement != "":
        return "replacement"

    return "empty"


def reconstruct_span(
    doc: NucleDocument,
    mistake: NucleMistake,
) -> tuple[str | None, str | None, bool, bool]:
    """
    Returns:
        paragraph_text
        error_span
        valid_offset
        cross_paragraph
    """

    sp, ep = mistake.start_par, mistake.end_par
    so, eo = mistake.start_off, mistake.end_off
    cross_paragraph = sp != ep
    if not (0 <= sp <= ep < len(doc.paragraphs)):
        return None, None, False, cross_paragraph
    if not (0 <= so <= len(doc.paragraphs[sp])
            and 0 <= eo <= len(doc.paragraphs[ep])):
        return None, None, False, cross_paragraph
    if cross_paragraph:
        # Valid endpoints, but no single-paragraph span is reconstructed.
        return None, None, True, True
    paragraph = doc.paragraphs[sp]
    if so > eo:
        return paragraph, None, False, False
    return paragraph, paragraph[so:eo], True, False



def documents_to_dataframe(
    docs: list[NucleDocument],
) -> pd.DataFrame:
    rows: list[dict] = []

    annotation_id = 0

    for doc_index, doc in enumerate(docs):
        for mistake_index, mistake in enumerate(doc.mistakes):
            (
                paragraph,
                error_span,
                valid_offset,
                cross_paragraph,
            ) = reconstruct_span(doc, mistake)

            is_insertion = (
                not cross_paragraph
                and valid_offset
                and mistake.start_off == mistake.end_off
            )

            rows.append(
                {
                    "annotation_id": annotation_id,
                    "essay_index": doc_index,
                    "essay_id": doc.essay_id,
                    "mistake_index": mistake_index,

                    "start_par": mistake.start_par,
                    "start_off": mistake.start_off,
                    "end_par": mistake.end_par,
                    "end_off": mistake.end_off,

                    "paragraph": paragraph,
                    "error_span": error_span,
                    "error_type": mistake.error_type,
                    "correction": mistake.correction,
                    "comment": mistake.comment,
                    "missing_correction": mistake.missing_correction,
                    "edit_type": (
                        "invalid_offset" if not valid_offset else
                        "cross_paragraph" if cross_paragraph else
                        classify_edit(error_span, mistake.correction, mistake.missing_correction)
                    ),

                    "is_target_label": mistake.error_type in TARGET_LABELS,
                    "is_insertion": is_insertion,
                    "cross_paragraph": cross_paragraph,
                    "valid_offset": valid_offset,
                }
            )

            annotation_id += 1

    columns = [
        "annotation_id", "essay_index", "essay_id", "mistake_index",
        "start_par", "start_off", "end_par", "end_off", "paragraph",
        "error_span", "error_type", "correction", "comment", "missing_correction",
        "edit_type", "is_target_label", "is_insertion", "cross_paragraph", "valid_offset",
    ]
    df = pd.DataFrame(rows, columns=columns)
    for column in ["missing_correction", "is_target_label", "is_insertion",
                   "cross_paragraph", "valid_offset"]:
        df[column] = df[column].astype(bool)
    return df
