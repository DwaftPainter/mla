# NUCLE Data Quality

- Documents: **1,397**; annotations: **44,912**; target: **15,788**.
- CSV vs SGML: **match**.
- Overlap pairs: **455** (pairs, not rows).
- Duplicate content groups: {"paragraph_exact": 65, "paragraph_normalized": 62}.

| Quality flag | All annotations | Target annotations |
| --- | ---: | ---: |
| invalid_offset | 0 | 0 |
| cross_paragraph | 4 | 0 |
| missing_label | 0 | 0 |
| missing_correction | 0 | 0 |
| no_op | 30 | 6 |
| empty_edit | 0 | 0 |
| long_span | 387 | 2 |
| same_span_different_label | 47 | 11 |
| duplicate_candidate | 0 | 0 |
| overlapping_span | 531 | 170 |

Each annotation remains a separate sample with one label. Overlap, same-span different labels
and duplicate candidates require review; no rows are removed or merged by this audit.
Overlap uses half-open character spans in a valid single paragraph; touching spans do not overlap.
Insertion points strictly inside a span or at the same insertion position are flagged for review.
The CSV audit preserves annotation order. Source text and offsets are not normalized.
Detailed flags and related annotation IDs are in reports/nucle_annotation_audit.csv.
Content duplicates are in reports/nucle_duplicate_content.json; these are not split groups.
Missing labels/offset errors and no-op corrections need source/context review.
Short boilerplate, near duplicates and semantic label correctness need manual review.
There is no feature engineering, POS baseline, sentence splitting or train/validation/test split.
