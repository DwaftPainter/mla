# NUCLE Data Understanding Report

## 1. Dataset overview

- Total annotation rows: **44,912**
- Unique essays represented in annotations: **1,397**
- Unique error types: **28**
- Target-label rows: **15,788**
- Invalid offsets: **0**
- Cross-paragraph annotations: **4**

## 2. Five target classes

| error_type | count | percentage | essays_with_class |
| --- | --- | --- | --- |
| ArtOrDet | 6640 | 42.06 | 1178 |
| Prep | 2413 | 15.28 | 954 |
| Nn | 3768 | 23.87 | 1027 |
| Vform | 1443 | 9.14 | 690 |
| SVA | 1524 | 9.65 | 746 |

## 3. Edit-type distribution inside the five-class subset

| edit_type | count |
| --- | --- |
| replacement | 12821 |
| deletion | 2967 |

## 4. Target errors per essay

- Essays containing at least one target error: **1,355**
- Mean errors per essay: **11.65**
- Median errors per essay: **9.00**
- Minimum: **1**
- Maximum: **62**

## 5. Span-length diagnostics

Character-length statistics for valid, single-paragraph target spans only.
Excluded target rows: **0**.
Empty strings are measured only when reconstruction succeeded:

- Mean: **5.86**
- Median: **5.00**
- Minimum: **1**
- Maximum: **156**
- Empty spans: **0**

## 6. Duplicate diagnostics

Potential duplicate target annotations using:
`essay_id + offsets + error_type + correction`

- Rows participating in a duplicate group: **0**

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
