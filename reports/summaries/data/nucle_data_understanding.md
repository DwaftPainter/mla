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

## 6. Scope

Each annotation is one observation with its own error label; counts are not independent sentences.
This step preserves every annotation and original offsets, including rows requiring quality review.
Run data_quality next to audit schema, missing data, overlaps and duplicate content.
No sentence splitting, POS, feature extraction, cleaning or dataset splits are performed.
