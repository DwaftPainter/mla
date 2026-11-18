# Week 11: training results and completed error review

Updated 1 November 2026. The team has reviewed and approved the 18 validation case reviews, 40 BERT/LR case reviews, and 22 consolidated review decisions. This report adds results and interpretations beyond the earlier report. Gold labels, source offsets, model predictions, and benchmark scores remain unchanged.

## BERT now learns the training task

The extension compared CLS with the mean representation of the marked error span. Encoder learning rate stayed at 5e-5; the classification head used 1e-4 or 1e-3. Each configuration used seeds 20261013, 20261014, and 20261015, with up to 20 epochs and early-stopping patience 4. Each run selected its checkpoint by validation Macro-F1; configuration selection used the three-seed validation mean.

Train and validation scores below come from the same selected checkpoint in each run. The reported standard deviation is the sample standard deviation across three seeds, not a confidence interval.

| Readout | Encoder LR | Head LR | Mean train Macro-F1 | Validation Macro-F1 | Selected epochs, in seed order |
| --- | --- | --- | ---: | --- | --- |
| CLS | 5e-5 | 1e-4 | 0.8108 | 0.7164 ± 0.0061 | 20 / 19 / 15 |
| CLS | 5e-5 | 1e-3 | 0.8458 | 0.7227 ± 0.0147 | 20 / 13 / 17 |
| Span mean | 5e-5 | 1e-4 | 0.9278 | 0.8183 ± 0.0011 | 18 / 14 / 14 |
| Span mean | 5e-5 | 1e-3 | 0.9225 | 0.8177 ± 0.0022 | 8 / 10 / 9 |

All 12 selected checkpoints exceed train Macro-F1 0.70. The primary span-mean run reaches train 0.9309 and validation 0.8173 at epoch 18, resolving the earlier severe underfitting. The remaining train–validation gap is substantial. Both training settings and readout changed between phases, so the entire improvement cannot be attributed to span mean alone.

The executed grid differs from the suggested 1e-4/3e-4/1e-3, five-to-ten-epoch recipe: no 3e-4 run exists, and encoder LR stayed at 5e-5. Train and validation were evaluated each epoch; train F1 above 0.70 was not implemented as a gate before validation inspection. The table reports what was actually run.

## Comparison on the existing difficult validation subsets

These are the same gold-label pairs and exact POS-signature subsets used earlier. Predictions remain five-class. Pair F1 averages the two fixed gold-class F1 values after filtering; an out-of-pair prediction remains an error.

| Validation subset | Samples | LR Full | BERT primary | BERT mean ± sample std |
| --- | ---: | ---: | ---: | --- |
| Vform / SVA | 445 | 0.7417 | 0.7695 | 0.7726 ± 0.0061 |
| Vform / SVA, exact VERB signature | 243 | 0.7594 | 0.7582 | 0.7616 ± 0.0049 |
| Vform / SVA, exact AUX signature | 98 | 0.7216 | 0.7347 | 0.7250 ± 0.0086 |
| Nn / ArtOrDet | 1,561 | 0.8869 | 0.8909 | 0.8907 ± 0.0034 |
| Nn / ArtOrDet, exact NOUN signature | 631 | 0.8445 | 0.8591 | 0.8655 ± 0.0083 |
| Nn / ArtOrDet, exact DET signature | 353 | 0.5524 | 0.4891 | 0.5102 ± 0.0365 |

Primary BERT improves the full Vform/SVA pair by about 0.0277, but is nearly tied with LR on the exact-VERB subset. Its exact-DET result is about 0.0632 lower. Contextual representations therefore do not resolve every same-POS confusion. These are descriptive subset comparisons, without separate significance tests.

## Test results and execution order

All systems below use the same 2,368 test annotations. Tree S+L+P+M is included explicitly rather than being replaced by the Tree variant with the higher test score. Full includes structural, lexical, POS, morphology, and dependency features.

| System | Validation Macro-F1 | Test Macro-F1 | Test accuracy | Test Macro-F1 CI95% |
| --- | ---: | ---: | ---: | --- |
| Majority | 0.1184 | 0.1184 | 0.4206 | [0.1104, 0.1279] |
| POS rule | 0.7048 | 0.7106 | 0.7441 | [0.6766, 0.7365] |
| LR Full | 0.8067 | 0.8163 | 0.8357 | [0.7876, 0.8382] |
| Tree S+L+P+M | 0.7929 | 0.7967 | 0.8121 | [0.7674, 0.8201] |
| Tree Full | 0.7879 | 0.7996 | 0.8095 | [0.7693, 0.8246] |
| BERT-small, primary seed | 0.8173 | 0.8146 | 0.8438 | [0.7919, 0.8350] |

Primary BERT minus LR is -0.0017, paired CI95% [-0.0184, 0.0182]. This does not establish either model as superior. Three-seed BERT test mean is 0.8182 ± 0.0038; it is not an ensemble score or a replacement for the fixed primary comparison. BERT improves Nn F1 to 0.8098 versus LR's 0.7781, but its SVA F1 is lower: 0.7392 versus 0.8106.

The classical lock was recorded at 08:39:33 UTC on 23 October 2026; its test-scoring start was recorded nine seconds later. The BERT extension was reported on 28 October, after the classical test had been seen. It reused that test rather than an independent external test. Its selection file has a hash chain but no independent wall-clock lock timestamp; no extra timestamp is inferred.

Confidence intervals use 1,000 paired bootstrap replicates over 97 connected test groups. They are pointwise and do not capture training-seed or split variation. This update used saved outputs, without retraining or scoring test again.

## Why 18 validation cases were reviewed

LR Full misclassifies **364 of 2,368 validation annotations (15.37%)** and correctly classifies 2,004. A mismatch with gold does not itself establish that the source annotation is wrong.

| Gold → prediction | Validation errors | Initially selected |
| --- | ---: | ---: |
| Vform → SVA | 40 | 3 |
| SVA → Vform | 35 | 3 |
| Nn → ArtOrDet | 94 | 3 |
| ArtOrDet → Nn | 44 | 3 |

Selection took the first three sorted annotation IDs in each direction, then six additional errors with multi-token spans. The 18 cases cover about 4.95% of the validation errors. This is purposeful qualitative review, not a statistically representative prevalence sample.

The team reviewed and approved all 18 interpretations. The assignments below apply only to this validation audit, not the whole project.

| Member | Cases | Assigned annotation IDs |
| --- | ---: | --- |
| Quân | 6 | 10824, 11696, 12352, 11501, 12050, 12301 |
| Duy | 6 | 12314, 14761, 14764, 12019, 14939, 15390 |
| Công | 6 | 1269, 18774, 19065, 19068, 19069, 19185 |

All 18 spans and gold labels match the source. Saved alignment is exact in 17 cases; one has trailing whitespace. Reading neighboring edits explains why isolated targets can be misleading: 15390 depends on an adjacent modal deletion, and 19068 combines agreement and participle changes alongside a subject-number edit. These findings do not justify changing gold after viewing predictions.

## Completed BERT/LR review and consolidated decisions

The team has reviewed and approved all 40 BERT/LR case reviews and all 22 consolidated review decisions, confirmed on 1 November 2026.

| BERT/LR sample group | Cases |
| --- | ---: |
| BERT wrong, LR correct | 12 |
| BERT correct, LR wrong | 12 |
| Both wrong | 10 |
| Both correct, controls | 6 |

The first bucket prioritizes six gold-SVA cases. The both-wrong sample contains nine ArtOrDet and one Nn due to bucket traversal; these proportions do not estimate corpus prevalence.

The consolidated list combines 17 earlier IDs and six BERT priorities, with one overlap. Eight of its 22 IDs also belong to the 40-case sample, giving **54 distinct source-checked annotations** in this extended review.

| Approved interpretation or action | Cases | Annotation IDs |
| --- | ---: | --- |
| Existing label supported by the edit and context | 9 | 18531, 37631, 24526, 9351, 41235, 35455, 211, 37624, 32801 |
| Reconsider label/edit correspondence | 6 | 34539, 12324, 26635, 32705, 37909, 16809 |
| Check correction or replacement boundaries | 3 | 18501, 23037, 30366 |
| Edit may be optional | 3 | 201, 11916, 6079 |
| Multiple operations in one edit | 1 | 32874 |

Neighboring annotations explain the initially questionable number edit at 32801 and the agreement edit at 18531. Case 23037 has a plausible number-edit intention but a questionable span/replacement pairing. Cases 18501 and 30366 raise concerns about the resulting construction. Cases 24526, 9351, 41235, and 35455 contain specific POS/dependency concerns, without an intervention establishing effects on model predictions.

Approval records the team's accepted interpretations and actions to investigate. It does not silently relabel data, certify every correction as obligatory, or resolve every flagged annotation. A correct prediction against gold can coexist with a questionable correction. The reviews support descriptive explanations and future hypotheses, not causal claims.

## Evidence and public scope

The training table uses the 12 saved run summaries; subset comparisons use aligned validation predictions. Headline results are available in [overall_results.csv](../overall_results.csv), paired comparisons in [paired_differences.csv](../paired_differences.csv), and classical variants in [classical_ablations.csv](../classical_ablations.csv). See the [complete report](final-report.md) for the broader study and reproducibility limits.

The research workspace retains source sentences, full corrections, row-level predictions, review records, and historical locks. These are excluded from this public portfolio. Only aggregate findings and annotation IDs are presented here. No generation scripts, private review packets, PDF reports, or task checkpoints are added by this update.
