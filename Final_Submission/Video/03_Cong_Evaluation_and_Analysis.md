# Công — Complete individual video script: evaluation and analysis

Project version: **NUCLE-final-20261117**. Spoken language: English. **No duration limit or target length.** Give the calculation, demonstration, interpretation and personal evidence their full explanation; follow the scenes without an artificial time budget.

All JSON/CSV basenames below refer to named records in Results/results.json. Use its json, tables or texts section, or the included evidence-view commands; these records are not separate files in the submission. Common setup, recording guidance and the personal-contribution checklist are in [README.md](../README.md).

## Recording brief and evidence boundaries

The assigned scope in new_task.md is test metrics, confusion matrices, group-bootstrap confidence intervals, model/ablation comparisons, the 18-validation/40-test audit, reproducibility and the report's conclusions and limitations. Required outputs: result tables/figures, error analysis and report.

Use **REAL SAVED EVIDENCE** for final experiment artifacts, **LIVE AGGREGATE RECALCULATION** when recomputing metrics from their confusion counts, and **SYNTHETIC LIVE EXECUTION** for the small group-resampling example. The synthetic example is not the research bootstrap. Complete your section of [personal-contribution checklist](../README.md#personal-contribution) before recording. Commands run from the submission root; see [submission README](../README.md) for the interpreter and preparation.

## Scene 1 — Define the evaluation question and your responsibility

**Show:** face camera, project version, the five labels and your responsibility row in the main README.

**Do:** draw known error span → predicted error type → comparison with original gold → metrics and audit. Separate evaluation from changing training or gold.

**Say:**

“Hello, I am Công. My assigned responsibility is test evaluation and interpretation: metrics and confusion matrices, grouped bootstrap intervals, model and ablation comparisons, validation and test case review synthesis, reproducibility checks, and the final conclusions and limitations.

“The question is how well we classify an already located error into ArtOrDet, Prep, Nn, Vform or SVA. This is not full grammatical correction. My demonstration will calculate a real class score, explain the paired comparison, follow a real error into the review evidence, and distinguish what the public package can reproduce from what needs restricted inputs.”

**Contribution boundary:** the assignment identifies the scope. Scene 14 must show a specific action you actually performed and the artifact supporting it.

## Scene 2 - Reproduce the main result and state its scope

**Run:**

```powershell
python Code/reproduce_main_result.py
```

**Show:** successful terminal output, then Results/reproduced/demo.html in a browser, main_results.csv, per_class_metrics.csv, checks.json and main_result.svg. Keep the command and output connected in the recording.

**Say:**

"This command calculates precision, recall, per-class F1, Macro-F1 and accuracy from saved confusion counts. It checks all twenty classical result names, twenty-five headline class records, twelve BERT validation runs and selected seed summaries. These are new arithmetic checks of saved evidence, not copied display values.

"Results/results.json stores the original experiment records under named keys. The separate main_results.csv is the readable saved headline table. Consolidation changes the number of files, not the observations or predictions.

"The generated HTML and chart display saved evidence. Expanding a table does not run checkpoint inference. This command does not train a predictor, restore private learner text or regenerate group-bootstrap intervals."

**Expected output:** show actual PASS output, LR Full 0.8162577160 and primary BERT 0.8145949582. A successful aggregate check is not full experimental reproduction.

## Scene 3 — Calculate Nn precision, recall and F1 from actual counts

**Run:**

```powershell
python Code/inspect_evidence.py metrics --model LR_03 --label Nn
```

**Show:** the printed matrix and fractions beside Results/results.json entry json["Results/evidence/test_confusions.json"]. Use label order ArtOrDet, Prep, Nn, Vform, SVA.

**Do:** outline the Nn gold row horizontally, the predicted Nn column vertically and diagonal cell 419. Add these callouts:

| Quantity | Real LR test value | Meaning |
| --- | --- | --- |
| True positives | 419 | Gold Nn and predicted Nn |
| Gold support | 565 | All gold Nn cases |
| Predicted Nn | 512 | All cases assigned Nn |
| False positives | 93 | Predicted Nn, different gold |
| False negatives | 146 | Gold Nn, different prediction |

**Say:**

“Rows are gold; columns are predictions. Nn precision is 419 divided by 512, or approximately 0.8184. Recall is 419 divided by 565, approximately 0.7416. False positives are 512 minus 419, which is ninety-three. False negatives are 565 minus 419, which is one hundred and forty-six.

“Class F1 is twice TP divided by twice TP plus FP plus FN. Here that is 838 divided by 1,077, giving 0.778087279. This is a calculation of one real result, not an illustrative percentage.

“We calculate F1 for each of the five labels and take their arithmetic mean for Macro-F1. Taking the harmonic mean of macro precision and macro recall would be a different operation. The implementation uses zero for undefined class scores, so Majority's unpredicted classes do not disappear from the five-class average.”

**Interpretation:** Nn recall is lower than precision. Do not infer the cause from this matrix alone; it identifies the direction and amount of error.

## Scene 4 — Explain the five-model chart and paired comparison

**Show:** Results/figures/benchmark_ci.png, Results/results.json entry tables["Results/classical/overall_results.csv"], classical/paired_differences.csv and bert/paired_differences.csv. Zoom the chart legend and comparison direction.

**Say:**

“The test Macro-F1 values are 0.1184 for Majority, 0.7106 for POS, 0.8163 for LR Full, 0.7996 for Tree Full and 0.8146 for primary BERT-small. This uses the same 2,368 test annotations in ninety-seven groups.

“LR minus POS is about 0.1057 with paired interval 0.0829 to 0.1288. LR minus Tree is about 0.0166 with interval 0.0018 to 0.0301. Both intervals are above zero in this saved evaluation. The BERT-minus-LR point difference is minus 0.0017, with interval minus 0.0184 to plus 0.0182. That comparison does not establish superiority or equivalence.

“The uncertainty for a difference is computed directly from paired replicates. I should not infer significance merely from whether two separate model intervals overlap, and I should not subtract the endpoints of those separate intervals.”

**Do:** place a zero reference line beside the paired intervals. Explain which direction favours which model before interpreting an endpoint. These are pointwise intervals, not a simultaneous guarantee for every comparison shown.

## Scene 5 — Run group resampling so the sampling unit is visible

**Run:**

```powershell
python Code/video_lab.py bootstrap
```

**Show:** Code/video_lab.py and group_bootstrap in Code/src/evaluation/bert_final_analysis.py; open Results/reproduced/video_lab/synthetic_group_resampling.json.

**Do:** draw three authored groups: A has two rows, B has one, C has three. Highlight each replicate's A/B/C counts and expanded row indices. Preserve the synthetic label.

**Say:**

“This is a live call to the project's group-bootstrap function on six authored examples. It uses three deliberately chosen illustrative replicates, not a random research run and not a confidence-interval estimate.

“The first replicate selects group A twice, B zero times and C once. That is three group draws but seven annotation rows. The next weights are zero, two and one, producing five rows. Weights one, one and one produce six. Group count stays fixed while annotation count varies because group sizes differ.

“All rows from a selected group move together. The same group weights are applied to both toy predictors. Macro-F1 is recomputed for each model and their difference is taken inside that replicate. The demonstration asserts that the weighted group calculation equals explicitly repeating the rows.

“If I sampled individual annotations independently, I would break this group structure. If I used different resamples for the two models, I would lose the paired comparison. Three selected replicates only explain the mechanics; I will now return to the saved research interval.”

**Expected output:** annotation totals 7, 5 and 6, expanded row lists and paired differences. Do not relabel toy predictors as benchmark scores or estimate a 95 percent interval from this demonstration.

## Scene 6 — State the actual research bootstrap and its limits

**Show:** Results/results.json entry json["Results/bert/bootstrap.json"] and Results/results.json entry json["Results/classical/bootstrap.json"]; then Results/results.json entry json["Results/evidence/split_summary.json"]. Highlight replicate count, group count, seed and metric.

**Say:**

“The saved test uncertainty uses 1,000 group-bootstrap replicates over ninety-seven connected groups with seed 20261023. The groups reflect related source content and can contain more than one essay. The fixed five-label Macro-F1 calculation assigns zero when a class is absent in a replicate. Paired comparisons share group multiplicities.

“These intervals describe this procedure's resampling uncertainty on the observed grouped test set. They do not include every source of uncertainty: new splits, new corpora, different annotation conventions or all training seeds. The sample standard deviation over three BERT seeds describes training-seed variation, not the same quantity as a group-bootstrap interval.

“The public aggregate confusion matrices have lost the row-to-group allocation. They can reproduce point scores, but not recreate the saved group-bootstrap distribution. That requires the authorised row-level predictions and group mapping. I will display the saved intervals with their provenance rather than claim to have regenerated them.”

**Interpretation:** one holdout with grouped uncertainty is not external validation. Avoid saying an interval containing zero proves that two predictors are equal.

## Scene 7 — Compare ablations without overclaiming causality

**Run:**

```powershell
python Code/inspect_evidence.py ablation
```

**Show:** Results/results.json entry tables["Results/classical/paired_differences.csv"]. Highlight the comparison row and pipeline IDs at the same time.

**Say:**

“For LR, Full minus target-span lexical masking is 0.0134, with paired interval 0.0049 to 0.0201. Full minus removing all L is 0.0281, with interval 0.0142 to 0.0394. The distinction matters: the first retains lexical context and the original parse, while the second removes the whole lexical family.

“Full minus P is about 0.0181, with interval 0.0069 to 0.0296. Full minus M is about 0.0040 with an interval containing zero, and Full minus D is about 0.0060, also containing zero. This does not prove morphology or dependency information is useless. Families overlap, each pipeline is tuned, and sample uncertainty remains.

“Tree does not reproduce every LR pattern. Its masked point score is slightly higher than Full and that paired interval includes zero. I should not generalize one model's result to every learner.

“Full minus D is the S+L+P+M alias, so duplicate names are not independent confirmations. These comparisons explain already specified experiments; I am not using them to choose a new model after opening the test set.”

**Do:** mark the alias and the intervals crossing zero. Keep the distinction between target masking and all-L removal visible.

## Scene 8 — Explain why accuracy and Macro-F1 rank models differently

**Show:** Results/figures/per_class_f1.png and confusion_comparison.png; run:

```powershell
python Code/inspect_evidence.py metrics --model BERT --label SVA
```

**Say:**

“BERT has higher accuracy: 0.8438 versus LR's 0.8357, corresponding to 1,998 versus 1,979 correct predictions. Yet its primary Macro-F1 is slightly lower. Macro-F1 gives equal weight to each class, not to each annotation.

“Nn F1 improves from LR's 0.7781 to BERT's 0.8098, while SVA falls from 0.8106 to 0.7392. That tradeoff is hidden by accuracy alone. The saved joint outcomes are 1,860 correct for both, 138 correct only for BERT, 119 correct only for LR and 251 wrong for both.

“LR's directional confusions also matter. There are 134 Nn cases predicted as ArtOrDet and fifty-nine in the reverse direction. Vform to SVA has twenty-nine; SVA to Vform has eighteen. Reversing row and column would reverse the explanation. These counts suggest where to inspect cases; they are not proof of a single grammatical cause.”

**Do:** stop on one gold row and one predicted column before switching charts. Do not read every cell aloud.

## Scene 9 — Trace a real assigned error into the audit

**Run:**

```powershell
python Code/video_lab.py case --annotation-id 19068
```

**Show:** Results/results.json entry json["Results/evidence/real_example_19068.json"] and Results/results.json entry texts["Results/reproducibility/team_review_record.md"]. Use a text-free annotated diagram: real validation ID → original interval → feature subset → saved decision → existing review note.

**Say:**

“This is real validation annotation 19068, one of the cases assigned to me. Its sentence-relative interval is ten to eighteen, eight characters, with exact alignment and two tokens. Gold is SVA; saved LR Full and Tree Full both predict Vform. The visible parse signature includes aux plus ROOT and singular-number features.

“The source example, offsets, feature subset and saved prediction belong to the actual final project. The learner sentence, correction and lexical coordinates are deliberately omitted. This command reads evidence; it does not infer from redacted metadata.

“The existing review describes a multi-operation interpretation involving subject number. I attribute that explanation to the review record. It is a reason to examine the annotation and correction boundary, not proof that the model's prediction is linguistically correct or that the parser caused the failure. Gold remains unchanged for scoring.”

**Interpretation:** explain the path all the way to the review conclusion. Do not substitute an authored readable sentence and call it the real case.

## Scene 10 — Explain how the review sets were selected

**Run:**

```powershell
python Code/inspect_evidence.py audit
```

**Show:** Results/results.json entry json["Results/evidence/audit_summary.json"]. Draw separate boxes for validation selection, the forty-case test review and consolidated decisions, with the documented overlap.

**Say:**

“The eighteen validation cases were purposefully selected from 364 LR validation errors: three sorted IDs for each of four directional confusions, plus six multi-token cases. Eighteen is approximately 4.95 percent of those errors. The selection gives useful coverage; it does not estimate error-category prevalence.

“The forty test cases include twelve correct only for LR, twelve correct only for BERT, ten wrong for both and six correct-for-both controls. The controls help us avoid a review containing only failures. These quotas are not a random sample of all test disagreements.

“The consolidated twenty-two decisions overlap the forty-case test review in eight annotations, producing fifty-four distinct annotations in that extended union. I should not add eighteen, forty and twenty-two and call the result a verified unique-case count.

“The saved 1 November team record documents accepted interpretations and follow-up actions. It is not independent expert adjudication and does not prove a new personal action by today's speaker. We do not edit gold after seeing predictions to raise the score.”

**Do:** label the 18-case validation set separately. Make the 40 + 22 − 8 = 54 relation explicit without implying it also counts the validation set.

## Scene 11 - Separate aggregate checks from historical replay

**Run:**

```powershell
python Code/inspect_evidence.py ablation
python Code/reproduce_main_result.py
```

**Show:** the actual command output and the historical_reproduction/current_lock_check records under json in Results/results.json. Distinguish present aggregate recalculation, recorded historical pipeline replay and a future authorised full restoration.

**Say:**

"The ablation view identifies saved aliases, and reproduction recalculates point metrics. The live bootstrap lab checked group weighting against explicit row expansion. These checks do not establish that every historical archival file currently matches.

"The historical record describes raw-to-preparation-to-features replay and the fifty-two-fit final classical stage. It does not recreate every earlier grid, all BERT training or independent adjudication, and is not newly rerun by today's public command.

"The recorded archival check found sixty-nine matches among seventy-four entries, four missing older reporting sources and one changed validation CSV. A historical test-prediction CSV representation difference is documented too. Those discrepancies remain visible; old hashes were not rewritten.

"Aggregate score agreement, exact archival bytes and full experimental reproducibility are different claims. Full restoration needs the authorised inputs and recorded environment described in Data/DATA_README.md and README.md."

**Interpretation:** show the recorded reasons and the scope of each claim. The submission contains no separate manifest or package-integrity utility.

## Scene 12 — Connect results to the report and related work

**Show:** Report/Final_Report.pdf, the Related Work comparison, Results section, Discussion and limitations. Zoom one comparison row and the corresponding task definition.

**Say:**

“The report must explain what these results mean for this task. The strong POS baseline shows that local grammatical information is useful. LR Full improves over that reference. The small encoder is competitive but primary-test superiority over LR is not established, and the class-level tradeoff matters.

“The related-work comparison distinguishes error-type classification at a known span from correction generation, error explanation and other label systems. Those papers motivate design choices and limitations, but their reported scores are not our measured baselines. Different data, labels, inputs and metrics prevent a direct ranking from unrelated headline numbers.

“Limitations include restricted data access, grouping that cannot detect every semantic paraphrase, automatic parse errors, purposeful rather than random case review, seed sensitivity and reuse of the holdout after classical results were known. No fresh external corpus evaluation was run. External evaluation and stronger independent annotation review are future work, not completed experiments.”

**Evidence:** keep statements traceable to the report's comparison table and saved project artifacts; do not invent a new citation or external score for the video.

## Scene 13 — Show the required evaluation outputs

**Show:** the output paths below and point to their matching report figures or sections.

| Required output | Artifact to demonstrate | What the viewer can check |
| --- | --- | --- |
| Result tables | Results/results.json entry tables["Results/classical/overall_results.csv"]; Results/results.json entry tables["Results/bert/overall_results.csv"] | Validation/test distinction and fixed primary |
| Metrics and uncertainty | test_confusions.json; per-class CSVs; paired_differences.csv; bootstrap.json | Point-score arithmetic, direction and saved CI provenance |
| Figures | Results/figures/benchmark_ci.png; per_class_f1.png; confusion_comparison.png | Overall comparison, class tradeoff and directional errors |
| Error analysis | audit_summary.json; team_review_record.md; real_example_19068.json | Selection, overlap, actual saved mismatch and review limits |
| Report | Report/Final_Report.pdf | Question, methods, related comparison, results and justified conclusions |
| Public reproduction | Results/reproduced/checks.json; demo.html; saved evidence bundle | Aggregate consistency and aggregate consistency, not private pipeline restoration |

**Say:**

“These outputs connect the numbers to inspectable counts, visual explanations and case evidence. I have shown what is recalculated live, what is displayed from the saved experiment, and what cannot be reconstructed from public aggregates alone.”

## Scene 14 — Substantiate your personal contribution and conclude

**Show:** your completed the personal-contribution checklist in README.md and the specific artifact supporting your actual action.

**Say:**

“The saved validation review assignments for me are 1269, 18774, 19065, 19068, 19069 and 19185. The evaluation responsibility is the assignment; this recorded review scope is a separate piece of evidence. I will now identify one evaluation, review or reporting action I actually performed, show the artifact, explain a decision or correction, and state what I personally checked and what involved shared or AI help.”

**Required personalised segment:** demonstrate an actual change, calculation check, review entry or interpretation correction from your own work. Explain its effect on the result or conclusion. If your activity was verification, describe verification accurately; do not claim sole implementation. Fill the worksheet before recording rather than reading placeholders or inventing a contribution history.

**Closing narration:**

“My conclusion is that evaluation quality requires more than a high number: a defined task, recoverable counts, the correct uncertainty unit, an honest comparison and review limits. On this version, LR is a strong reference and primary BERT has not demonstrated a Macro-F1 advantage. The package supports checking that conclusion while making its reproduction boundaries explicit.”

## Recording acceptance checks

| Marking criterion | Required visible evidence |
| --- | --- |
| Technical explanation | Real TP/FP/FN arithmetic; Macro-F1; grouped paired resampling; uncertainty versus seed dispersion; review selection and overlap |
| Evidence and demonstration | Live aggregate reproduction, metrics command, synthetic group execution, real 19068 trace, aggregate checks and saved provenance |
| Personal contribution | Actual-action worksheet with supporting artifact, decision/correction, personal checks and accurate attribution |
| Production quality | English narration/captions, readable row/column and CI labels, persistent evidence labels, visible commands/results and clear audio |

Give the calculation its own screen; avoid displaying a full page of tiny tables. Hold the zero line and interval direction while discussing uncertainty. Check captions for Macro-F1, Nn, SVA, group counts, overlap arithmetic and minus signs. Preserve complete evidence explanations rather than cutting to a preset runtime.
