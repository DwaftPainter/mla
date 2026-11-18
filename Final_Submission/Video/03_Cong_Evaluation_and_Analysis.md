# Công - Presentation and code walkthrough guide: evaluation and analysis

Project: **NUCLE-final-20261117**. Prepare and present in English. **No duration limit.** This is a guide to understanding the evaluation, its implementation and its conclusions. Use the questions and worked calculations to explain naturally; there is no speech to memorise.

## How to use this guide

Follow a claim back to its denominator, code and evidence. For each comparison, identify the evaluation unit, metric, direction of subtraction, uncertainty procedure and limit of interpretation. Showing a chart is useful only when you explain what calculation it represents.

Open [README](../README.md), [the report](../Report/Final_Report.pdf), [main_results.csv](../Results/main_results.csv) and [results.json](../Results/results.json). The bundle stores original record names under json, tables and texts. For example, `json -> Results/evidence/test_confusions.json` names an entry inside the bundle, not another file to locate. Existing [figures](../Results/figures/benchmark_ci.png) can support the explanation. No extra helper program or test file is needed.

Keep source inspection, saved experimental evidence and actual live execution clearly identified. Public aggregate evidence is sufficient for a saved-output demonstration; it does not contain all private rows, checkpoints or inputs needed to rerun inference.

## 1. Define the question and trace the evaluator

The task classifies a supplied error location into ArtOrDet, Prep, Nn, Vform or SVA. The observation is an annotation, not an essay. The predictor receives the original sentence and known span; it does not receive the correction or target label.

Evaluation asks how often a frozen predictor agrees with the original annotation category on held-out content groups. This definition matters: a mismatch is an error against the dataset convention. It does not by itself prove that the predicted grammatical interpretation is impossible. Supplied locations also mean these results do not measure full error detection or correction generation.

### Code map

Paths below are relative to Code/.

| Question | File / functions | What to follow |
| --- | --- | --- |
| How are scores defined? | [src/evaluation/metrics.py](../Code/src/evaluation/metrics.py): scores | Gold-row/prediction-column counts, zero handling, fixed-label mean |
| How are frozen predictors scored? | [src/evaluation/nucle_final_evaluation.py](../Code/src/evaluation/nucle_final_evaluation.py): verify_lock, predict, score_test | Identity checks, feature reconstruction, coordinate remapping, prediction and aligned output |
| How are diagnostic slices defined? | Same file: subset_metrics, matrix_metric | Gold-based filtering, retained outside-pair predictions, selected-class F1 |
| How are intervals calculated? | Same file: bootstrap | Per-group matrices, shared multinomial weights, percentile differences |
| How is BERT compared on the same resamples? | [src/evaluation/bert_final_analysis.py](../Code/src/evaluation/bert_final_analysis.py): group_bootstrap, run | Group/row alignment, historical weights and paired differences |
| How are the forty BERT/LR cases selected? | Same file: balanced_take, choose_audit, create_audit | Outcome quotas, SVA priority, frozen IDs before source review |
| What is the separate classical audit? | [src/evaluation/nucle_final_error_audit.py](../Code/src/evaluation/nucle_final_error_audit.py): choose, run | Different LR/Tree selection, despite a similar case count |
| What does replay verify? | [src/evaluation/nucle_final_evaluation.py](../Code/src/evaluation/nucle_final_evaluation.py): replay | Stored vectors -> saved states -> exact label agreement |

A useful explanation route is `frozen prediction -> aligned gold -> confusion counts -> class scores -> paired group uncertainty -> case review -> bounded conclusion`.

## 2. Derive Macro-F1 from real counts

Start with `json -> Results/evidence/test_confusions.json -> matrices -> LR_03`. Label both axes explicitly: rows are gold, columns are predictions; order is ArtOrDet, Prep, Nn, Vform, SVA.

For class k, the diagonal is TP, the column sum is all predicted k, and the row sum is all gold k. Therefore FP = column sum - TP and FN = row sum - TP. This gives:

`precision_k = TP / (TP + FP)`

`recall_k = TP / (TP + FN)`

`F1_k = 2 TP / (2 TP + FP + FN) = 2 TP / (row_sum + column_sum)`

The second form explains the vectorized implementation in matrix_metric. It is the same metric, not a shortcut that changes the definition.

### Worked project example: LR Nn

| Quantity | Observed value | Meaning |
| --- | --- | --- |
| TP | 419 | Gold Nn, predicted Nn |
| Gold support | 565 | Nn row sum |
| Predicted Nn | 512 | Nn column sum |
| FP | 93 | 512 - 419 |
| FN | 146 | 565 - 419 |

Precision = 419/512 = 0.818359375. Recall = 419/565, about 0.741593. F1 = 838/1077 = 0.778087279. Nn recall is lower than precision: more of its gold cases are missed than its predictions are false, relative to the relevant denominators. That is a descriptive error pattern; the matrix does not identify its linguistic cause.

Macro-F1 is `(F1_ArtOrDet + F1_Prep + F1_Nn + F1_Vform + F1_SVA) / 5`. It is not the harmonic mean of macro precision and macro recall. Accuracy is the diagonal total divided by all rows. In this single-label task, micro-F1 would equal accuracy; Macro-F1 gives each class equal influence instead.

### Read scores as an implementation contract

scores rejects empty or unequal-length input sequences and unknown labels. It allocates a fixed five-by-five matrix, increments one cell per aligned observation, computes class precision/recall with zero for undefined denominators, then averages all five class F1 values. Majority's unpredicted classes remain in that average; they are not removed to make the score larger.

The length check cannot detect a shuffled prediction list of the same length. Correct sample-ID alignment must be established upstream. This is why provenance and joins are part of evaluation correctness, not just file management.

## 3. Explain why higher accuracy can coexist with lower Macro-F1

The saved test set has 2,368 annotations in 97 connected groups. The five headline Macro-F1 values are Majority 0.1184, POS 0.7106, LR Full 0.8163, Tree Full 0.7996 and primary BERT-small 0.8146. Keep the primary checkpoint distinct from the three-seed mean.

Primary BERT gets 1,998 predictions correct, accuracy 0.84375. LR gets 1,979 correct, accuracy about 0.83573. Yet BERT has slightly lower Macro-F1. Show [per_class_f1.png](../Results/figures/per_class_f1.png) and `tables -> Results/bert/per_class_metrics.csv` to resolve this apparent contradiction.

| Class | LR Full F1 | Primary BERT F1 | BERT - LR, approximately |
| --- | --- | --- | --- |
| ArtOrDet | 0.869272 | 0.882324 | +0.013053 |
| Prep | 0.914923 | 0.921811 | +0.006887 |
| Nn | 0.778087 | 0.809793 | +0.031706 |
| Vform | 0.708434 | 0.719818 | +0.011384 |
| SVA | 0.810573 | 0.739229 | -0.071344 |

The exact Macro-F1 difference equals one fifth of the sum of the five class differences. The SVA decline outweighs the other gains on this metric. Accuracy instead counts each annotation equally, so improving a more frequent class can offset many fewer correct predictions in another class. This is an arithmetic explanation, not a causal account of why BERT loses SVA.

The joint prediction outcomes are both correct 1,860; BERT only correct 138; LR only correct 119; both wrong 251. These add to 2,368; 1,860+138 and 1,860+119 recover the respective correct totals. These are correlated outcomes on the same observations, not two unrelated accuracy samples.

Use [confusion_comparison.png](../Results/figures/confusion_comparison.png) to inspect directions. LR has 134 Nn -> ArtOrDet errors and 59 reverse errors; Vform -> SVA is 29 and SVA -> Vform is 18. A large cell suggests cases to review. It does not establish that every cell contains the same grammatical phenomenon.

## 4. Follow score_test from a lock to an evaluated label

Explain why each check exists, rather than reading the function line by line.

1. **verify_lock:** validate the lock-file hash and every recorded source/input/state hash. A changed encoder or checkpoint can make an old score incomparable even if the filename remains the same. Hashes establish expected byte identity; they do not prove that the scientific method is sound or who wrote the code.
2. **Scoring guard:** reject a workspace already containing test_predictions.csv or test_scoring_started.json. The start marker is written before the expensive operation. This protects the intended frozen evaluation workflow; it is not a mathematical guarantee that nobody has ever seen test results elsewhere.
3. **Select rows by split:** load test samples only and assert 2,368. Fit nothing on these rows. The full/masked encoders' fit-ID sets are checked against train membership.
4. **Reconstruct representation:** parse each distinct sentence once, extract features, transform with the frozen encoder and assert exact agreement with stored test vectors. Reusing a parse reduces repeated work; it does not merge annotations into one observation.
5. **Predict:** build an old-index -> local-index map from the selected keep list. LR uses sparse matrix multiplication plus bias and argmax; Tree restores its root and traverses it. The same coordinate mapping used in fitting must be used at inference.
6. **Align output:** predictions are written alongside sample ID, gold, group and diagnostic metadata. Alias variants reuse the appropriate predictions. The evaluator then computes metrics and saves encoded rows for replay.

A sparse vector coordinate is meaningful only relative to the exact encoder and keep list. Reordering columns without remapping changes the predictor even if every number still looks plausible. That is a useful concrete failure mode to explain while pointing at predict.

Gold and group IDs legitimately enter scoring and resampling after prediction; they must not enter the predictor values. This distinction prevents the mistaken claim that evaluation cannot access labels at all.

### Diagnostic slices need a separate explanation

subset_metrics defines all, empty/single-token/multi-token spans, selected label pairs and some POS-conditioned pairs. The present retained samples have nonempty spans; supporting an empty slice in code does not mean the experiment evaluated insertion examples.

For Vform/SVA, rows are filtered by gold membership. Predictions outside the pair stay in the five-by-five matrix and count as missed gold cases. Only the two specified class F1 values are averaged for the pair score. Dropping outside-pair predictions would silently remove errors and inflate performance. A slice metric has a different population/label average from the headline five-class score.

## 5. Explain why bootstrap samples groups

Multiple annotations can share a sentence, essay or connected source-content component. They are dependent. Treating every annotation as an independent resampling unit can misrepresent uncertainty. The split and bootstrap therefore use connected groups, which can contain more than one essay.

The experiment has 97 test groups. Each bootstrap replicate draws 97 group identities with replacement and includes every annotation in each selected group, repeated according to that group's multiplicity. A group can be omitted or selected several times. The number of group draws stays fixed; the number of annotation rows can vary.

### Conceptual example, not a new experimental result

Let A contain two annotations, B one and C three.

| Multiplicities for (A,B,C) | Group draws | Resampled annotations |
| --- | --- | --- |
| (2,0,1) | 3 | 2×2 + 0×1 + 1×3 = 7 |
| (0,2,1) | 3 | 0×2 + 2×1 + 1×3 = 5 |
| (1,1,1) | 3 | 2+1+3 = 6 |

These three rows only explain the mechanism; they do not estimate a confidence interval. Present the actual recorded 1,000-replicate procedure separately. Group resampling addresses dependence represented by the grouping design; it cannot discover every unrecorded semantic relationship or guarantee external generalization.

## 6. Derive the vectorized bootstrap code

In nucle_final_evaluation.bootstrap, groups are sorted to define a stable group-index map. The generator uses the saved seed 20261023. A multinomial draw is an efficient representation of repeated group sampling:

`W = rng.multinomial(G, [1/G, ..., 1/G], size=B)`

Here B=1,000 and G=97. W has shape `(1000,97)`; every row sums to 97. Entry W[b,g] is how many times group g appears in replicate b.

For each model, the code constructs C with shape `(97,5,5)`. C[g,i,j] counts gold i / prediction j in group g. Flatten the two class axes to 25 cells, multiply, then restore them:

`boot = (W @ C.reshape(97,25)).reshape(1000,5,5)`

This expression sums the selected group confusion counts for all replicates without physically duplicating text or prediction rows. It does not average scores. matrix_metric computes each replicate's class F1 from its aggregated row sums, column sums and diagonal, then takes the fixed-label mean.

### Why averaging group F1 would be wrong

F1 is a ratio. In general, `F1(C_A + C_B)` differs from `(F1(C_A)+F1(C_B))/2`. Averaging group F1 would give each group an equal score contribution regardless of its confusion counts and evaluate a different estimand. The actual procedure resamples groups but evaluates annotation-level Macro-F1 on the combined resample.

matrix_metric accepts either one matrix or a stack; summing the last two axes and taking the diagonal preserves the leading replicate axis. Its divide operation sets zero where the class denominator is zero. The main score keeps all five labels even if a resample lacks a class. Missing-class replicates are recorded, not automatically discarded.

For diagnostic slices, bootstrap first checks sufficient original support: at least 20 rows, 10 groups and gold presence for each selected class. A replicate is valid when it contains some slice rows. At least 950 valid replicates are required to publish its CI. That threshold is a completeness rule, not the definition of 95% confidence.

A single overall five-by-five confusion matrix cannot reproduce this procedure: it has lost the allocation of counts to groups. Aggregate counts reproduce point scores; group-specific counts or aligned private predictions are needed for grouped uncertainty.

## 7. Explain paired differences and confidence intervals carefully

Both predictors must use the same row identities, gold labels, group order and W. In bert_final_analysis.group_bootstrap, group sets, weight shape and prediction lengths are checked. The analysis loads the historical classical resampling weights and checks them against the expected seeded draws, so BERT/LR comparisons share the same replicates.

For each replicate b, compute `d_b = F1_BERT,b - F1_LR,b`. The saved percentile CI is the 2.5th and 97.5th percentiles of these differences. It is not a normal approximation of mean plus/minus 1.96 standard deviations, and it is not obtained by subtracting the endpoints of separate model CIs.

Pairing retains the covariance from evaluating both models on the same content. The identity `Var(A-B) = Var(A)+Var(B)-2 Cov(A,B)` explains why independently resampling the models answers a different uncertainty question. Do not claim pairing must always make an interval narrower; the covariance matters.

| Contrast, first minus second | Point difference | Saved 95% paired interval, rounded |
| --- | --- | --- |
| LR Full - POS | +0.105668 | [0.082916, 0.128835] |
| LR Full - Tree Full | +0.016609 | [0.001793, 0.030061] |
| Primary BERT - LR Full | -0.001663 | [-0.018410, 0.018153] |

Use [paired_differences_focus.png](../Results/figures/paired_differences_focus.png) to show the zero line. Identify the subtraction direction before interpreting a sign. The first two intervals support a positive difference under this procedure. The BERT/LR interval spans zero; it does not establish BERT superiority, LR superiority or formal equivalence. Equivalence requires a prespecified practically acceptable margin and a suitable analysis, which was not performed.

These are pointwise intervals. Many contrasts were inspected; no familywise multiplicity correction is claimed. A confidence interval is not a statement that a fixed true difference has a 95% probability of lying inside this particular observed interval. Present it as uncertainty from the stated repeated-sampling approximation and its assumptions.

The procedure resamples observed test groups, conditional on the fitted predictors. It does not integrate all uncertainty from training seeds, split selection, a new corpus, parsing decisions or disputed annotation conventions.

## 8. Separate seed variability, primary results and ensembling

The selected BERT configuration was chosen by mean best-checkpoint validation Macro-F1 over three fixed seeds. The primary seed remains 20261013; it was not replaced by the seed with the best test score.

The primary test Macro-F1 is about 0.814595. The three-seed test mean is about 0.818173 with sample SD about 0.003829. A mean of independently evaluated run scores is not an ensemble prediction. An ensemble would combine outputs and then evaluate the combined predictor; that was not done here.

The sample SD uses three run scores and denominator n-1. It describes variation among these three fits. The group-bootstrap CI describes resampled test-content uncertainty for a fixed fit. Neither is interchangeable with the other, and three seeds are limited evidence about all possible training variation.

The extension reused the holdout after classical test findings were available. Its own configuration selection used validation, but this history still limits claims of a completely untouched, independently confirmatory comparison. A genuinely new external test would answer a stronger question and was not run.

## 9. Interpret ablations as defined pipeline comparisons

Use the complete classical tables in the bundle, not just the five-row headline CSV. Identify both variant IDs and whether configurations were separately validation-selected.

| LR Full minus variant | Point difference, approximately | Saved 95% CI, rounded |
| --- | --- | --- |
| Target lexical masking | 0.013423 | [0.004901, 0.020108] |
| Remove all L | 0.028129 | [0.014188, 0.039407] |
| Remove P | 0.018082 | [0.006950, 0.029620] |
| Remove M | 0.003980 | [-0.006562, 0.013566] |
| Remove D | 0.006019 | [-0.005397, 0.019113] |

Target masking removes target lexical candidates while retaining lexical context and the original parser analysis. Removing all L removes the lexical family. These interventions answer different questions. They involve train-fitted encoders and retrained/tuned predictors, rather than zeroing a column at test time.

POS, morphology and dependency families overlap in information: dependency attributes can include POS/morphological signals, and the fixed parser still saw the original words. An ablation difference is therefore not an isolated causal contribution of one linguistic ability. A CI crossing zero does not prove a family contains no information.

Tree's masked point score is slightly above Full and its interval includes zero, so the LR trend is not universal. Full minus D aliases S+L+P+M; duplicate pipeline names are not independent confirmations. The twenty named classical pipelines correspond to sixteen distinct fitted states. Keep those counts separate when describing evidence strength.

## 10. Trace real validation example 19068 without inventing its cause

Open `json -> Results/evidence/real_example_19068.json`. Follow `annotation ID -> span -> linguistic feature subset -> gold/prediction -> review question`.

| Publicly recorded fact | What it permits you to explain |
| --- | --- |
| Validation annotation 19068; exact [10,18) | Eight-character span with valid sentence-relative reconstruction |
| Two covering tokens | This is a multi-token, not single-token, target |
| aux + ROOT dependency signature; singular-number information | The visible parser-derived representation contains these signals |
| Gold SVA; saved LR/Tree prediction Vform | Both predictors disagree with the dataset category |
| No full sentence/correction/lexical vector in this record | Public metadata alone cannot reproduce inference or adjudicate the correction |

The auxiliary-plus-root pattern suggests a useful review question: might the source correction involve both agreement and verb-form changes? Answering requires the original correction and neighbouring annotations. It is a hypothesis to investigate, not a conclusion established by these fields.

Distinguish three levels: the observed mismatch, a source-supported linguistic interpretation, and a causal explanation of the model's decision. Even a plausible interpretation does not prove that a specific dependency feature caused the prediction. No counterfactual attribution experiment was performed. Gold remains unchanged for reported scoring.

Your assigned validation IDs are 1269, 18774, 19065, 19068, 19069 and 19185 under `json -> Results/evidence/audit_summary.json -> validation -> member_ids -> Cong`. The aggregate team review record documents group decisions; do not attribute a detailed comment about this case to that record if it is not actually present there.

## 11. Explain what the audit samples can and cannot establish

The eighteen validation cases were purposefully chosen from 364 LR errors: three sorted IDs in each of four principal confusion directions, plus six multi-token cases. Eighteen is about 4.95% of those errors, but this is coverage-oriented selection, not a random prevalence estimate.

The required forty BERT/LR test cases use bert_final_analysis.choose_audit:

- Twelve BERT-wrong/LR-correct, with six prioritized gold SVA cases and six additional cases.
- Twelve BERT-correct/LR-wrong.
- Ten both wrong.
- Six both correct controls.

balanced_take cycles gold/BERT/LR buckets with seeded within-bucket ordering, using seed 20261028. The IDs are fixed before looking up learner sentences and source corrections. This makes the selection traceable and reduces the opportunity to choose only persuasive correction stories. It does not turn quota sampling into a random sample of all test errors.

create_audit joins those IDs to source annotations, checks source hash, gold and span identity, then prepares review material. It records edit-operation summaries from casefolded whitespace-token sequences. An automatic insert/delete/replace description is an alignment summary, not a grammatical adjudication. Missing correction and an empty replacement must remain distinct.

### Do not confuse two forty-case code paths

nucle_final_error_audit.choose is a separate classical packet: 25 LR errors across gold/prediction buckets, five LR-correct/Tree-wrong cases and ten LR-correct controls. Its seed and strata differ. A similar total count does not make it the BERT/LR forty-case audit. Point to the relevant selector for the claim being made.

The 22 consolidated decisions contain nine supported, six reconsider, three boundary, three optional and one multi-operation status. Eight overlap the forty-case set, so the extended union has `40 + 22 - 8 = 54` distinct annotations. The eighteen validation cases form a separate set. Reconsider means further review is justified; it does not mean a gold error has been conclusively established.

Correct-for-both controls matter because label agreement does not certify that a correction is unique or necessary. The 1 November approval records team interpretation and follow-up; it is not independent expert adjudication. Neither purposeful quotas nor team agreement supports a population percentage of annotation mistakes. No post-prediction gold changes were made.

## 12. Distinguish levels of reproducibility

A useful discussion separates claims that require different evidence:

| Claim | Necessary evidence / scope |
| --- | --- |
| Recalculate a reported point score | Correct confusion counts and fixed metric definition |
| Recompute grouped CIs | Aligned predictions plus group allocation and identical resampling procedure |
| Replay a frozen predictor | Matching checkpoint, encoder/coordinates, encoded rows and valid lock |
| Regenerate features | Authorised raw input, exact preparation/split/parser/encoder and versions |
| Retrain the experiment | Training source/configuration/inputs/environment and actual completed runs |
| Generalize externally | A new suitable evaluation dataset; replay cannot supply this |

The later assistant-executed isolated check completed all thirteen classical stages. It produced sixteen byte-identical checkpoints; all twenty named pipelines' 2,368 test labels matched, 47,360 comparisons in total. All thirty compared paired CI rows matched. These are strong reproduction checks, but 47,360 comparisons are not 47,360 independent observations: the same 2,368 annotations in 97 groups are evaluated repeatedly, and some names alias states.

The fresh execution created its own matching lock. The historical archival check still reports 69 matches out of 74 entries, four missing old reporting sources and a changed validation CSV; a test-CSV representation difference is also documented. Label agreement, metric agreement and byte identity are different claims. Line endings or column formatting can alter a CSV hash without changing its semantic predictions. Do not rewrite old hashes merely to make a check pass.

The BERT check comprised a separate one-epoch smoke run and read-only validation inference from three existing selected checkpoints, reproducing all 2,368 validation labels per seed. It did not retrain the full twelve-run BERT grid or open a new BERT test comparison. Attribute these completed checks to the assistant, as recorded in [AI_Use_Log](../AI_Log/AI_Use_Log.md); they do not establish a student's personal execution history.

### What the existing replay command actually does

From a prepared Code/ workspace with matching private states, stored encoded rows, predictions and source/input lock:

```powershell
.venv/Scripts/python.exe -m src.evaluation.nucle_final_evaluation replay
```

replay verifies the lock, loads saved encoded rows, checks their ID order, obtains predictions from saved states and asserts zero label disagreements. A successful run prints CHECKPOINT REPLAY EXACT. It does not refit the encoder, reparse source text, retrain models or regenerate bootstrap intervals.

The already prepared isolated workspace is `.cache/submission_runtime_20261118/Final Submission/Code/` in the working repository, outside the submission. A public-only copy is missing replay prerequisites. If you show saved evidence, identify it as saved evidence; do not call it live inference. Do not rerun score in an already scored workspace; the original guard deliberately rejects that operation.

## 13. Build a natural presentation around evidence

A coherent route is: define the question, calculate Nn F1, resolve the accuracy/Macro-F1 tradeoff, trace score_test, explain grouped paired uncertainty, inspect real case 19068, then discuss audits and limits. Use the detailed sections as preparation, not mandatory paragraphs to speak in order.

For each displayed artifact, be ready to answer:

- **What is this?** Name the split, predictor, metric and saved/live status.
- **How was it produced?** Trace a relevant function and its inputs/outputs.
- **What does it show?** State a concrete count, difference or diagnostic pattern.
- **What remains uncertain?** Explain the methodological boundary that matters here.

Keep the source block and corresponding artifact visible together when possible. Enlarge the Nn row/column, label the zero line on a paired chart and show the shapes beside the bootstrap multiplication. Avoid scrolling through a whole file while giving an unrelated explanation. If using a conceptual drawing, label it as conceptual; if executing replay, keep the actual terminal result visible.

Connect the report's Related Work comparison to the task definition. External error detection, correction generation and explanation papers have different input information, outputs, corpora and metrics. A qualitative methodological comparison is useful; ranking their headline scores against known-span five-class Macro-F1 is not justified. Those papers were not rerun as external baselines here.

The defensible conclusion is that the POS reference is strong, LR Full improves over it, and primary compact BERT is competitive without an established Macro-F1 advantage over LR. SVA behavior and source-aware case review explain why the overall table is incomplete on its own. Limits include supplied gold locations, residual content overlap, parser errors, purposeful audit selection, limited seeds and holdout reuse. No external-corpus, calibration or educational-intervention result is claimed.

## 14. Questions to answer from understanding

| Question | Reasoning to be able to reconstruct |
| --- | --- |
| Why not just report accuracy? | Frequent classes dominate row totals; equal-class F1 exposes a different tradeoff. |
| Is Macro-F1 the F1 of averaged precision and recall? | No: calculate F1 per fixed class first, then average. |
| Why do prediction IDs matter if list lengths match? | Equal lengths do not establish corresponding observations. |
| Why group resampling? | Related annotations move together; the dependence unit differs from the scoring unit. |
| Why not average the F1 of sampled groups? | F1 is nonlinear; aggregate confusion counts before scoring. |
| Can the overall matrix recover the bootstrap CI? | It lacks row-to-group allocation. |
| Why share W across models? | Paired differences preserve same-content covariance. |
| Is 950 valid replicates the meaning of 95% CI? | No: it is a minimum completeness threshold; percentile endpoints define the interval here. |
| Does a zero-crossing difference CI prove equality? | No; uncertainty includes both directions and equivalence was not tested. |
| Is the three-seed mean an ensemble or CI? | Neither: it is the mean of three run scores with separately reported sample SD. |
| Are forty purposefully selected cases representative percentages? | Their quotas target coverage; prevalence needs an appropriate sampling design. |
| Does exact replay prove successful retraining or generalization? | It verifies saved-state label reproduction under its prerequisites. |

For personal contribution, prepare one supported action: a metric check, an audit decision with its evidence, a figure interpretation/correction, or a reproduction check you actually performed. Explain the input, your reasoning, the resulting artifact and what you checked personally. An assigned role or review ID is not proof that you implemented every evaluator; distinguish shared work and AI assistance. Add no invented authorship, run or numerical improvement.

**Technical explanation** is the metric/resampling/source reasoning. **Evidence and demonstration** is the real calculation, chart, code trace and case, with saved/live status clear. **Personal contribution** is a supported action and decision. **Production quality** is readable axes/equations/code, consistent version and model names, clear audio and a coherent argument. Use these as preparation goals rather than lines to recite.
