# Duy — Complete individual video script: models and training

Project version: **NUCLE-final-20261117**. Spoken language: English. **No duration limit or target length.** Follow the complete sequence and leave enough screen time for every explanation, execution and interpretation.

All JSON/CSV basenames below refer to named records in Results/results.json. Use its json, tables or texts section, or the included evidence-view commands; these records are not separate files in the submission. Common setup, recording guidance and the personal-contribution checklist are in [README.md](../README.md).

## Recording brief and evidence boundaries

The scope in new_task.md is Majority, POS rule, Logistic Regression, Decision Tree, validation tuning, ablation/masking and BERT-small diagnosis. Required outputs: checkpoints, chosen configurations, predictions and validation results. Show all four and identify the private artifacts that are not distributed.

Use persistent screen labels: **REAL SAVED EVIDENCE** for final experiment records and **SYNTHETIC LIVE EXECUTION** for actual solver calls on authored toy vectors. The latter demonstrates mechanics, not benchmark performance. Commands run from the submission root; the live solver requires the research environment in [submission README](../README.md). Complete your section of [personal-contribution checklist](../README.md#personal-contribution) before recording. An assigned responsibility or a demonstration does not establish original code authorship.

## Scene 1 — State the modelling question and your role

**Show:** face camera, project version, five labels and the Duy responsibility row in the main README.

**Do:** draw original sentence + known error interval → five-class predictor → label. Mark correction as excluded from input.

**Say:**

“Hello, I am Duy. My assigned part is model construction and training: the Majority and POS baselines, Logistic Regression and Decision Tree, validation selection, feature ablation and lexical masking, and the BERT-small experiments and diagnosis.

“We predict ArtOrDet, Prep, Nn, Vform or SVA at an already known error location. We are not measuring error detection or generating corrected sentences. There is no ‘correct sentence’ output class. I will connect the training inputs, fitted decision rule, validation choice and a saved real error. I will also run the actual classical solvers on explicitly synthetic vectors so the calculations are visible.”

**Evidence:** identify assigned scope here; substantiate your own actual work in Scene 14.

## Scene 2 — Establish exactly what reaches the learner

**Show:** Results/results.json entry json["Results/evidence/feature_manifest.json"] and the separate feature/target contract in README.md. Run:

```powershell
python Code/video_lab.py case --annotation-id 15390
```

**Do:** highlight partition, gold, span, nonlexical_features and saved predictions. Draw a dotted boundary around the omitted lexical coordinates.

**Say:**

“The classical models receive sparse numeric vectors from Quân's train-fitted encoder. Full has 16,180 coordinates. Targets are separate; correction and gold must not become features. Train has 11,052 annotations; validation and test each have 2,368, with related essays partitioned together.

“This is real validation annotation 15390, assigned to me in the saved review record. Its sentence-relative interval is eighty-three to eighty-eight, with the end excluded: five characters. Alignment is exact. Gold is SVA, but both saved Full models predict Vform. Visible features include a one-token VERB signature, infinitive VerbForm and a singular third-person subject representation.

“These are text-free partial features. I cannot recreate the published decision from this JSON alone because the full lexical vector and trained checkpoint are absent. The command displays a saved prediction, not a new Full-model inference.”

**Interpretation:** show SVA → Vform. The visible signature suggests review questions; it does not prove which weight caused the error.

## Scene 3 — Explain the two simple baselines

**Show:** Code/src/training/nucle_benchmark.py and Code/src/evaluation/nucle_final_evaluation.py. Locate Majority and follow the POS-rule call to its actual implementation. Open Results/results.json entry tables["Results/classical/overall_results.csv"].

**Do:** highlight the Majority and POS rows, keeping validation and test columns separately labelled.

**Say:**

“Majority learns the most frequent training label and predicts it everywhere. Here that is ArtOrDet. Its test accuracy is 0.4206, but Macro-F1 is 0.1184 because four classes receive no correct predictions. Accuracy alone would disguise this weakness.

“The POS baseline uses grammatical signatures, train-derived label mappings and deterministic fallbacks, including ADP and DET anchor behavior. It does not read a correction to decide the answer. Its test Macro-F1 is 0.7106. That makes it a useful reference for whether learning improves over simple grammatical information.

“These are our project baselines under the same labels and prepared data, not reimplementations of external papers. A noun is not automatically an Nn error and a verb is not automatically SVA: error type can depend on context and annotation boundaries.”

## Scene 4 — Run the actual LR and Tree solvers

**Run:**

```powershell
python Code/video_lab.py models
```

**Show:** Code/video_lab.py beside Results/reproduced/video_lab/synthetic_model_demonstration.json and synthetic_model_state.json. Keep the synthetic label visible.

**Do:** identify shape 10 × 5, query `{2: 1.0}`, logits, softmax, and tree route. Highlight calls to fit_logistic and GiniTree.fit. Open the saved weights, bias and tree structure.

**Say:**

“This command calls our actual Logistic Regression and Gini Tree implementations. Its ten authored rows contain two rows per label and five artificial coordinates. Those coordinates are not research features, and this query is not a learner sentence. We are demonstrating solvers, not measuring new grammatical accuracy.

“For LR, the vector is multiplied by learned weights and bias is added. Coordinate two activates the Nn score in this illustration. Softmax normalizes five logits and argmax selects Nn. In the checked environment the Nn probability is approximately 0.479. It is not a research confidence estimate or evidence of calibration.

“The Tree tests coordinate zero and goes left, coordinate one and goes left, then coordinate two at threshold 0.5 and goes right to Nn. Its printed route exposes every branch. The saved JSON makes both toy states inspectable; neither is the final project's checkpoint.”

**Expected output:** both toy predictions are Nn. Read the current floats on screen; do not change them to match a rehearsed decimal.

## Scene 5 — Explain the LR objective and selection

**Show:** fit_logistic in Code/src/models/nucle_logistic.py; LR_03 in Results/results.json entry json["Results/classical/selected_configurations.json"]. Display:

`z = xW + b; p = softmax(z)`

`objective = mean cross entropy + ||W||² / (2 × C × n)`

**Do:** connect toy weights to W, then switch to the real-evidence label for LR_03. Highlight configuration and termination.

**Say:**

“The research LR is multinomial softmax with an L2 penalty, optimized by PyTorch LBFGS. Larger C weakens the penalty in this objective. The grid crosses C values 0.01, 0.1, 1 and 10 with unweighted or balanced class loss: eight settings per tuned variant.

“Full selects C equal to 0.1 without class weighting using validation Macro-F1, convergence eligibility and deterministic ties. The optimizer permits 1,000 iterations and 1,250 evaluations. Gradient tolerance is ten to the minus six and change tolerance ten to the minus nine. The saved Full run stops at 369 iterations under change tolerance. That is not evidence that its gradient met the gradient threshold.

“The toy used C equal to one and a smaller iteration budget. Its execution does not replace the recorded Full configuration. Numerical stopping is also not proof of perfect classification or population generalization.”

## Scene 6 — Explain the Tree's split and leaf behavior

**Show:** Code/src/models/nucle_tree.py, especially _impurity, _build and leaf prediction; Tree_03 in selected_configurations.json.

**Do:** annotate `Gini = 1 − sum(p_class²)`, parent/child impurity reduction, depth and minimum-leaf checks. Revisit a toy branch.

**Say:**

“The Tree chooses feature thresholds that reduce Gini impurity. Sparse absent coordinates are zero. Numeric features and one-hot categorical coordinates use different candidate-threshold handling; candidate ordering and ties are deterministic.

“The validation grid uses depths four, eight and twelve, minimum leaf sizes five and twenty, and unweighted or balanced impurity: twelve combinations. Full selects depth eight, minimum leaf size five, without class weighting.

“Balanced weights affect split impurity, but the leaf label uses raw class-count majority and a fixed label-order tie break. Minimum leaf size counts raw samples too. Calling this fully weighted leaf voting would be inaccurate. Depth and leaf size constrain fitting capacity; their selected values come from validation rather than from whichever test score looks best.”

## Scene 7 — Demonstrate ablations and distinguish lexical experiments

**Run:**

```powershell
python Code/inspect_evidence.py ablation
```

**Show:** variant construction in Code/src/evaluation/nucle_final_evaluation.py, printed IDs and selected_configurations.json.

| Suffix | Input change |
| --- | --- |
| _00 / _01 / _02 / _03 | S+L / add P / add M / Full with D |
| _04 | Mask target lexical candidates; keep context and original parse |
| _05 | Remove all L |
| _06 / _07 | Remove P / remove M |
| _08 | Remove D; reuse _02 |

**Say:**

“These are defined training pipelines. The span mask removes target lexical candidates before fitting a fresh train-only encoder, while preserving left/right lexical context and original parse features. Removing all L is a different, stronger intervention.

“Saved LR test Macro-F1 is 0.8163 for Full, 0.8028 for span masking and 0.7881 without L. Full minus P is approximately 0.7982. Công will interpret paired uncertainty; here I am identifying the trained pipeline behind each row. Tree's masked point result is slightly higher than Full, so the LR trend is not universal.

“Dropping P does not erase all POS information because D contains head and subject POS. Dropping M does not erase every morphological signal in other families. These compare our representations, not isolated causal linguistic knowledge.

“Twenty classical names comprise two baselines and eighteen LR/Tree names. Full minus D aliases S+L+P+M in each family, leaving sixteen distinct LR/Tree checkpoints. The fifty-two final-stage fits are forty new Full-minus-P/M grid fits and twelve historical-configuration refits. They do not count every tuning experiment in the project's history.”

**Evidence check:** highlight alias metadata; do not count aliases as independent fitted models.

## Scene 8 — Retain and diagnose the initial BERT failure

**Show:** Results/results.json entry json["Results/evidence/initial_bert_summary.json"] and Results/results.json entry json["Results/reproducibility/model_source.json"]. Identify the exact source/revision.

**Say:**

“The selected learning-rate setting in the initial BERT-small phase has mean validation Macro-F1 about 0.1984 with sample standard deviation 0.0030. We preserve that negative result. A pretrained name does not guarantee that our optimization or task readout works well.

“The model is google/bert_uncased_L-2_H-128_A-2 at the pinned revision in model_source.json: a compact two-layer, 128-hidden encoder, not a large chat model. No generation prompt is part of this classifier.

“The extension examines training settings and the representation of the known target span. Poor optimization and an unsuitable readout are diagnostic hypotheses. The initial failure does not establish one exclusive cause.”

## Scene 9 — Explain token boundaries and span-mean readout

**Show:** Code/src/training/bert_span_classifier.py, Code/src/models/bert_span_readout.py and Code/src/training/bert_small_experiment.py. Locate actual input-building and mask code.

**Do:** draw synthetic `She [target] walk [/target] to school.` as a conceptual token diagram. Colour interior target subwords separately from markers and padding; compare CLS with averaging target states.

**Say:**

“Left context, target and right context are tokenized separately. Markers locate the target, and context is cropped while preserving the full target within the 128-token limit. Character indices and subword indices cannot simply be treated as identical.

“CLS uses the sentence-level representation. Span mean averages the hidden states of interior target subwords, excluding markers and padding, then uses the shared pooler and classification head. An empty-mask fallback uses the start marker; retained project spans are non-empty, so that fallback does not imply evaluated insertion errors.

“The design assumes a provided error location. It should not be ranked directly against a system that must first find the error.”

**Do:** highlight the actual target_mask and forward methods, showing where markers/padding are excluded and where the shared pooler is called.

**Say after inspection:** "This source inspection explains the implemented readout. The saved training table shows its recorded experimental results; we are not running a new pretrained BERT experiment."

## Scene 10 — Demonstrate validation selection across twelve runs

**Run and show:**

```powershell
python Code/inspect_evidence.py training
```

Open Results/results.json entry tables["Results/bert/validation_grid.csv"] and training_runs.csv. Highlight the same three seeds for every configuration.

**Say:**

“The extension crosses CLS and span mean with head learning rates ten to the minus four and ten to the minus three. Encoder learning rate stays five times ten to the minus five. Seeds 20261013, 20261014 and 20261015 make twelve runs.

“The four validation means are approximately 0.7164, 0.7227, 0.8183 and 0.8177. Selection averages best-checkpoint validation Macro-F1 across the same three seeds. It selects span mean with head learning rate ten to the minus four, with sample standard deviation about 0.0011.

“Common settings include batch size thirty-two, at most twenty epochs, patience four, min_delta 0.0001, dropout 0.1, weight decay 0.01, warmup ratio 0.1 and gradient clipping at one. There is no class weighting. These runs use four CPU threads; the classical stage uses ten.”

**Do:** highlight the seed-mean selection code and the train/validation loading path in bert_small_experiment.py. Match the selected grid row to summary.json.

**Interpret:** "Selection compares configuration means, rather than promoting the strongest individual seed. The training/validation path loads those partitions; the separate test command uses the locked selection. This source inspection and the saved table explain the protocol without claiming fresh training."

## Scene 11 — Explain checkpoints, patience and the remaining gap

**Show:** Results/figures/primary_learning_curve.png, Results/results.json entry tables["Results/bert/seed_results.csv"] and checkpoint/early-stopping code. Zoom axes and legend first.

**Say:**

“For fixed primary seed 20261013, the best checkpoint is epoch eighteen, with training Macro-F1 around 0.9309 and validation 0.8173. The gap remains. The extension repairs severe initial underfitting but does not eliminate overfitting or domain limits.

“Any strict validation improvement saves a new best checkpoint. Patience resets only when the improvement exceeds the stopping reference by min_delta. A small improvement can therefore become the saved best without resetting patience. The runner reloads the best state and checks all validation predictions against the saved labels.

“Resume restores model, optimizer, scheduler, history and random state with environment checks. A seed alone is not enough. Thread differences have affected exact historical validation replay too, which is why the environment is recorded.

“The initial phase and extension differ in training settings and readout. The controlled readout comparison is inside the extension grid; attributing the entire initial-to-final improvement to span mean would exceed the evidence.”

**Interpretation:** the plot shows train and validation, not test learning curves. The best epoch is not selected through test scores.

## Scene 12 — Identify the required model outputs

**Show:** checkpoint paths and hashes in selected_configurations.json, Results/results.json entry json["Results/bert/summary.json"], README.md and Data/DATA_README.md.

**Say:**

“The handoff needs a fitted state, its configuration and encoder identity, row-aligned predictions and validation results. This package includes chosen configurations, checkpoint paths and hashes, aggregate results and reproduction code. Full research states, lexical vectors and row inputs require authorised restoration.

“A hash identifies an expected state; it does not make an absent checkpoint available. Our toy JSON demonstrates saving state but must not be relabelled as a research checkpoint. Public evidence commands display saved decisions.”

| Output | Evidence to show | Availability |
| --- | --- | --- |
| Checkpoints | Selected classical entries; BERT run and epoch | Paths/hashes recorded; private states require restoration |
| Chosen configurations | selected_configurations.json; validation_grid.csv; summary.json | Included; selected through validation |
| Predictions | Saved real-case decision, confusion counts, run-order contract | Public examples/aggregates; complete private row files omitted |
| Validation results | All twelve BERT runs; selected classical metrics | Included, distinct from test scores |

## Scene 13 — Interpret the final comparison without changing the primary

**Show:** Results/figures/benchmark_ci.png, seed_results.csv and Results/results.json entry tables["Results/bert/paired_differences.csv"]. Highlight BERT minus LR.

**Say:**

“LR Full test Macro-F1 is 0.8163; fixed primary BERT-small is 0.8146. BERT minus LR is about minus 0.0017, with saved paired 95 percent interval minus 0.0184 to plus 0.0182. It does not establish BERT superiority or equivalence.

“The selected BERT configuration's three-seed test mean is 0.8182 with sample standard deviation 0.0038. Seed 20261015 is the best test seed, but the primary remains 20261013. Choosing the best after seeing test results would change the comparison. Seed dispersion and group-bootstrap uncertainty answer different questions.

“The BERT extension occurred after classical test results were available and reused that holdout. Validation-based extension selection does not turn it into a fresh external test. Selected models are not refitted on train plus validation before scoring.”

## Scene 14 — Close the real case and substantiate personal contribution

**Show:** case 15390, Results/results.json entry texts["Results/reproducibility/team_review_record.md"] and your completed personal contribution account. Keep private learner wording out of frame.

**Say:**

“Annotation 15390 is a saved SVA-to-Vform failure. The existing review notes an adjacent modal edit, making correction boundaries relevant to its interpretation. I attribute that reading to the saved review. I am not presenting new expert adjudication or changing gold.

“The saved record assigns me validation cases 12314, 14761, 14764, 12019, 14939 and 15390. That review record and my broader modelling assignment are different evidence. I will now show one action I actually performed, its supporting artifact, the decision or correction it led to, and the part I personally checked.”

**Required personalised segment:** use the prepared account's actual action. Show the relevant change, log, configuration decision or review entry. Explain its consequence and identify shared/AI assistance. If you reviewed or verified existing code, say that rather than claiming implementation. Do not read unfilled placeholders.

**Closing narration:**

“The modelling handoff is a defined feature/target contract, validation-selected states and configurations, saved predictions, and explicit diagnosis of both failed and improved BERT phases. The evidence supports strong classical baselines and a competitive small encoder while retaining the remaining uncertainty.”

## Recording acceptance checks

| Marking criterion | Required visible evidence |
| --- | --- |
| Technical explanation | LR objective/C; Tree split/leaf rules; distinct masks; BERT readout, seed-mean selection and checkpoint/patience |
| Evidence and demonstration | Live toy solvers; real case 15390; all twelve validation runs; actual source inspection; output paths and availability |
| Personal contribution | Completed actual-action record, supporting artifact, decision/correction and clear assignment/authorship distinction |
| Production quality | English narration/captions; readable equations and charts; saved/synthetic labels; clear chapters and audible recording |

Show filenames at transitions; keep outputs visible during interpretation. Check captions for SVA, Vform, LBFGS, Macro-F1 and the CI minus sign. Cut waiting only while preserving command/result continuity. The complete explanation determines runtime, without a preset limit.
