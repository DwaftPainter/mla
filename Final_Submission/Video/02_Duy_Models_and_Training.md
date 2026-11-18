# Duy - Presentation and code walkthrough guide: methods and training

Project: **NUCLE-final-20261117**. Prepare and present in English. **No duration limit.** This is a technical explanation guide, not a prepared monologue. Use it to understand why the models behave as they do and explain the reasoning naturally.

## How to use the guide

For a model, be able to describe its input, learned parameters, training objective, validation rule, inference rule and limitations. Open the relevant function and connect a small block to that explanation. A list of class/function names alone does not explain a method.

[README](../README.md) contains setup and run order. [Results/results.json](../Results/results.json) contains json, tables and texts entries retaining original record names; they are not separate files. [main_results.csv](../Results/main_results.csv) and [the report](../Report/Final_Report.pdf) provide the overall comparison. Distinguish source inspection, saved experimental evidence and an actual live execution. No additional helper program or test file is needed.

## 1. Frame the learning problem before discussing architecture

One sample is an annotation: original sentence plus known span, with one target category among ArtOrDet, Prep, Nn, Vform and SVA. The classifier must choose one of five mutually exclusive output indices even when an actual correction could involve several grammatical operations.

For an authored illustration, changing marked `go` to `goes` in `He go home` addresses agreement, whereas changing `went` to `gone` in `They have went` addresses verb form. These are conceptual teaching examples, not NUCLE records. Both targets can be tagged VERB, which shows why token category alone does not determine edit category. The actual project taxonomy and source annotation remain the scoring reference.

That distinction explains two limitations: gold locations make the problem easier than detection, and one label may simplify a multi-operation edit. A mismatch with gold is a classification error under this protocol; it is not automatically proof that the model's linguistic interpretation is nonsensical.

Classical learners receive a fixed 16,180-dimensional sparse Full vector. BERT receives marked tokenized original text. These are different representations of the same supplied location, not models solving unrelated input tasks.

### Code map

| Question | File / functions | What to follow |
| --- | --- | --- |
| What do simple references exploit? | [features/nucle_pos.py](../Code/src/features/nucle_pos.py): POSRule; [training/nucle_benchmark.py](../Code/src/training/nucle_benchmark.py) | Train counts, signature lookup and fixed rules |
| How is LR fitted? | [models/nucle_logistic.py](../Code/src/models/nucle_logistic.py): fit_logistic | Sparse logits, cross-entropy/L2, LBFGS closure and stopping diagnostics |
| How is Tree fitted? | [models/nucle_tree.py](../Code/src/models/nucle_tree.py): GiniTree | Weighted impurity, candidate splits, stopping and leaf prediction |
| How are representations compared? | [training/nucle_ablation.py](../Code/src/training/nucle_ablation.py); [training/nucle_final_selection.py](../Code/src/training/nucle_final_selection.py) | VARIANTS, feature_indices, masked_features, make_encoders, fit_one |
| How is the known span tokenized? | [training/bert_span_classifier.py](../Code/src/training/bert_span_classifier.py): encode_span | Three regions, markers, budget and cropping |
| What does span mean calculate? | [models/bert_span_readout.py](../Code/src/models/bert_span_readout.py): target_mask, BertReadout.forward | Hidden states -> selected pooling -> shared head |
| How is BERT trained/selected? | [training/bert_small_experiment.py](../Code/src/training/bert_small_experiment.py) | load_rows, optimizer_for, train_one, save_best, load_best, select, test |

Paths in this table are under Code/src/.

## 2. Why keep Majority and POS baselines

Majority counts train labels and always predicts the most frequent one, ArtOrDet. It learns a class prior but no input-dependent distinction. Its test accuracy 0.4206 reflects the frequency of that class; its Macro-F1 0.1184 exposes the zero performance on the other four classes.

POSRule is a hybrid rule/lookup model. Exact ADP maps to Prep and exact DET to ArtOrDet. Other observed ordered POS signatures use their majority training label; unseen signatures fall back to the global train-majority label. Ties use fixed label order.

A DET+NOUN span is not the exact DET anchor. This matters when explaining a real multi-token error. Signature categories describe covering tokens, while the target describes an annotated edit. The rule never needs the source correction to make its decision.

Its 0.7106 test Macro-F1 is a meaningful reference: a complex learner should be judged against information available to this simple grammatical lookup. These are project baselines, not external-paper reimplementations.

**Explain in your own words:** what does each baseline know, what does it ignore, and what would an improvement over it demonstrate? Avoid equating five labels with a universal random-chance F1 of 0.2; chance performance depends on class frequencies and prediction behavior.

## 3. LR is a linear decision rule over engineered features

Let X have shape n × d, W shape d × 5 and b shape 5. The code computes:

`Z = XW + b`

`p(y=c|x) = exp(z_c) / sum_k exp(z_k)`

Prediction is the largest logit/probability. This is multinomial softmax, not five independent binary decisions. For Full, d=16,180, so W and b contain 80,905 scalar parameters. The sparse representation makes a high-dimensional row feasible when most coordinates are zero.

For classes a and b, the decision boundary is `(W_a-W_b)·x + b_a-b_b = 0`. Contributions are additive in the encoded coordinates. However, an engineered local POS-pattern category already represents a combination of original observations. A linear classifier over those features is not equivalent to a context-free bag-of-words model.

### Objective and C

The unweighted implemented objective is:

`mean cross-entropy + ||W||² / (2 × C × n)`

Cross-entropy penalizes assigning low probability to the gold class. L2 discourages large weights; the bias is not included in that penalty. Larger C weakens regularization. A differentiable cross-entropy objective is a training surrogate; the model is selected using validation Macro-F1, which is not directly optimized by this loss.

Balanced class loss uses train-frequency weights n/(5×class_count). It increases the influence of rare classes; it does not guarantee better Macro-F1 because the representation, optimization and resulting false positives/negatives also matter. Full actually selects no weighting.

For an unweighted example, cross-entropy is `-log(p_gold)`. Its derivative with respect to a class logit is `p_c - 1[c=gold]`: gradient descent increases the gold score and decreases competing scores according to their probabilities. Averaging rows and adding the L2 derivative `W/(C×n)` connects the loss formula to the gradients computed by backward. With PyTorch's weighted mean, the cross-entropy numerator is the sum of weighted row losses and its denominator is the sum of the selected class weights; it is not simply divided by n. The stated L2 denominator remains C×n.

Feature scale matters: raw lengths/distances and one-hot indicators are not standardized by a fitted scaler here. A coordinate's coefficient cannot be interpreted without its scale, coding and co-occurring features.

### Follow the code, not just the formula

In fit_logistic:

1. Sparse COO input and row/label alignment are checked.
2. W and b start at zero; the seed is fixed.
3. The closure clears gradients, computes sparse matrix multiplication, loss and penalty, then backpropagates.
4. LBFGS with strong-Wolfe line search may call the closure several times while choosing one update. Clearing gradients inside the closure prevents accidental accumulation across trial evaluations.
5. Final objective and maximum absolute gradient are recomputed, and stopping diagnostics are returned with detached state.

LBFGS uses curvature information from recent updates to choose a step. It is appropriate to understand this as full-batch numerical optimization of the fixed feature problem, not a neural mini-batch training loop.

Full selects C=0.1 without weighting. Its recorded fit stops after 369 iterations under change tolerance. The source verifies gradient tolerance directly but labels other below-cap stops as change tolerance. Do not turn that diagnostic into a claim that the gradient threshold was met or every configuration reached the same optimum.

Limits are 1,000 iterations/1,250 evaluations, tolerance_grad=1e-6 and tolerance_change=1e-9. Hitting a cap and satisfying an eligibility criterion are different outcomes. Explain why the selector excludes ineligible LR candidates.

## 4. Tree learns conditional partitions rather than additive scores

Read GiniTree._impurity, _build and predict_one. A node's Gini impurity is `1 - sum_c p_c²`; zero means all weighted mass is in one class. Candidate gain compares the parent impurity with the weighted average of child impurities.

A greedy split is chosen at each node. The method does not optimize the entire tree globally. Unlike LR, it can express conditional interactions: a POS signal can lead to different predictions depending on a morphology or structural split further down the route.

As a conceptual calculation, a node with two equally weighted classes has Gini `1-(0.5²+0.5²)=0.5`; a pure node has zero. A useful split reduces the weighted child impurity relative to the parent. Purity on train alone is not the objective of evaluation: a tiny pure leaf can memorize observations and generalize poorly, which motivates depth/leaf constraints and validation selection.

Missing sparse values are zero. One-hot splits use threshold 0.5; numeric candidates come from gaps between sorted observed values. A test `value <= threshold` routes left; otherwise right. Gain ties prefer smaller feature indices/thresholds according to the candidate tuple, making the choice deterministic.

Depth, purity and minimum leaf size stop growth. Small leaves permit more detailed fitting but may be unstable; larger leaves smooth over rare patterns. The grid crosses depth 4/8/12, leaf size 5/20 and none/balanced weighting. Full selects depth 8, leaf size 5, no weighting.

**Implementation detail:** balanced weights affect split impurity. The actual leaf label uses raw class-count majority, and the leaf-size constraint counts raw rows. It is not a fully weighted leaf-voting algorithm. That is why source inspection matters beyond naming the method CART.

A source branch diagram explains the rule, but does not reveal the exact private tree path for a redacted case. Do not invent a route or imply that the most plausible-looking feature caused the prediction.

## 5. Separate fitting, hyperparameter selection and final scoring

Training adjusts W/b or tree nodes using train labels. Validation chooses settings by Macro-F1. Test measures the selected predictor on the held-out partition. Checkpoint freezing records which state/input/source combination was selected before scoring.

The LR grid has eight settings: four C values × two weight choices. Tree has twelve: three depths × two leaf sizes × two weight choices. Stable candidate order resolves exact validation ties. LR convergence eligibility is considered before a candidate can win. The selected Full predictors remain train-fitted; the project does not refit them on train+validation before test.

A slightly larger validation score does not establish a statistically reliable difference or justify fishing through test results. Selection and uncertainty are different questions. Also, repeated design decisions after seeing the holdout weaken its role as a fresh independent evaluation even if the final selector itself reads validation only.

**Evidence to open:** selected_configurations.json and classical/overall_results.csv under the bundle. Show the chosen settings beside their validation columns, with test columns clearly labelled.

## 6. What the ablations actually change

| Variant | Mechanism | Interpretation boundary |
| --- | --- | --- |
| S+L, then +P, +M, +D | Keep encoded coordinates belonging to the declared families | Incremental pipelines, not direct measurements of independent linguistic faculties |
| Target-span lexical mask | Rebuild lexical candidates without target tokens, retain contexts/original parse, fit a new train-only encoder | Changes the train representation and its vocabulary; not merely hiding a word at test |
| Remove all L | Drop every lexical-family coordinate | Stronger intervention than target masking |
| Remove P or M | Drop explicit family columns | Related signals can survive in D and other correlated features |
| Remove D | Reuse S+L+P+M | Alias, not another independent fit |

feature_indices parses the encoded feature-family prefix. remap_rows creates local indices so each variant's vectors match its local coefficient/tree feature IDs. Prediction must apply the same keep/remap contract before loading state.

masked_features recomputes lexical candidates with no target-token IDs. It preserves left/right lexical slots and the parse of the original sentence. Automatic POS/morphology/dependencies can still depend indirectly on the original word through the parser. Therefore masking does not erase every lexical influence in the pipeline.

The mask fits a new encoder after changing the train candidates, so even its retained vocabulary can differ. Model weights and sometimes selected hyperparameters also change. A performance difference is evidence about these complete pipelines, not proof of a single causal feature effect.

There are twenty named results: two baselines plus eighteen LR/Tree names. Two aliases leave sixteen distinct LR/Tree states. The final-stage 52 fits comprise forty new P/M-removal grid fits and twelve historical-choice refits; they do not count all earlier tuning. Source make_encoders also checks transformation and POS-rule agreement rather than assuming saved arrays match the current extractor.

## 7. Diagnose the initial BERT failure carefully

The actual compact encoder is google/bert_uncased_L-2_H-128_A-2 at revision 30b0a37ccaaa32f332884b96992754e246e48c5f: two layers, hidden size 128, two attention heads. The initial selected setting has mean validation F1 about 0.1984 ±0.0030. Keep that failed phase visible.

Use diagnostic reasoning rather than one retrospective story:

| Observation to examine | Possible explanation | What would distinguish it |
| --- | --- | --- |
| Train and validation both poor | Optimization, input alignment, insufficient adaptation or weak representation | Loss/gradient checks and whether correctly aligned small examples can be fitted |
| Train strong, validation weaker | Overfitting, distribution shift or label ambiguity | Learning curves and source-aware held-out review |
| Readouts differ under matched settings | How location/context is summarized matters | Compare CLS/span mean within the same grid/budget |
| Marker or label handling inconsistent | Implementation error | Follow masks, label mapping, token IDs and reload behavior |

These are methodological questions, not claims that every listed diagnostic was executed in this submission. Low F1 alone does not establish the cause, and the entire initial-to-extension gain cannot be assigned to pooling because training settings also changed.

## 8. How a character span becomes BERT input

encode_span tokenizes left text, target text and right text separately. It builds:

`[CLS] left [SPAN_START] target [SPAN_END] right [SEP]`

Four special-token slots are reserved. The remaining budget is max_length - 4 - number_of_target_subwords. Context is cropped around the target, preserving every target subword. If the target cannot fit, the source raises an error rather than silently truncating it.

Separate regional tokenization preserves the intended target even for partial-token annotations, but can differ from tokenizing the complete sentence naturally. That is a representation decision worth naming. Dynamic padding lets a batch use its required length, and attention masks exclude padding from attention.

The marker embedding rows and classifier head are trained after adding tokens/resizing embeddings. A newly initialized classification head warning is expected when adapting a pretrained encoder to five labels; it is not itself a runtime failure.

The data loader intentionally loads/tokenizes only requested partitions. Train/validation paths do not need to tokenize test rows. IDs remain alongside rows for joining outputs; gold labels are training targets, not words inserted into the sentence.

### Why the encoder produces contextual states

The input IDs select token embeddings, combined with position and segment embeddings. The compact model has hidden size 128 and two attention heads per layer, so each head uses 64-dimensional query/key/value vectors. For a sequence of T tokens, a head computes query/key similarities, scales them by sqrt(64), masks padding, applies softmax and uses those weights to mix value vectors:

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(64) + padding_mask) V`

The score matrix is T×T. A target token can incorporate information from words before and after it; this is bidirectional attention, with no autoregressive causal mask. Concatenating the heads and applying output projection, residual connections, layer normalization and the feed-forward block yields updated token states. The two layers repeat this contextualization. These encoder operations are implemented by the loaded BERT model called through base.bert, rather than handwritten in BertReadout.

This explains both the opportunity and cost: context beyond the local lexical window can affect a target state, while attention work grows quadratically with retained sequence length. Cropping to max length 128 controls that cost but can remove useful distant information. Attention weights alone would not establish a causal explanation of a class prediction, and this project did not run an attention-attribution study.

## 9. What span mean computes, and what it does not imply

BertReadout first runs the same contextual encoder for either readout. CLS uses the pooler output derived from the first-token state. Span mean builds a mask strictly between the markers, intersects it with attention_mask and averages those hidden states:

`h_span = sum_t(mask_t × h_t) / sum_t mask_t`

It then applies the shared pooler dense+tanh, dropout and classifier. This is not an average of class probabilities or an average after the nonlinear pooler. Those operations are different.

Follow the tensor shapes in BertReadout.forward: last_hidden_state is batch×T×128; the Boolean target mask is batch×T and is expanded for elementwise multiplication. Summing over T and dividing by each row's target count produces batch×128. The pooler keeps 128 dimensions, then the classifier returns batch×5 logits. Argmax returns one class index per row; softmax is not required merely to find that index. Pooling connects token-level states to one annotation-level decision.

Crucially, each h_t is contextualized by attention over the input. Averaging target states does not make the model context-free. Markers are excluded from the averaging mask but still participate in the encoder input. CLS can also attend to markers; a poor CLS result does not prove that attention cannot represent a target.

target_mask requires one valid start/end marker and correct ordering. An empty interior falls back to the start marker; the retained project spans are nonempty, so do not claim tested insertion-only coverage from that fallback.

## 10. Understand the optimizer and the actual training loop

In bert_small_experiment.py, optimizer_for groups pooler/classifier parameters at head_lr and embeddings/transformer at encoder_lr. Biases and LayerNorm weights receive no weight decay; other parameters use the configured AdamW decay. Thus head_lr does not refer only to the last linear classifier.

train_one performs the normal differentiable training sequence: clear gradients -> forward -> cross-entropy -> backward -> clip gradients -> optimizer step -> scheduler step. It checks finite loss. The encoder is fine-tuned, not frozen into fixed embeddings.

The schedule uses a warmup fraction followed by linear scheduling over planned batch steps. Gradient clipping limits the update-driving gradient norm; it is not a substitute for the loss or weight decay. Dropout is active during model.train and disabled by model.eval for evaluation. Train evaluation uses a fixed non-shuffled pass; training batches shuffle reproducibly by seed/epoch.

The actual extension uses batch 32, max length 128, up to twenty epochs, patience four, min_delta 0.0001, dropout 0.1, weight decay 0.01, warmup 0.1, clipping 1.0, no class weighting and four CPU threads. Explain the role of these settings instead of reciting them as an unconnected list.

## 11. Best checkpoint, early stopping and resume are separate mechanisms

At each epoch, evaluate train and validation. Any strict improvement over best saves a new best model and validation predictions. Resetting patience uses a different reference: improvement must exceed reference + min_delta.

A small improvement can therefore become the saved best without resetting patience. The best epoch is chosen by validation, not by the last epoch or test score. Reloading the best state must reproduce every saved validation label; save/load consistency is checked explicitly.

The last.pt state is for continuation and includes optimizer, scheduler, history, random state and environment. The best directory is the selected inference state. They serve different purposes. Restoring only weights or repeating a seed does not recreate an optimizer/scheduler continuation; environment and source/input identity also matter.

Primary seed 20261013 selected epoch 18, train F1 about 0.9309 and validation 0.8173. The gap is still present. Explain it as a generalization gap, not proof of a single overfitting mechanism or evidence that all extended settings failed.

## 12. Why the grid selects a configuration mean, not the best seed

The grid is two readouts × two head rates × three seeds = twelve runs. Encoder LR is 5e-5 throughout. Open `tables -> Results/bert/validation_grid.csv` and map each row to its setting:

| Readout | Head LR | Mean validation Macro-F1 | Sample SD across three seeds |
| --- | --- | --- | --- |
| CLS | 1e-4 | 0.716420 | 0.006098 |
| CLS | 1e-3 | 0.722706 | 0.014712 |
| Span mean | 1e-4 | 0.818324 | 0.001129 |
| Span mean | 1e-3 | 0.817683 | 0.002171 |

The winner is span_mean/head 1e-4. The two span-mean means are close; ranking them by the declared rule does not establish a statistically reliable advantage of one rate. Within this extension, matched settings support the usefulness of the readout change. They do not isolate the reason for the entire initial-to-extension improvement.

select validates checkpoints/prediction checksums, checks code identity, gathers each declared seed and ranks the mean best-checkpoint validation F1. The first declared seed remains the primary. test verifies the locked selection/state/input identities before opening test rows.

The three-seed mean is not an ensemble prediction. No probability averaging or voting is implied. The primary BERT test F1 is 0.8146; the selected configuration's seed mean is 0.8182 ±0.0038. Replacing the primary with test-best seed 20261015 would introduce post-test selection.

BERT-minus-LR is about -0.0017 with paired CI [-0.0184,0.0182]. Neither superiority nor equivalence is established. The extension reused the holdout after classical results were available; this is not an independent new corpus evaluation.

## 13. Use real case 15390 to connect computation and uncertainty

Open its json entry. It is validation, exact span [83,88), gold SVA, LR_03/Tree_03 Vform. The visible features include VERB, VerbForm=Inf, a neighbouring AUX and singular third-person subject information.

Follow the actual chain: original span -> extracted features -> train-fitted coordinates -> learned class scores/tree route -> saved category. The public record establishes the mismatch, not the complete coefficients/branch route. Morphology describes a parser analysis; the teacher's category describes the edit. They need not have a one-to-one mapping.

A useful review question is whether surrounding auxiliary edits or a multi-operation correction affect interpretation. The public record does not include that correction, so do not turn the question into a new adjudication or claim a causal weight explanation.

### Practical evidence route for a natural presentation

Start with the real mismatch, explain why the POS baseline cannot fully solve it, then walk through LR and Tree. Use the ablation table to motivate representation limits. Move to BERT's explicit location markers, pooling choice, training loop and configuration selection. Finish with the primary comparison and required output handoff.

Optional original preflight from a prepared Code/ workspace:

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment check --config configs/bert_small_training.json
```

It checks actual inputs/forwards; it does not train the grid. Private inputs are required. Saved configuration/run tables are sufficient when filming the public package.

## 14. Questions to answer from understanding

| Question | Key reasoning |
| --- | --- |
| Why can LR work with more dimensions than samples? | Sparse engineered inputs and regularization constrain fitting; this is not a guarantee of generalization. |
| Why use cross-entropy if reporting Macro-F1? | It is a differentiable surrogate; validation selects the setting using the target metric. |
| Does balanced loss guarantee rare-class improvement? | It changes training influence, not the final precision/recall tradeoff in a guaranteed way. |
| Is Tree leaf voting weighted here? | Impurity is weighted when enabled, but leaf labels/minimum leaf counts are raw. |
| Is target masking a test-time string replacement? | It changes train candidates/encoder while keeping contexts and parse; variants are retrained/tuned. |
| Does span mean ignore the sentence? | Its averaged states are contextual and still attend to the input. |
| Why two learning rates? | Adapt the pretrained encoder differently from pooler/classifier; both groups learn. |
| Why can a checkpoint improve without patience resetting? | Best and min_delta stopping reference are different comparisons. |
| Why not choose the best test seed? | Test must not become a configuration/seed-selection set. |
| What does a checkpoint hash prove? | Identity of expected bytes, not availability, scientific validity or original authorship. |

For the handoff, identify checkpoint/encoder identity, chosen configurations, aligned predictions and validation records. Full private states are not in the public package. Your recorded review IDs are 12314, 14761, 14764, 12019, 14939 and 15390 under audit_summary.json -> validation -> member_ids -> Duy. Show one actual modelling/review/check action and its evidence; distinguish your work from shared/AI assistance and later assistant-executed validation.

**Technical explanation** should connect objectives, optimization and representation. **Evidence and demonstration** should use actual source, tables, curves and the real case. **Personal contribution** needs a supported action/decision/check. **Production quality** means readable equations/axes, accurate rate/seed/CI captions and a coherent narrative. Use these as understanding goals, not text to recite.
