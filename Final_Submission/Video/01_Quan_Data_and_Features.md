# Quân — Complete individual video script: data and feature engineering

Project version: **NUCLE-final-20261117**. Language: English. There is **no target duration or time budget**. Record the scenes in order and give each explanation and demonstration the time it needs. This is a complete script, not a short presentation with optional technical inserts.

All JSON/CSV basenames below refer to named records in Results/results.json. Use its json, tables or texts section, or the included evidence-view commands; these records are not separate files in the submission. Common setup, recording guidance and the personal-contribution checklist are in [README.md](../README.md).

## Recording brief and evidence boundaries

The assigned scope comes from new_task.md: read NUCLE/SGML, validate annotations and offsets, explain preparation v1/v2, split train/validation/test, run spaCy, extract S/L/P/M/D, fit the encoder on train, and produce vectors. The final handoff must identify prepared/split data, vectors, encoder and feature documentation.

The video combines two evidence types. **REAL SAVED EVIDENCE** uses the final project's aggregate artifacts and real annotation 10824. **SYNTHETIC LIVE EXECUTION** runs actual project functions on newly authored text, allowing readable text, feature values and failures to be shown without redistributing learner essays. Keep these labels visible. A synthetic run is not a rerun of the research pipeline or its published score.

Use a Python interpreter with the prepared research dependencies for the live lab commands. All paths below are relative to the submission root. The standard-library evidence commands also work without those dependencies. See README.md at the submission root for environment preparation and the exact production workflow. Before recording, complete your personal account using the checklist in README.md; choose only personal claims supported by your actual work.

## Scene 1 — Introduce the problem and define your scope

**Show:** face camera, then the README's task definition and responsibility row. Show the project version as a lower-third caption.

**Do:** point to the five output labels and the four input fields. Do not begin with a long repository tour.

**Narration:**

“Hello, I am Quân. My assigned part of this project is the data and feature pipeline. I will explain how an annotation in NUCLE becomes a valid sample, how we keep related essays in the same partition, and how an original sentence becomes a vector for the classifier.

“Our task is five-class grammatical error-type classification at an already known location. The output is ArtOrDet, Prep, Nn, Vform or SVA. We are not asking this model to find the error location or generate a corrected sentence. One annotation is one sample, so a sentence can contribute several related samples.

“I will use real saved case 10824 to connect the stages. I will also execute the actual data functions on a clearly marked synthetic example so that the text, offsets and vectorization are visible.”

**Completion cue:** a viewer can state the input, output and the part assigned to you before you move on.

## Scene 2 — Explain the source, version and annotation schema

**Show:** Data/DATA_README.md; Code/src/data/nucle_sgml.py, classes NucleDocument/NucleMistake and NucleSGMLParser; Code/src/data/parse_nucle.py, reconstruct_span.

**Do:** locate DOC/@nid, paragraph text, MISTAKE's four endpoints, TYPE, CORRECTION and the missing_correction flag. Keep the code at readable zoom and highlight one field at a time.

**Narration:**

“The experimental source is nucle3.2.sgml, identified by the SHA-256 checksum recorded here. A different file with a similar name would not necessarily reproduce the same IDs or samples. Our saved project records contain 1,397 essays and 44,912 annotations before restricting the task to five labels.

“NUCLE stores text separately from its annotations. A mistake supplies a start paragraph, start character position, end paragraph and end character position. TYPE supplies the category. CORRECTION is useful for an audit, but it is not an input to our classifier. A missing correction and an explicitly empty correction are also different: an empty replacement may represent deletion.

“This parser collects the fields we need while preserving the paragraph characters used by its offset contract. It handles literal ampersands and formatting newlines as specified in the implementation. The contract is defined on the parsed paragraph, not on byte positions in the SGML markup. Preparation must preserve that parsed text rather than normalize it again.”

**Evidence to interpret:** the code shows why ordinary HTML cleanup, whitespace stripping or applying corrections first would be unsafe for character offsets. Do not claim that every possible SGML edge case is solved merely because the parser runs.

## Scene 3 — Execute SGML parsing and trace paragraph-to-sentence offsets

**Show:** terminal and the lab source Code/video_lab.py, function pipeline. Add the caption “SYNTHETIC LIVE EXECUTION — actual project functions”.

**Run:**

```powershell
python Code/video_lab.py pipeline
```

**Do:** stop at the first output section. Show the authored SGML and its two annotations. Then open Results/reproduced/video_lab/synthetic_data_pipeline.json and expand the two details entries. Highlight paragraph_span, sentence_start_par, error_start/error_end and error_span together.

**Narration:**

“This small SGML document is newly authored for the demonstration; it is not NUCLE. The command imports the project's parser, reconstruct_span, sentence preparation and feature functions. The first paragraph is ‘She walk to school. They has books.’ The annotations select ‘walk’ and ‘has’.

“For the first annotation, paragraph positions four to eight select ‘walk’. The end position is excluded. For the second annotation, the paragraph interval is twenty-five to twenty-eight. Its sentence begins at paragraph position twenty, so the sentence-relative interval becomes five to eight. The selected characters are still ‘has’.

“This is the transformation I need to preserve: subtract the sentence start from both endpoints, then check that the paragraph slice and the sentence slice agree. Passing a bounds test alone is not enough. The interval could lie within the sentence while selecting the wrong text. The round-trip substring check catches that kind of mistake.”

**Completion cue:** show the same selected word before and after the offset conversion. Do not rush past the numbers; annotate 25 − 20 = 5 and 28 − 20 = 8 on screen.

## Scene 4 — Explain and demonstrate preparation v1 versus v2

**Show:** the lab's v1_v2 output; Code/src/data/prepare_nucle.py, segment_paragraph; Code/src/data/sentence_boundaries.py, repaired_starts. Then open Results/results.json entry json["Results/evidence/preparation_comparison.json"].

**Do:** compare the two arrays for the synthetic citation example. Highlight the boundary after “2009).” and before “The”. Show the round-trip assertion in the helper, followed by the real comparison counts.

**Narration:**

“The synthetic citation example contains punctuation attached to the next word. In the output I am showing, v1 leaves one sentence, while v2 creates a boundary before ‘The’. Both versions preserve all characters; joining their sentence slices recovers the original paragraph exactly. We are changing sentence intervals, not editing the learner's text.

“The real saved comparison is separate from this small illustration. Across the same 15,788 target annotations, 544 sentence inputs change, 533 sentence IDs change, 30 context representations change, and two token-span representations change. Those sets overlap, so adding them would double-count cases. Source metadata and source-span offsets remain unchanged.

“Boundary rules read original text rather than gold categories, corrections or reviewer conclusions. A difficult boundary is not an excuse to use the answer label when constructing the input. The two unresolved boundary cases are retained and flagged.”

**Interpretation:** v2 changes what context a sample sees without silently relabeling it. Do not attribute a later model improvement to this change without a matching controlled comparison.

## Scene 5 — Interpret preparation quality and retention evidence

**Show:** Results/results.json entry json["Results/evidence/preparation_summary.json"] and the report's annotation disposition table.

**Do:** highlight 44,912 total, 29,124 out of scope, 15,788 retained, and zero not_constructible. Then show alignment and review-status fields as separate objects.

**Narration:**

“Filtering is defined by the five target TYPE values. We do not relabel all other preposition-related categories into Prep. The saved counts show 29,124 annotations outside this task and 15,788 retained target samples. No target sample is counted as unconstructible in the final preparation.

“Alignment is exact for 14,447 cases, partial-token for 1,130 and whitespace-edge for 211. A partial-token annotation can be valid at character level even though its covering token is wider. We preserve the annotation and record the mismatch instead of pretending that the full token was the original gold span.

“The 96 unresolved review cases and two unresolved boundary cases describe different flags. They are not disjoint groups to add together. Retaining these cases makes the limitation visible rather than deleting difficult examples after seeing model errors.”

**Completion cue:** identify one mechanical guarantee and one remaining uncertainty. A correct substring invariant does not certify the grammatical necessity or category of every edit.

## Scene 6 — Show the connected-group split and explain leakage prevention

**Show:** Code/src/data/group_split_nucle.py, build_grouping and UnionFind; the split view.

**Run:**

```powershell
python Code/inspect_evidence.py split
```

**Do:** first highlight sample counts, then essay counts, then group counts. Draw or annotate A–B–C as a conceptual connection chain; label this an illustration rather than an actual group in the dataset.

**Narration:**

“Randomly splitting annotation rows would allow multiple errors from the same sentence or essay into different partitions. Splitting essays alone is also insufficient when essays share relevant content. The code builds connected components from the documented duplicate and shared-paragraph rules. If A connects to B and B connects to C, all three must stay together.

“Grouping comparisons use Unicode normalization, casefolding and collapsed whitespace. That normalized comparison text is not substituted for the original predictor sentence. A narrow boilerplate exemption handles short content shared by at least ten essays, with at most ten words and eighty characters, and no target annotation. Identical sentence-plus-span inputs always connect essays.

“The final assignment uses seed 20261013. We have 11,052 train annotations, 2,368 validation and 2,368 test. The test contains 144 essays in 97 connected groups. Counts of annotations, essays, groups and independent learners are not interchangeable. The split balances labels approximately while keeping components intact; it does not detect semantic paraphrases.”

**Evidence to interpret:** the published grouping summary records seven boilerplate exemptions and 1,280 connected groups. Show these saved fields rather than inventing a new split during the recording.

## Scene 7 — Distinguish sentence preparation from trained linguistic analysis

**Show:** Code/src/data/prepare_nucle.py, make_nlp, and Code/src/features/nucle_pos.py, load_tagger. Return to the live lab's extracted features.

**Do:** point to spacy.blank/sentencizer in preparation and the loaded en_core_web_sm tagger/parser in feature extraction.

**Narration:**

“There are two different uses of spaCy here. Sentence preparation uses the fixed tokenizer/sentencizer and source-only boundary rules. Feature extraction uses the trained English package en_core_web_sm version 3.8.0 with spaCy 3.8.7. Its tagger and parser provide linguistic attributes, while named-entity recognition is disabled.

“The saved feature manifest records 11,272 distinct original sentences parsed. Reusing a sentence parse for several annotations does not merge those annotations: each location still gets its own span and context features. Automatic POS, morphology and dependencies are observations from a model, not expert gold linguistic analyses.”

**Completion cue:** make clear which component decides sentence intervals and which supplies linguistic features. Do not use the terms sentencizer and dependency parser as synonyms.

## Scene 8 — Explain all five feature families using visible values

**Show:** Code/src/features/nucle_features.py, extract_features; live synthetic details/features; then the real case's nonlexical coordinates.

**Run:**

```powershell
python Code/video_lab.py case --annotation-id 10824
```

**Narration:**

“S is structural: span length, number of covering tokens, sentence length and relative position. Real case 10824 has a nine-character span covering two tokens in a 27-token sentence.

“L contains lowercased unigrams and bigrams in the span and separate two-token left and right regions, plus position-specific lexical categories. The readable lexical values in the live lab belong only to the authored synthetic text. The real learner's lexical coordinates are excluded from the public demonstration.

“P represents the span POS signature, local pattern and alignment. The real case's active signature is DET plus NOUN. M records Number, Person, Tense and VerbForm; this example contains a singular Number category. D records dependency signatures, head information, distances and direct subject information. Here the saved dependency signature is det plus nsubj.

“A feature coordinate with value one is an encoded category, not a probability. Also, D contains some POS and morphology of heads or subjects. Dropping P or M in an ablation therefore does not remove every representation of that kind of linguistic information.”

**Do:** point to one actual S coordinate, one P coordinate and one missing/unknown coordinate. Explain each before scrolling. Avoid reading every JSON key aloud.

## Scene 9 — Fit on train, transform validation, and show the sparse vectors

**Show:** TrainOnlyEncoder.fit and transform_one. Open the lab's synthetic_encoder.json and synthetic_vectors.json in separate tabs.

**Do:** show fit_ids containing only synthetic-train-0 and synthetic-train-1. Match an active validation coordinate to the feature-name order. Highlight the live demo dimension and then the saved benchmark dimension in Results/results.json entry json["Results/evidence/feature_manifest.json"].

**Narration:**

“The encoder is fitted before validation transformation. It collects numeric keys and train-observed categories, adds the missing/not-applicable/unknown fallbacks, and keeps a deterministic coordinate order. The lexical n-gram vocabulary is limited to the most frequent five hundred train terms with deterministic tie breaking. That limit is not the entire L dimension, because local categorical lexical slots also contribute columns.

“The validation illustration is ‘A cat eat fish.’ Transform uses the same fitted vocabulary and category order. It can activate unknown coordinates, but it cannot add a vocabulary entry because a validation word is new. The saved feature-name list remains unchanged.

“The dimension printed by this live run belongs to an encoder trained on two synthetic rows. It must not be described as the published encoder. The actual project's Full representation has 16,180 dimensions. Both examples demonstrate the same fit/transform contract, but they use different data and fitted parameters.

“Sparse storage keeps coordinate/value pairs instead of filling every zero explicitly. The real export writes features.jsonl and targets.jsonl separately, so the target label is not silently treated as an input coordinate.”

**Completion cue:** viewers can trace raw features to an index/value vector and explain why the fitted encoder must accompany a trained classifier.

## Scene 10 — Execute the guards instead of only claiming leakage safety

**Show:** the final checks printed by the live pipeline; the corresponding fit/extraction guards in the source.

**Run, using the prepared research interpreter:**

```powershell
python Code/video_lab.py pipeline
```

**Narration:**

“The lab deliberately attempts to fit an encoder on a validation row and confirms that the project rejects it. It also changes label, correction and sample ID while preserving the original input, and checks that the features remain identical. Finally, it supplies the wrong saved substring and confirms that extraction fails.

“These checks exercise specific failure conditions. They are stronger than saying that I remembered not to leak a label. The live pipeline also shows train-scoped unknown handling and offset failure on authored inputs. Its successful output does not prove that every corpus annotation or parse is semantically correct.”

**Do:** show the actual lab output from your run; the earlier scene's output can remain open. Do not present historical development-test counts as new lab output. Keep the expected rejection messages visible and explain that they are successful guard demonstrations.

## Scene 11 — Complete the real example's data-to-prediction trace

**Show:** case 10824, then Results/reproduced/demo.html's real-case panel and main result table. Keep a “REAL SAVED VALIDATION EVIDENCE” label visible.

**Run:**

```powershell
python Code/reproduce_main_result.py
```

**Narration:**

“Now I return to the real project example. Annotation 10824 is in validation. Its original span becomes a valid prepared sample, is analysed by spaCy, and is transformed with the train-fitted encoder. The saved LR Full and Tree Full predictions are ArtOrDet, while source gold is Nn.

“This is a real saved mismatch, not a new prediction from the redacted JSON. The full lexical vector and trained checkpoint are necessary for live benchmark inference. The two-token DET-plus-NOUN anchor shows why a POS signature alone does not identify an error type, but it does not prove that a particular coordinate caused the wrong decision.

“The reproduction command checks aggregate results and regenerates the browser evidence. It is useful for demonstrating consistency, while its scope remains different from rerunning the raw data pipeline.”

**Completion cue:** verbally name every link in the real trace and distinguish observed label mismatch from a causal explanation.

## Scene 12 — Identify outputs, personal contribution and handoff

**Show:** the personal contribution account completed from your own records, the saved member-ID assignment, and the output map below.

| Required output from new_task.md | What you should show |
| --- | --- |
| Prepared and split data | Preparation/split summaries, source hash and the restricted artifact paths described in DATA_README |
| Feature vectors | Live synthetic vector plus actual benchmark feature manifest; explain the restricted full export |
| Fitted encoder | Live synthetic encoder; original export schema and benchmark hash/reference |
| Feature description | S/L/P/M/D extraction code, representation table and one interpreted coordinate |

**Narration, supported by the existing record:**

“My assigned scope is the data and feature stage. The saved validation-review assignment lists cases 10824, 11696, 12352, 11501, 12050 and 12301 under my name. In this demonstration I have traced case 10824 and executed the offset and train-only checks on authored data.”

**Personal claim to prepare before recording:** add one specific action you actually performed, one decision or correction you made, and the file/log/review evidence supporting it. If you implemented a component, name that component accurately. If you reviewed or verified it, describe that work instead of claiming authorship. Use the wording choices in the personal-contribution checklist in README.md; do not read an unfilled template as a factual claim.

**Closing narration:**

“The handoff to modelling consists of a fixed sample/split identity, feature and target records, a train-fitted encoder and the feature contract. A model must receive the matching coordinate order. The important limits are that gold locations are supplied, parses can be imperfect, and source-preserving mechanics do not resolve every annotation ambiguity. Those limits carry into the modelling and evaluation stages.”

## Recording notes tied to the four marking criteria

| Criterion | Observable evidence in this recording |
| --- | --- |
| Technical explanation | Paragraph-to-sentence arithmetic; v1/v2 difference; group leakage; all five feature families; fit versus transform and sparse coordinates |
| Evidence and demonstration | Executed pipeline and guard assertions, real case 10824, saved count/feature tables, regenerated result output |
| Personal contribution | Named role plus six-case record, a truthful personal action/decision/correction and supporting artifact |
| Production quality | Consistent scope labels, readable code/terminal, clear numerical callouts, quiet audio, accurate English captions and evidence-based chapter transitions |

Use chapter titles matching these scenes. Hold each highlighted item until the explanation is complete. Record code, output and interpretation in the same segment where practical; if a loading pause is cut, preserve the command and its result. Do not add a duration cap during editing. End when the required reasoning and evidence are complete.
