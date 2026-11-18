# Quân - Presentation and code walkthrough guide: data and features

Project: **NUCLE-final-20261117**. Prepare and present in English. **No duration limit.** This is an explanation guide, not a speech to memorise. Use the reasoning, code pointers and evidence to explain the project in your own words.

## How to use this guide

For each topic, understand the problem, the design decision, the function implementing it and the evidence that supports the claim. During recording, choose a readable section of code and one example; explain the relationship rather than reading every line. Use the questions at the end to test your understanding.

[README](../README.md) gives environment/setup/run order. [Results/results.json](../Results/results.json) contains json, tables and texts objects. An entry such as `json -> Results/evidence/feature_manifest.json` is a key inside that bundle, not a separate file. The report is [Final_Report.pdf](../Report/Final_Report.pdf), and [DATA_README](../Data/DATA_README.md) includes all six licence-page images.

Keep saved evidence, source inspection, conceptual examples and actual execution distinct. No additional demo program or test file is needed. Full learner text and lexical artifacts require authorised access. Commands, when used, run from a prepared Code/ workspace, not an unprepared public copy.

## 1. The central methodological question

The data pipeline must convert a teacher annotation into a reliable model input without changing what was annotated or exposing the answer to the model.

An observation is one annotation, not one essay. Its predictor is an original sentence with a known character interval. Its target is one category: ArtOrDet, Prep, Nn, Vform or SVA. Several annotations may share a sentence. That creates dependence between rows and affects splitting and later uncertainty estimates.

The categories concern articles/determiners, prepositions, noun number, verb form and subject-verb agreement, respectively. A POS tag describes a token's grammatical role; an error label describes the annotated problem. For example, a VERB tag alone cannot distinguish a verb-form error from an agreement error. The correction/context is relevant to human interpretation even though the correction is excluded from predictor input.

The supplied error location is legitimate input for this task, but makes the task easier than detecting errors in arbitrary text. A good classifier here is not automatically a good detector or corrector. Keep that distinction in your opening explanation.

### Code map

| Stage | File / functions | Input -> output | Reason it matters |
| --- | --- | --- | --- |
| SGML parsing | [src/data/nucle_sgml.py](../Code/src/data/nucle_sgml.py): NucleSGMLParser, load_nucle | Raw text/stand-off annotations -> documents and mistakes | Preserve the coordinate system |
| Annotation table | [src/data/parse_nucle.py](../Code/src/data/parse_nucle.py): reconstruct_span, documents_to_dataframe | Document/mistake -> span, validity, target and audit fields | Separate malformed endpoints from linguistic questions |
| Sentence preparation | [src/data/prepare_nucle.py](../Code/src/data/prepare_nucle.py): segment_paragraph, construct_input, build_dataset | Paragraph coordinates -> sentence-relative sample | Preserve the original annotated substring |
| Boundary repair | [src/data/sentence_boundaries.py](../Code/src/data/sentence_boundaries.py): repaired_starts | Source text/token offsets -> sentence starts | Handle difficult punctuation without target-label information |
| Grouping/split | [src/data/group_split_nucle.py](../Code/src/data/group_split_nucle.py): build_grouping, UnionFind, stratified_group_assignment | Source-content graph -> partition manifest | Prevent related content crossing partitions |
| Linguistic analysis | [src/features/nucle_pos.py](../Code/src/features/nucle_pos.py): load_tagger, align_span | Original sentence/span -> token/POS alignment | Connect character annotations with token analyses |
| Features/encoding | [src/features/nucle_features.py](../Code/src/features/nucle_features.py): extract_features, TrainOnlyEncoder | Linguistic attributes -> fixed sparse coordinates | Learn vocabulary/categories only from train |

All files are under Code/. A useful high-level flow is:

`SGML -> checked annotation -> sentence/span -> group split -> fixed linguistic features -> train-fitted encoder -> vectors and separate targets`

## 2. Why the parser is a state machine, not generic text cleanup

Read NucleMistake/NucleDocument first, then follow handle_starttag, handle_data and handle_endtag. The parser maintains flags such as _inside_p and _inside_correction and separate buffers for paragraph/type/correction text.

When a DOC begins, it creates a document. A P or TITLE inside TEXT begins a paragraph-like block. MISTAKE creates an annotation with four endpoints. TYPE and CORRECTION accumulate into the current mistake. Closing blocks commits the buffers to their objects. If a new paragraph starts before the preceding one closes, the code finalises the previous block instead of silently merging them.

This is selective parsing: it extracts the structure needed for annotation coordinates. It is not a claim to implement every SGML convention.

### Two important details to point at

- `feed` protects ampersands before calling HTMLParser. HTML entity handling could otherwise change a literal source sequence and shift character positions. The protection and parser decoding must be understood together; seeing convert_charrefs=True alone does not explain the full behavior.
- Paragraph export strips formatting newlines at the ends with `strip("\n\r")`. That is narrower than stripping all whitespace. Ordinary spaces remain part of the parsed paragraph's offset contract.

Offsets are Python Unicode code-point positions in that parsed paragraph, not byte positions in markup. An accented character can use multiple UTF-8 bytes, and Unicode normalization can change a string's code-point sequence. Therefore replacing characters or globally cleaning whitespace after reading the annotations can point the same integers at different text.

**Explain naturally:** start with a concrete failure: removing one space before an error moves every later position. Then point to the parser's preservation choices. This makes the implementation decision meaningful without narrating every flag.

## 3. Separate endpoint validity, edit type and grammatical correctness

reconstruct_span checks paragraph indices and endpoint bounds. For a same-paragraph annotation, it slices `paragraph[start_off:end_off]`. A cross-paragraph annotation can have valid endpoints but no single-paragraph span; the returned flags distinguish those cases.

Do not equate a missing correction with an empty replacement. The missing_correction flag records whether a CORRECTION field was supplied. Empty replacement can be a deletion. These are audit distinctions, not class labels fed to the predictor.

The annotation_id is a sequential index under one exact source ordering. Its meaning is tied to the source checksum. A different source release may assign the same integer to another annotation. The sample identity therefore includes source hash plus annotation ID. Sentence identities additionally depend on sentence segmentation, so they can change between v1 and v2 while the source annotation remains the same.

The pipeline checks mechanics, not whether every teacher decision is uniquely justified. A valid slice is compatible with an ambiguous edit, an optional correction or several operations under one label.

**Evidence:** DATA_README records the exact nucle3.2.sgml checksum. The preparation summary records 44,912 annotations, 29,124 outside the five-class task and 15,788 retained targets. Filtering to five labels defines the study cohort; it is not a detector deciding whether other grammatical phenomena exist.

## 4. Understand sentence-relative conversion and its invariants

Read segment_paragraph before construct_input. The former creates source intervals for sentences and verifies that concatenating their text reproduces the parsed paragraph. construct_input finds a sentence containing the complete annotation interval, then subtracts its paragraph start.

Conceptual example, authored only for explanation:

`She walk to school. They has books.`

The second sentence starts at paragraph position 20. Paragraph span [25,28) selects `has`; its sentence-relative span is [5,8):

`25 - 20 = 5; 28 - 20 = 8`

The invariant is stronger than checking bounds:

`sentence[relative_start:relative_end] == paragraph[source_start:source_end]`

Then preparation also checks that the result equals the span reconstructed during data understanding. This gives a chain of consistent representations rather than one unchecked offset conversion.

A rejection reason such as cross_sentence means the input contract cannot represent that annotation as one sentence/span. It does not mean the error itself is linguistically invalid. The final retained target set has zero not-constructible cases, but that is an observed project count, not a guarantee for every future corpus.

### What v2 changes

v1 relies on the initial sentencizer starts. v2 calls repaired_starts and still slices the original characters. The repair API receives source text/token offsets, not labels, corrections or prediction errors.

The rules handle bounded citation-like regions, abbreviation continuations, opening brackets/quotes and punctuation attached to a following word. A boundary can fall inside an original tokenizer token; segment_paragraph explicitly constructs matching token fragments instead of editing the source to insert a space.

These are conservative heuristics, not a universal sentence grammar. Explain one boundary situation, the character-preservation check and one residual uncertainty. Do not claim the repair understands every citation or grammatical clause.

**Saved evidence:** preparation_comparison.json records 544 changed sentence inputs, 533 cases with changed sentence IDs, thirty changed context representations and two changed token-span representations across the same 15,788 targets. Annotation retention and source offsets are unchanged. The sets overlap. Changed context does not by itself prove a model-performance gain.

## 5. Character spans and token spans are different objects

align_span in nucle_pos.py selects every token with character overlap:

`token.start < error_end and token.end > error_start`

This is interval intersection, not a requirement that annotation boundaries exactly match token boundaries. A partial-token annotation therefore retains its character interval while receiving the analyses of overlapping tokens.

The code distinguishes exact, partial_token, whitespace_edge, whitespace_only and insertion-related states. These states tell the reader how character input relates to a tokenization. They do not redefine the teacher's span.

The final project's counts are 14,447 exact, 1,130 partial-token and 211 whitespace-edge cases. Ninety-six unresolved review flags and two boundary flags are separate concepts and must not be added as disjoint groups. The retained project spans are all nonempty even though some source functions support insertion boundaries.

A useful explanation is: the target is specified in character space; linguistic attributes live in token space; alignment documents the information lost or approximated when connecting them.

## 6. Why split a graph of related essays

Read build_grouping and UnionFind. Treat essays as graph nodes; duplicate/shared-content rules create edges. UnionFind computes connected components, so if A connects to B and B to C, all three stay together even when A and C are not directly identical.

Links come from exact/normalized complete essays, repeated normalized paragraphs and identical complete sentence-plus-span inputs. The code verifies actual tuples when grouping an input hash, rather than assuming a hash string alone establishes identity.

A narrow boilerplate exemption avoids giant components caused by short repeated noninformative content: at least ten essays, at most ten normalized words/eighty characters, with no retained target annotation in any occurrence. Seven exemptions are recorded. Identical model inputs still connect essays even if a paragraph qualifies for exemption.

Normalization is NFC, casefolding and collapsed whitespace for comparison. It does not replace the predictor text. This protects against exact/normalized content leakage, not every semantic paraphrase or shared author characteristic.

### How allocation works

stratified_group_assignment operates on whole components. A component carries a sample count and a count for each label. The greedy search considers the change in a partition's size/class-balance cost when adding a component. Class deviations have weight two in the implementation. Seeded order/ties vary across 128 restarts; candidate ranking prioritises avoiding missing classes and then lower objective.

Groups cannot be divided to hit perfect percentages. The target is approximately 70/15/15 while preserving the grouping constraint. Gold labels are used for stratification before modelling; that is different from feeding labels into predictors or choosing a split from test performance.

To explain the cost more precisely, let t be a partition's desired sample count and a its currently assigned count. The size term is `((a-t)/max(t,1))²`. Each class adds twice the same kind of relative-error term using that class's desired/actual counts. Relative deviations stop frequent labels from dominating purely through larger absolute counts. The greedy step chooses the smallest increase in cost, rather than the smallest absolute cost of a partly empty partition. The final restart objective also penalizes missing classes; restart ranking first minimizes the number of missing-class partitions. These are heuristics, not a proof of the globally optimal split. The saved split seed is 20261013.

| Partition | Target annotations | Represented essays | Represented groups |
| --- | --- | --- | --- |
| Train | 11,052 | 1,069 | 1,045 |
| Validation | 2,368 | 142 | 99 |
| Test | 2,368 | 144 | 97 |

The complete graph has 1,280 components over 1,397 source essays; some have no retained target samples. Do not confuse those counts with the supervised table, independent learners or the 97 test resampling units.

## 7. What the fixed spaCy analysis does and does not learn here

Preparation uses a blank tokenizer/sentencizer. Feature extraction uses spaCy 3.8.7/en_core_web_sm 3.8.0 with trained tagger/parser and disabled NER. These are different components with different roles.

The parser is already trained; the project does not fine-tune it on all corpus labels. Parsing all original sentences with a fixed analyser is different from learning a vocabulary from validation/test. The learned encoder must still be fitted only on train.

The code caches 11,272 distinct sentence analyses and extracts a separate row for every annotated location. Sharing a parse does not merge those samples or make them independent. Automatic morphology/dependencies can be wrong; a missing feature may mean an unavailable parser analysis, not absence of a real grammatical phenomenon.

## 8. Read each feature family as a representation choice

Use extract_features and real case 10824 side by side.

| Family | Actual mechanism | Information and limitation |
| --- | --- | --- |
| S | Token/character lengths, relative positions and span kind | Structural context; not semantic meaning |
| L | Casefolded unigrams/bigrams in separate span/left/right regions and lexical context slots | Direct lexical identity; can memorise dataset patterns |
| P | Ordered POS signature, context POS and local pattern | Grammatical category patterns; parser errors remain |
| M | Sorted unique Number/Person/Tense/VerbForm values from covered tokens | Compact morphology; loses token-specific attachment/order |
| D | Dependency/head signatures, linear distances and direct subjects | Relations beyond a two-token window; incomplete syntactic coverage |

lexical_candidates forms n-grams inside each region, not arbitrary bigrams crossing region boundaries. The left/right window is two tokens. The representation is not described as TF-IDF: lexical activations are binary and numeric values are kept as their actual numbers.

D deserves careful explanation. head_distance is mean absolute token-index distance to heads, not dependency-tree path length. Subjects are direct children of covered tokens with the specified subject relations; the implementation does not traverse every coordination or long dependency chain. subject_distance averages linear distances over covered-token/subject pairs. Signatures compress analyses; they cannot preserve every token-level relation.

Family names are not disjoint sources of linguistic information. D contains head/subject POS and Number/Person. Dropping P or M later removes explicit columns, not all traces of POS or morphology from the representation.

## 9. Understand fit, transform and sparse storage precisely

TrainOnlyEncoder.fit first requires nonempty, train-only rows and unique sample IDs. It collects numeric keys, observed categorical values and lexical term frequencies. Vocabulary/category ordering is deterministic. Missing, not-applicable and unknown categories receive explicit sentinels.

Three mechanisms become one vector:

- Numeric keys become N-prefixed coordinates with numeric values; there is no fitted standard-scaling stage here.
- Categorical keys become C-prefixed one-hot coordinates.
- The capped n-gram vocabulary becomes binary L coordinates; repeated occurrences do not increase the stored value above one.

The cap is 500 n-gram entries. It is not a cap on all L coordinates: lexical position slots also generate categories. L|span_unigrams is an intermediate field skipped by fit; do not incorrectly describe it as an additional bag-of-counts or whole-span one-hot feature. The actual Full dimension is 16,180.

transform_one uses the fixed index mapping. An unseen category activates __UNK__, not a newly added column. __MISSING__ means missing information; NOT_APPLICABLE and unseen-value handling have different meanings. Unknown n-grams activate a shared unknown coordinate. That fallback avoids shape errors but loses which unseen word occurred.

A sparse dictionary `{column_index: value}` is a coordinate representation, not a prediction. It reduces storage/computation when most entries are zero. A model state without the matching coordinate map is unusable even if the vector dimension happens to match.

For a small conceptual illustration, a raw `S|span_tokens` value of 2 becomes the value 2 at the coordinate named `N|S|span_tokens`. A known POS signature activates its categorical coordinate with value 1. If a later row has a signature absent from train, it activates that field's __UNK__ coordinate, still with value 1. Actual column integers come from the fitted encoder; do not invent them for a real case. This illustrates why numeric magnitude, category identity and column position are separate things.

### Leakage controls: explain their scope

extract_features explicitly reads sentence/span fields and checks document text and substring identity. Fit rejects non-train partitions. Features/targets are exported separately with IDs for joins. Keeping IDs for alignment is acceptable; using them as predictor values could memorise provenance.

The bare provenance field alignment is excluded as a generic input field, while a deliberately constructed P|span_alignment feature describes known span/token geometry. Do not claim that every occurrence of the word alignment is forbidden. A guard protects a specified contract, not every conceivable future wrapper mistake.

## 10. Connect the theory to real case 10824

Open `json -> Results/evidence/real_example_10824.json` and trace:

1. Source checksum + annotation ID identify the actual row in validation.
2. [44,53) is an exact nine-character interval, covering two tokens in a 27-token sentence.
3. The saved POS signature is DET+NOUN, morphology includes Number=Sing and dependency signature is det+nsubj.
4. Full uses the train-fitted encoder; the public record contains only selected nonlexical coordinates.
5. Both saved LR_03 and Tree_03 predict ArtOrDet, while gold is Nn.

This example explains why a multi-token POS signature is not identical to error type: the annotated category concerns the edit, not simply every POS inside its anchor. A DET coordinate does not prove why a particular fitted classifier chose ArtOrDet. Full lexical features, coefficients/branches and surrounding edit evidence would be needed for a stronger interpretation.

Do not invent a readable learner sentence to fill the redaction. Use the conceptual offset example separately and label it accurately.

## 11. A flexible recording route

Choose a narrative around one problem: preserving the annotated location while building a learnable representation.

- Start with the task contract and your role.
- Use the data guide/licence images to establish source and access; do not read six pages aloud or claim the form itself proves approval.
- Explain one parser preservation decision and the coordinate example.
- Move to v1/v2 evidence, then the graph/split table.
- Explain the five families through case 10824 and finish at train-only coordinates and the saved mismatch.
- Show one actual personal contribution and the data/features handoff.

An optional CLI view from a prepared Code/ workspace is:

```powershell
.venv/Scripts/python.exe -m src.data.prepare_nucle --help
```

It demonstrates real options, not a completed corpus run. Prefer completed run output when explaining retention counts. There is no need to launch the whole pipeline during a take or overwrite an existing output directory.

## 12. Questions you should be able to answer without memorising text

| Question | Reasoning your answer should contain |
| --- | --- |
| Why not clean all whitespace first? | Offsets refer to a particular character sequence; cleanup changes the target unless a verified coordinate mapping is maintained. |
| Why can a partial-token sample still be valid? | Character annotation and tokenization are different layers; record their relationship instead of changing the annotation. |
| Why graph components rather than essay IDs alone? | Shared content can connect distinct essays transitively. |
| Does grouping use labels? | Allocation uses label counts for stratification; source-only boundary repair does not use labels/corrections; neither puts labels into predictors. |
| Why is vocabulary fitted after splitting? | Test/validation content must not define learned categories or term selection. |
| Why more than 500 dimensions? | The cap applies to n-grams, not numeric/categorical/context coordinates. |
| Does __MISSING__ prove there is no subject? | It describes the available analysis/representation, not a grammatical truth. |
| Can the redacted real case rerun a checkpoint? | It lacks full lexical inputs/encoder/state; it supports traceability, not complete inference. |

For personal contribution, use an artifact from work actually performed: a parser/offset decision, a review correction or an independently executed check. The recorded assignments for Quân are 10824, 11696, 12352, 11501, 12050 and 12301 under audit_summary.json -> validation -> member_ids -> Quan. Those assignments are not a full authorship history. Attribute shared/AI work accurately.

## What makes the presentation convincing

**Technical explanation:** connect invariants, graph grouping and representation choices to code behavior. **Evidence and demonstration:** open actual bundle records, source functions and the real mismatch. **Personal contribution:** explain an actual decision/check with its artifact. **Production quality:** readable code/JSON, visible keys/intervals, clear audio and accurate English captions.

Speak from the logic: problem -> choice -> code mechanism -> observed evidence -> remaining limit. The guide is a reference for understanding, not a monologue or a fixed order to read.
