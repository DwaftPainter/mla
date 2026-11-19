# AI use and verification log

Project version: **NUCLE-final-20261117**. This log distinguishes recorded assistant work, historical experiment evidence and information that the students have not supplied.

## Known assistance in this session

| Task | AI/tool use | Output and checks |
| --- | --- | --- |
| Submission organisation | OpenAI Codex assistant; local Python/PowerShell tools | Read new_task.md, assembled required folders and relative-path instructions, selected text-free evidence and checked hashes |
| Report writing | AI-assisted English structure, prose and related-work synthesis | Report/Final_Report.pdf; editable drafts are retained outside the submission; tables checked against saved project artifacts, without invented experiments or scores |
| Literature/framework | Primary-source verification in earlier session work; inspection of the supplied Stevens manuscript | The literature-reading text and citation-policy JSON records in Results/results.json document reading scope, seven 2024-2026 sources and six older core/provenance exceptions |
| Aggregate reproduction | Assistant-authored existing reproduction and integrity helpers | An earlier assistant-authored aggregate wrapper and manifest utility were checked, then removed from the submission at the user's request; original evaluation modules remain |
| Individual video scripts | AI-assisted English narration, screen actions and production guidance | Three complete Markdown scripts, no duration cap; real cases 10824/15390/19068 and explicit saved/synthetic labels |
| Earlier optional demonstrations | AI-assisted synthetic helper programs were previously used and later removed at the user's request | Those programs are not part of the submitted ML pipeline; no benchmark was retrained by them |
| Submission reduction | Reviewed imports, documentation references and required evidence | Development tests, auxiliary scripts, duplicate drafts/PDFs, regenerated demo outputs and separate old validation logs omitted; no new code file added |

Relevant user instructions were to read new_task.md, keep the report truthful, include Related Work and cross-paper comparison, prioritize 2024-onward sources except core papers, and make detailed videos addressing technical explanation, evidence/demonstration, personal contribution and production quality. The later instruction was to reduce file count and consider omitting tests. The source assignment remains in the original working repository; its duplicate is omitted from this submission.

The exact assistant model identifier and full historical student AI conversations were not supplied and are not invented. Internal assistant deliberation is not represented as saved project evidence.

## Checks and corrections

The assistant recalculated the five headline systems, all twenty classical point results, twenty-five headline class records, twelve BERT validation runs and selected seed statistics. Report tables and selected claims pass 117 checks against saved artifacts. Final-layout command, link and portability checks are executed without adding submission validation files. Results/results.json preserves existing saved evidence under its original record names. CIs are matched to saved evidence, not regenerated from public aggregate matrices.

Before removing development files, the retained source suite returned **106 passed, 1 skipped**; the skip requires a licensed NUCLE fixture. An earlier focused video check returned fifteen passes. These are assistant-executed historical checks, not new checks by students. Their original test files/logs remain outside the slim submission. The user will record/add the videos independently; optional helper programs are no longer submitted.

A script arithmetic check caught an assistant-written offset error. The second authored sentence begins at paragraph position 20, so [25,28) becomes [5,8); the narration was corrected to the actual output. A later source review corrected Duy's BERT path to src/training/bert_span_classifier.py. No nonexistent implementation was fabricated to satisfy the original mistaken path.

The retained modelling/training algorithms, configuration and base uv.lock are preserved. Two original protocol-hash readers have an I/O fallback for protocol text moved into the evidence bundle; this location change does not alter training, selection or scoring. The submitted BERT supplement omits document-only python-docx; that documented dependency change does not alter the retained research Python. Old historical lock discrepancies remain disclosed, and no old hash was rewritten. Restricted learner text, corrections, complete lexical vectors and private checkpoints are not redistributed.

The supplied Stevens reporting paper was published in **2020**, while PMC availability was in 2021. Its structure is adapted retrospectively to this known-span NLP classification project; no clinical results, external evaluation, calibration or guideline-compliance claim was imported. Related Work is qualitative across differing tasks, inputs, labels and metrics; external papers were not rerun as baselines.

## Encoder model used in the actual classifier

Model: **google/bert_uncased_L-2_H-128_A-2**, revision **30b0a37ccaaa32f332884b96992754e246e48c5f**. It is a compact BERT encoder, not a generative chat model. Inputs contain original sentences with [SPAN_START]/[SPAN_END] markers. There is no natural-language instruction prompt or chat sampling temperature. Predictions are the argmax of five logits; gold labels and corrections do not enter predictor text.

The extension uses CLS/span-mean readout, encoder LR 5e-5, head LR 1e-4/1e-3 and seeds 20261013/20261014/20261015. Settings: up to twenty epochs, patience four, min_delta 0.0001, batch size 32, max length 128, dropout 0.1, weight decay 0.01, warmup 0.1, clipping 1.0, no class weighting and four CPU threads. Selection averages best-checkpoint validation Macro-F1 across the three seeds. Span mean/head LR 1e-4 is selected; primary seed stays 20261013. Settings and saved outputs are under Code/configs/ and Results/bert/; source revision/environment evidence is under Results/reproducibility/.

The initial failed BERT result remains in the evidence. The extension reused the holdout after classical test results were available; validation-based selection does not create an independent external test. Toy video training is separate from the frozen research predictors.

## Student additions that remain necessary

Assigned roles and six-case review assignments are documented, but a role is not an authorship history. Each student must use the README role table and their video guide's contribution prompts to prepare an action actually performed, supporting artifact, decision/correction, personal check and shared/AI assistance. Assistant checks are not personal checks by Quan, Duy or Cong. Actual recordings and full student identities have not been supplied.

For each historical AI use, add only information supported by your records: member/date, tool/model if known, affected task/file, saved prompt/output or accurate summary, personal check, correction/rejection and remaining uncertainty. Leave unknown details explicitly unknown. Do not invent a prompt, authorship claim, training run or numerical improvement.

## Required-layout rebuild

At the user's request, the final submission has exactly README.md and six folders: Report, Code, Results, Data, AI_Log and Video. Report contains the final PDF; Data contains the required guide; AI_Log contains this log; Video contains the three complete scripts. Actual student recordings remain pending. Development tests, manifests, package-check utilities/logs, duplicate guides, editable report drafts and contribution worksheets are outside the submission. No new code file was created.

The remaining experimental JSON records, CSV rows and aggregate review/literature texts are consolidated into Results/results.json, retaining original relative record keys and original pre-consolidation checksums. Results/main_results.csv is the unchanged saved headline CSV. The existing reproduction, inspection and synthetic-demo helpers were adjusted only to read this bundle. Retained research model/data/training/evaluation implementations were not refactored or retrained. No count, score, prediction, confidence interval or personal claim was invented by consolidation.

The report preserves Related Work and cross-paper comparison, prioritised recent references, fixed experimental conclusions and a short appendix. Its numeric/table checks passed 117 comparisons before the working drafts/check artifacts were moved outside the delivery. The original approved-access limitations and historical lock discrepancies remain visible in the report and evidence records.

## Removal of auxiliary demo files and Code/docs

The user asked why the Code folder contained documentation and three helper programs, then confirmed that video recording will be handled independently. The assistant omitted video_lab.py and inspect_evidence.py and stopped preparing video demonstrations. reproduce_main_result.py is retained and shortened to the explicitly required main-result reproduction: five confusion-count calculations, class-support checks and saved-CSV agreement, producing a single recalculated CSV. HTML/demo generation and unrelated helper checks are removed. No new code file was added.

The two docs files were original experiment protocols, not arbitrary report drafts. Their exact UTF-8 byte content and original SHA-256 values are stored under protocols in Results/results.json. Only the existing sha readers in nucle_final_selection.py and bert_span_classifier.py were adjusted to accept and verify those archived protocol bytes when the old paths are absent. Other missing files still fail. These source edits are disclosed; they do not claim byte-identical historical source replay or modify any experimental score/decision. The old docs directory is omitted from Code/.

## Removal of the separate reproduction wrapper

The user explicitly requested deleting reproduce_main_result.py. It was deleted from Code/. The earlier description that its separate filename was required was an overinterpretation: new_task.md asks the README to identify a result-reproduction script/notebook, and an existing project module can fulfil that role. README and the report now identify src/evaluation/nucle_final_evaluation.py and state its private-input/frozen-run prerequisites. No replacement helper file was created and no model was retrained. Earlier checks recorded in this log are historical assistant checks, not capabilities of a currently bundled wrapper.

## Isolated execution check - 18 November 2026

The assistant created a new uv environment in a separate copy under .cache/submission_runtime_20261118/, with Python 3.13.7, PyTorch 2.14.0+cpu, spaCy 3.8.7/en_core_web_sm 3.8.0 and Transformers 4.57.6. include-system-site-packages is false; the previous project environment was not reused. Base dependencies came from the submitted lock; the submitted BERT supplement and pytest were installed into this testing environment. Network permission was needed to download packages.

The relevant existing source tests returned **103 passed, 1 skipped**. The skipped module needs the licensed boundary fixture. Three tests for the omitted downloader were not included. One copied legacy test imported SGML_FILE from that omitted downloader; its local test import was corrected to the identical canonical constant in src.data.constants. Submitted source/test files were not added or changed by this adjustment.

All thirteen documented classical stages ran successfully on the real local corpus in the isolated workspace: understanding, quality, preparation, split, POS, features, benchmark, ablation, final selection, freeze, test scoring, bootstrap and error audit. Prepared samples, split, vectors, targets and encoder were byte-identical to the originals. All twelve selected ablation configurations and 28,416 validation predictions matched; one historical F1 JSON value differs only by about 1e-16. The final stage completed 52 fits. All sixteen generated checkpoint files were byte-identical to the originals. Across twenty named pipelines and 2,368 test annotations, all 47,360 predicted labels matched. LR Full reproduced Macro-F1 0.8162577160417342. All thirty compared paired CI rows matched, with zero endpoint difference. The checkpoint replay command printed CHECKPOINT REPLAY EXACT, and a repeat score command was correctly rejected without changing predictions.

BERT preflight passed for both readouts with finite logits and no test rows loaded. A separate one-epoch, one-variant, one-seed runtime smoke run completed training, saved/reloaded its best checkpoint and verified every validation prediction before locking its selection. It is not a replacement research result. The three existing selected BERT checkpoints were also evaluated read-only in the new environment: each reproduced all 2,368 saved validation labels with zero disagreements. The full twelve-run BERT training grid was not retrained in this check, and no new BERT test comparison was opened.

Execution logs, the environment freeze, test XML, artifact comparisons and validation_summary.json remain under .cache/submission_runtime_20261118/, outside the submitted folders. Original research inputs/outputs, the report's numerical findings and submitted Python sources were not overwritten. These are assistant-executed runtime checks, not checks personally attested by the students.

## Video-script alignment with the final submission

The assistant rewrote the three existing English video scripts against the final folder layout and consolidated evidence. All calls to removed helper programs, test modules and generated demo pages were removed. The scripts retain 12/14/14 detailed scenes, no duration cap, real validation cases 10824/15390/19068, source walkthroughs, visuals, truthful personal-contribution prompts and the four marking criteria. Optional live commands use existing project modules and state their actual private-input/lock requirements; saved records and conceptual illustrations are labelled separately.

Source-file paths, bundle-record keys, local links, the worked Nn calculation and the six licence-page images were checked. Case-specific interpretations not established by public metadata were rewritten as review questions rather than attributed to an unrelated aggregate review record. Quân's licence scene uses the six existing images; broken local PDF links in DATA_README.md were corrected without adding a PDF or code file. No recordings or new helper/test files were created, and project measurements were not changed.


## Video guides expanded for understanding - 18 November 2026

At the user's request, the assistant replaced the three existing video-script monologues with English presentation and code walkthrough guides. The new guides explain the methods, actual functions and data flow, design choices, worked calculations, limitations, evidence to display and questions for checking understanding. They retain no duration cap and cover technical explanation, evidence/demonstration, supported personal contribution and production quality. They are preparation references for natural explanation, not first-person speeches to memorise.

The data guide follows source-preserving parsing, coordinate invariants, graph splitting and train-only sparse encoding. The modelling guide explains loss/optimization, Tree behavior, masking, BERT attention/pooling, training and selection. The evaluation guide derives scores and vectorized paired group bootstrap, distinguishes audit selectors and explains reproducibility scope. The assistant checked 56 referenced source symbols, local links, bundle keys and the displayed numerical examples against retained code/evidence. No model was rerun for this documentation edit. Research measurements, gold labels and Python sources were unchanged; no helper, test or recording file was added. Student authorship and personal checks are not inferred from these assistant-written guides.

## Code and visual evidence embedded in video guides

At the user's request, the assistant added eighteen literal Python source excerpts directly to the three existing guides, with source links and explanations of which operations to highlight. The guides also embed all five existing result figures: the primary learning curve in Duy's guide and the benchmark, confusion matrices, per-class F1 and paired differences in Cong's guide. Two Mermaid diagrams explain the data pipeline and BERT readout. Quan's guide adds a preparation-results table and selected unchanged fields from real validation case 10824; Cong's adds the saved headline table.

All eighteen Python excerpts were checked against retained source after normalizing indentation. Fifty-seven relative links, Markdown fences, bundle keys, the displayed real-case values and the rounded headline table were checked. Existing figures were visually inspected before adding interpretation captions. Conceptual diagrams are identified separately from saved experimental measurements. No Python source, configuration, research result, figure image or checkpoint was changed; no helper, test, image or video file was created. These are assistant-performed documentation checks, not student contribution attestations or new training/evaluation runs.

## Flexible scripts for the required real-example demonstrations

The assistant expanded the existing three guides with eight-step presentation routes: screen actions, optional English speaking cues, annotated data-to-prediction tables, saved-output demonstrations, result/error interpretation and supported personal-contribution prompts. The case IDs remain 10824 for Quan, 15390 for Duy and 19068 for Cong. Each route maps its steps to the requested real-example, visual-explanation, demonstration and interpretation requirements. Speaking cues are adaptable and have no duration cap.

The assistant executed all three documented read-only PowerShell case views from the submission root; each displayed the intended existing JSON record. The headline CSV view also ran successfully. Case 19068 is explicitly separated from test confusion counts because it belongs to validation; no BERT case prediction is invented for case 15390. The scripts distinguish source walkthroughs, saved outputs and fresh model execution, and do not infer students' actual actions from assigned roles. No research runner or model was rerun for this addition, and no code, notebook, image, helper or recording file was added.

## Complete spoken scripts for the entire presentations

The user subsequently requested full scripts for all methodology and code explanations, superseding the earlier preference for short optional speaking cues. The assistant reorganized the same three Markdown files into 39 sequential spoken steps: eleven for Quan, fourteen for Duy and fourteen for Cong. Each step has a screen/action direction, a complete English spoken passage and the retained technical notes where relevant. The shorter eight-step demonstration routes were replaced; their real-case flow tables and read-only views were retained within the full sequence. Cong's scoring walkthrough now precedes metric derivation and result interpretation.

The scripts cover the complete assigned scope, source/code mechanisms, saved-result demonstrations, actual cases, interpretation and limitations. They retain all eighteen literal Python excerpts, five existing result figures and two diagrams. Related Work is discussed using the report's already recorded task/input comparisons and inspection scope; no new external-paper experiment or score was invented. Only the personal-contribution passages have explicit placeholders, because individual actions and verification history cannot be inferred from assigned roles or assistant checks.

Documentation checks confirmed all 39 sequential steps and matching spoken/action blocks, eighteen source-matching Python excerpts, 56 relative links, Markdown fences, bundle keys, real-case values and the saved headline/Nn calculations. Obsolete statements saying that only optional cues were provided were removed. No research code, configuration, result, figure image or checkpoint was changed, and no new file or model run was created for this revision. These are assistant-written scripts and assistant-performed consistency checks; students must verify any personal-contribution statement before recording.


## Single preparation policy and canonical runtime names

The user requested removing the obsolete preparation-version comparison from the delivered code. The assistant deleted Code/src/data/compare_preparation.py, removed the legacy preparation branch and --policy selector, and retained one source-preserving segmentation path. Preparation uses nucle_preparation; dependent feature/POS/benchmark/ablation/diagnostic paths and active schema/policy identifiers now use canonical names without numbered suffixes. The BERT sample configuration, README, data guide, report and video scripts were aligned. The user's pre-existing README edit was preserved. No replacement helper or test file was added to Code; it now has thirty Python files.

Original evidence records, protocols and their historical checksums were not relabeled or rewritten. Results/results.json adds current_pipeline metadata to distinguish current names and validation from archival identifiers. Old locks do not become valid by changing a path; the validation workspace created a fresh matching lock. The report and Quan's script now discuss single-dataset preparation checks instead of requiring or claiming a fresh comparison with an absent earlier dataset.

Using the previously created independent environment, the assistant ran six stages in a fresh canonical workspace under .cache/canonical_pipeline_20261118/: understanding, quality, preparation, grouped split, POS and features. Samples, sentences, split/essay-group manifests, feature rows and targets were byte-identical to the verified final run. Encoder fields were identical except for the schema identifier. CLI checks confirmed that --help has no policy selector, the removed option is rejected and the comparison module is absent.

Existing fitted states were restored for fresh-lock scoring, grouped bootstrap and checkpoint replay; no refitting was performed. All 47,360 test-label comparisons across twenty named pipelines matched, and the prediction CSV, complete test metrics and bootstrap JSON matched the verified outputs exactly. LR Full test Macro-F1 remains 0.8162577160417342. The original replay command passed. The BERT preflight also passed on actual canonical train/validation inputs, without training or test loading.

Relevant existing regression modules were copied only to the external validation workspace. The obsolete comparison-check module and omitted downloader tests were excluded. Copied fixtures were updated for the single retained boundary policy: quotation and ellipsis expectations were legacy-branch expectations, with the ellipsis behavior also checked against the historical final default. The removed-policy test now checks rejection of the obsolete argument; copied feature-schema fixtures use the canonical identifier, and one legacy downloader import uses the same constant from src.data.constants. The final run returned 102 passed, 1 skipped, with the missing licensed boundary fixture causing the skip. An initial run exposed the legacy ellipsis expectation; it was corrected in the external test copy, without changing segmentation to satisfy an obsolete expectation. Four upstream warnings remain.

These are assistant-executed checks, not student attestations. Detailed pipeline, comparison, CLI, preflight and test records remain outside the submission under .cache/canonical_pipeline_20261118/validation/. The research measurements and fitted checkpoint bytes were not changed. Source hashes and provenance metadata change with the code/policy naming cleanup, so this revision does not claim byte-identical replay of an old source lock.

Tracked LF/CRLF styles were preserved to keep the diff focused; Python AST and JSON values were checked unchanged by this formatting correction. A separate exact-final-source replay copy then created another fresh lock and reran scoring/bootstrap/replay against the existing canonical prepared inputs and fitted states. Predictions, metrics and bootstrap again matched. The original validation workspace and historical locks were left intact.
