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

Assigned roles and six-case review assignments are documented, but a role is not an authorship history. Each student must use README.md's personal-contribution checklist to prepare an action actually performed, supporting artifact, decision/correction, personal check and shared/AI assistance. Assistant checks are not personal checks by Quan, Duy or Cong. Actual recordings and full student identities have not been supplied.

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
