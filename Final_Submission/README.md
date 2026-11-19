# Group 05 - NUCLE Error-Type Classification

Project version: **NUCLE-final-20261117**. Given an original English sentence and a known error span, the project predicts one of five labels: ArtOrDet, Prep, Nn, Vform and SVA. The final report is [Report/Final_Report.pdf](Report/Final_Report.pdf).

## 1. Group members and contributions

| Member | Contribution |
| --- | --- |
| Quân | NUCLE/SGML parsing; annotation and offset checks; source-preserving preparation; grouped train/validation/test split; spaCy analysis; S/L/P/M/D feature extraction; train-only encoder and feature vectors. |
| Duy | Majority and POS baselines; Logistic Regression and Decision Tree; validation tuning; ablation and lexical masking; initial BERT-small experiments, extended training and diagnosis. |
| Công | Test metrics, confusion matrices and grouped bootstrap intervals; model/ablation comparisons; synthesis of the 18-validation/40-test case reviews; reproducibility checks and report analysis. |


## 2. Setup instructions

The recorded experiments used Windows, Python **3.13.7** and CPU execution. Install Python and uv, then run these PowerShell commands from the submission root:

```powershell
cd Code
uv sync --locked --no-dev --python 3.13.7
uv pip install --python .venv/Scripts/python.exe -r requirements-bert.txt
```

Run all subsequent commands from **Code/**, using `.venv/Scripts/python.exe`. The base dependencies are specified in pyproject.toml and uv.lock; requirements-bert.txt pins the BERT supplement. The supplement is not a complete transitive environment lock.

Before running the research pipeline:

1. Obtain authorised NUCLE access and follow [Data/DATA_README.md](Data/DATA_README.md). The exact experimental source is nucle3.2.sgml; place it at `Code/data_set/raw/nucle3.2.sgml` and verify the documented SHA-256.
2. Restore the original permitted annotation-review/overlap decisions and any historical inputs required by the selected stage. Raw SGML alone does not recreate human review decisions. Full vectors, lexical encoders, checkpoints and row-level predictions are not included in this public submission.
3. For BERT, restore the pinned pretrained snapshot and the relative local path in `Code/reports/contextual_model_validation/model_source.json`. The model identifier/revision is recorded in `Results/results.json`, under `json["Results/reproducibility/model_source.json"]`.
4. Use a separate authorised copy of the submission, keeping Code/ and Results/ alongside each other. Original protocol text and hashes are stored under `protocols` in Results/results.json; the relevant runners read those verified records when the old docs files are absent.

The research paths `data_set/`, `reports/` and `configs/` are relative to Code/. They are distinct from the submission's Data/ and Results/ folders. Settings and seeds for the BERT extension are in [Code/configs/bert_small_training.json](Code/configs/bert_small_training.json). The grouped split seed is **20261013**; BERT seeds are **20261013, 20261014 and 20261015**.

## 3. Run order

Complete each stage's prerequisites before starting the next. The sequence below identifies the existing project modules; it is not an automatic reconstruction of missing private review records or every historical experiment.

| Order | Command from Code/ | Output or purpose |
| --- | --- | --- |
| 1 | `.venv/Scripts/python.exe -m src.data.data_understanding` | Parse the raw source and export annotation/statistical evidence. |
| 2 | `.venv/Scripts/python.exe -m src.data.data_quality` | Produce annotation-audit and duplicate-content evidence. |
| 3 | `.venv/Scripts/python.exe -m src.data.prepare_nucle` | Prepare source-preserving sentence/span samples using the original review inputs. |
| 4 | `.venv/Scripts/python.exe -m src.data.group_split_nucle --seed 20261013` | Create the connected-group train/validation/test split. |
| 5 | `.venv/Scripts/python.exe -m src.features.nucle_pos` | Fit the train-only POS rule and save validation evidence. |
| 6 | `.venv/Scripts/python.exe -m src.features.nucle_features` | Extract S/L/P/M/D, fit the encoder on train and export vectors/targets. |
| 7 | `.venv/Scripts/python.exe -m src.training.nucle_benchmark` | Run the classical train/validation benchmark. |
| 8 | `.venv/Scripts/python.exe -m src.training.nucle_ablation` | Run the defined ablation/masking experiments. |
| 9 | `.venv/Scripts/python.exe -m src.training.nucle_final_selection` | Fit the final-stage configurations; the six-variant validation outputs from stage 8 are required. |
| 10 | `.venv/Scripts/python.exe -m src.training.nucle_final_selection --freeze` | Freeze selected states, inputs and source identities before test scoring. |
| 11 | `.venv/Scripts/python.exe -m src.evaluation.nucle_final_evaluation score` | Generate classical test predictions and metrics. |
| 12 | `.venv/Scripts/python.exe -m src.evaluation.nucle_final_evaluation bootstrap` | Generate paired connected-group bootstrap intervals. |
| 13 | `.venv/Scripts/python.exe -m src.evaluation.nucle_final_error_audit` | Generate source-aware error analysis from the recorded predictions. |

Preparation, selection and scoring runners protect existing outputs. Use a fresh workspace for a fresh execution; do not overwrite completed experiments or remove the test-scoring guard. Preparation has one source-preserving policy and no version-selection or comparison step.

The canonical output locations are `data_set/processed/nucle_preparation/`, `data_set/features/nucle_features/`, `reports/nucle_pos_baseline/`, `reports/nucle_feature_benchmark/` and `reports/nucle_feature_ablation/`. The BERT configuration uses the same prepared samples. Original evidence/protocol records keep their historical identifiers and hashes; the current path mapping is recorded under `current_pipeline` in Results/results.json. A historical lock is not made valid by renaming paths; use a fresh matching run and lock for the current source.

After restoring BERT inputs, run the extension in this order:

```powershell
.venv/Scripts/python.exe -m src.training.bert_small_experiment check --config configs/bert_small_training.json
.venv/Scripts/python.exe -m src.training.bert_small_experiment train --config configs/bert_small_training.json --output reports/bert_small_experiment
.venv/Scripts/python.exe -m src.training.bert_small_experiment select --experiment reports/bert_small_experiment
.venv/Scripts/python.exe -m src.training.bert_small_experiment test --selection reports/bert_small_experiment/selection.json --output reports/bert_small_experiment/test
.venv/Scripts/python.exe -m src.evaluation.bert_final_analysis --experiment reports/bert_small_experiment --output reports/bert_final_analysis
```

The BERT analysis also requires the classical prediction/bootstrap artifacts from the earlier stages. Selection uses the mean best-checkpoint validation Macro-F1 across the three declared seeds. The primary seed remains 20261013; test results must not be used to choose a new primary.

## 4. Script that reproduces a main result

Use the original project script [Code/src/evaluation/nucle_final_evaluation.py](Code/src/evaluation/nucle_final_evaluation.py). It scores the frozen classical predictors and produces the LR Full result reported in the final report.

After completing setup and stages 1-10 in a fresh authorised workspace, run from Code/:

```powershell
.venv/Scripts/python.exe -m src.evaluation.nucle_final_evaluation score
```

The main outputs are:

- `Code/reports/final_evaluation/test_metrics.json`: the LR Full Macro-F1 is stored at `all.models.LR_03.macro_f1`.
- `Code/reports/final_evaluation/test_predictions.csv`: row-aligned gold labels and predictions.
- `Code/reports/final_evaluation/test_scoring_complete.json`: sample/group counts and scoring provenance.

The saved reference is **LR Full test Macro-F1 0.8162577160**, accuracy **0.8357263514**, on **2,368 test annotations**. These are the recorded experimental results, not a claim that training was rerun while writing this README.

For an already completed experiment with the matching archival states, encoded test inputs, predictions and lock restored, verify checkpoint predictions instead of scoring the test again:

```powershell
.venv/Scripts/python.exe -m src.evaluation.nucle_final_evaluation replay
```

Replay reports `CHECKPOINT REPLAY EXACT` when the restored predictors reproduce the stored labels. Exact archival verification requires the original source bytes and inputs; the current protocol-location fallback does not make an old source lock valid automatically. Known historical source/CSV discrepancies are recorded in Results/results.json.

Without restricted inputs, inspect [Results/main_results.csv](Results/main_results.csv) and [Results/results.json](Results/results.json). The bundle's `json`, `tables` and `texts` sections retain the original evidence names as record keys. They include complete classical comparisons, BERT validation/seed results, confusion counts and audit records. Viewing saved results is not a new model replay. No additional reproduction helper or notebook is required or included.
