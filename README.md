# ML Core — reusable PyTorch training foundation

This is a complete, local supervised learning project. A small neural network learns to classify points from two noisy concentric rings. It learns its weights from labeled examples using backpropagation and Adam. There are no external model APIs, credentials, dataset downloads, or services.

** Install and run

Use Python 3.11 or 3.12. Python 3.12 is the recommended version for this project. The project uses uv to manage the Python version, virtual environment, dependencies, and lock file. A CPU is sufficient.

1. Install uv

If uv is not installed, follow the official uv installation instructions. Then verify:

uv --version

2. Pin Python 3.12

From the project root:

uv python install 3.12
uv python pin 3.12

This creates/updates .python-version. The project intentionally uses Python <3.13, so do not pin Python 3.13.

3. Synchronize dependencies

uv sync

uv reads pyproject.toml, resolves the dependencies, creates or updates .venv, and writes uv.lock.

If .venv was previously created with an incompatible Python version, remove it and run uv sync again.

4. Verify the environment

uv run python --version
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import numpy; print(numpy.__version__)"

5. Run the project

uv run python -m src.training.train

uv run python -m src.evaluation.evaluate

uv run python -m src.inference.predict --features 0.7 0.0

uv run pytest

Run commands from the project root. uv run automatically uses the project environment, so manually activating .venv is normally unnecessary.

If you prefer to activate it manually, use .venv\Scripts\activate on Windows Command Prompt/PowerShell or source .venv/bin/activate in Git Bash/Linux/macOS.

**Intel Mac compatibility:** official macOS x86 wheels stop at PyTorch 2.2.x ([PyTorch announcement](https://pytorch.org/blog/pytorch2-2/)). `requirements.txt` selects 2.2.2 and NumPy 1.26.4 on that platform. Use Python 3.9–3.12, rather than 3.13/3.14; for example, `/usr/bin/python3 -m venv .venv` if that interpreter is 3.9. This legacy version is for this local demo; load only checkpoints you created. Other platforms use PyTorch 2.6 or later. Package installation needs internet access; training and inference thereafter work offline.

## Architecture and folders

```text
ml-core/
├── data/
│   ├── raw/                     # Generated original rings.csv
│   ├── processed/               # Training normalization metadata
│   └── split/
│       ├── train/               # data.pt: normalized features, labels, row IDs
│       ├── validation/          # data.pt
│       └── test/                # data.pt
├── src/
│   ├── __init__.py
│   ├── config.py                # YAML and project-relative paths
│   ├── checkpoint.py            # Shared architecture/weight restoration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # CSV, cleaning, splitting, normalization
│   │   └── dataset.py           # Dataset and DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py             # Trainable MLP
│   │   └── loss.py              # CrossEntropyLoss
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py             # Adam, backpropagation, validation, saving
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Read-only metric calculations
│   │   └── evaluate.py          # Final test evaluation
│   └── inference/
│       ├── __init__.py
│       └── predict.py          # Raw point -> class and probabilities
├── configs/config.yaml
├── notebooks/exploration.ipynb
├── tests/                       # Dataset, gradients, metrics, checkpoint tests
├── checkpoints/.gitkeep
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
├── .python-version                # Pinned Python version
├── pytest.ini
├── requirements.txt               # Legacy dependency list during uv migration
├── README.md
└── .gitignore
```

The data path is CSV → preprocessing → Dataset → DataLoader → model → loss → backpropagation → optimizer → learned weights → checkpoint. Evaluation and inference restore those learned weights independently. The model module knows nothing about files; training receives batches through DataLoaders.

## Dataset

The first training run creates `data/raw/rings.csv` deterministically with seed 42. This is a synthetic educational dataset, not a measurement of a real-world task.

| Property | Default |
|---|---|
| Input | Two floating-point Cartesian coordinates, `x1` and `x2` |
| Target | Integer class: 0 = `inner_ring`, 1 = `outer_ring` |
| Classes | 2 |
| Size | 600 rows, 300 per class |
| Inner / outer radius | 0.7 / 1.7 |
| Noise | Independent Gaussian noise, standard deviation 0.12 per coordinate |
| Training | 70%, 420 rows, 210 per class |
| Validation | 15%, 90 rows, 45 per class |
| Test | Remaining 15%, 90 rows, 45 per class |

Angles are sampled uniformly around each ring; Gaussian coordinate noise is then added. The neural network sees the coordinates, not the radius or generation rule. This is a nonlinear classification problem. Small synthetic test sets can be very easy; high accuracy does not establish real-world performance.

The CSV is preserved if it already exists. To change sample count, radii, or noise, choose a new `paths.raw` in the config (or deliberately remove the generated CSV). A different split seed regenerates the partitions during training. Use separate split and checkpoint paths for experiments you want to retain. For custom CSVs use the same header and two numeric features with labels 0 and 1; split sizes then follow the actual CSV, not `dataset.samples`.

## Preprocessing and data loaders

1. Generate the demo CSV if absent, then read it in `preprocessing.py`.
2. Reject malformed columns, NaN/infinity, invalid labels, and duplicate feature rows. Invalid rows fail clearly rather than being silently imputed or dropped.
3. Shuffle row indices within each class using a seeded NumPy generator. Allocate the train and validation counts with integer flooring; the rest goes to test. Each split must contain both classes.
4. Fit each coordinate's mean and population standard deviation on **training rows only**. Replace a zero standard deviation with 1.
5. Apply `(x - training_mean) / training_std` to every partition. Convert features to `torch.float32` and class labels to `torch.long`.
6. Save normalization metadata and one tensor file per split, including original row IDs so partition isolation is inspectable.

`PointDataset` loads only prepared tensors. `make_loader` creates minibatches using the configured batch size (32 by default), shuffles training examples, and leaves validation and test order stable. Training never parses the CSV or transforms raw rows itself. The fixed seeds reproduce data, model initialization, and batch ordering on the same software/device; bitwise results across versions or hardware are not promised.

## Model and loss

The MLP is `Linear(2,16) → ReLU → Linear(16,16) → ReLU → Linear(16,2)`. It has 354 trainable weights and biases. For an input tensor of shape `[batch_size, 2]`, it returns `[batch_size, 2]` **logits** (unbounded class scores).

`loss.py` defines `CrossEntropyLoss`, which compares logits with integer labels. Do not apply softmax before this loss; it already incorporates the appropriate log-probability calculation. Softmax is applied only when inference reports probabilities.

## Training, validation, and test are different

| Phase | Data | Purpose | Updates weights? |
|---|---|---|---|
| Training | Training split | Minimize cross entropy using examples and labels | Yes |
| Validation | Validation split after each epoch | Select the epoch with lowest validation loss | No |
| Test evaluation | Held-out test split after selection | Estimate performance of the selected model | No |

Validation uses `model.eval()` and `torch.no_grad()`. It never calls an optimizer. The default network has no dropout or batch normalization, but setting evaluation mode is still the correct inference protocol. `no_grad()` separately disables gradient recording.

The test labels never contribute to gradients or checkpoint selection. Run final evaluation after finishing model choices; repeatedly tuning based on test results compromises the held-out estimate.

## What happens when training starts?

`python -m src.training.train` executes `main()` in `src/training/train.py`:

1. Parse `--config` and load `configs/config.yaml`; paths inside YAML resolve relative to the project root.
2. Seed PyTorch and prepare data via the separate preprocessing module.
3. Build training and validation DataLoaders; record held-out row IDs for later consistency checks.
4. Construct the MLP, cross entropy loss, and Adam optimizer.
5. For each of 30 epochs, call `model.train()` and process every training minibatch.
6. Calculate validation metrics without gradients; log sample-weighted loss and accuracy.
7. Whenever validation loss improves, save `checkpoints/best.pt`.
8. Write all epoch metrics to `checkpoints/best.history.json`. Final test evaluation is an explicit separate command, so it is not part of model selection.

### Exactly where weights change

Inside the minibatch loop in `src/training/train.py`:

```python
optimizer.zero_grad()      # Clear old gradients, not parameters.
logits = model(inputs)     # Use current weights in the forward pass.
loss = criterion(logits, targets)
loss.backward()            # Compute d(loss)/d(parameter) with backpropagation.
optimizer.step()           # THIS line updates the model weights and biases.
```

`loss.backward()` populates each parameter's `.grad`; it does not update the parameter itself. `optimizer.step()` uses those gradients and Adam's running first and second moment estimates, the configured learning rate, and weight decay to update parameters in place. The familiar `weight -= learning_rate * gradient` describes basic SGD; Adam adapts this step per parameter. Repeating these steps is the actual learning process. The tests verify that parameters change after a step.

Training loss and accuracy aggregate the predictions made during the epoch while weights evolve. Validation metrics measure the fixed weights at that epoch's end.

## Checkpoints

`best.pt` is generated by real training; no fabricated pretrained weights are distributed. Generated weights and datasets are ignored by Git. Lower validation loss wins; an exact tie keeps the earlier checkpoint. A new run overwrites the configured checkpoint and history, so change paths to preserve runs.

The checkpoint contains:

- `model_state_dict`: learned tensors for every layer.
- `optimizer_state_dict`: Adam state, useful for a future resume implementation.
- `epoch`, `validation_metric` (loss), its name, and full validation metrics.
- `configuration`: the actual model, training, dataset, and path settings.
- `preprocessing`: training mean/std, class names, seed, and raw CSV SHA-256.
- `test_ids`: the held-out row indices used for this run.

`src/checkpoint.py` reads with `map_location="cpu"` by default and explicit `weights_only=True`, rebuilds the architecture from the saved config, loads learned tensors, and sets evaluation mode. It does not create an optimizer or train. Inference is self-contained: it needs only the checkpoint, not raw or prepared data. The saved optimizer is not an automatic resume feature; the training command intentionally starts a new run.

## Final test evaluation

```bash
python -m src.evaluation.evaluate
python -m src.evaluation.evaluate --checkpoint checkpoints/best.pt
```

This loads the **best saved model**, not the final epoch's in-memory model, and only opens the test partition. It returns cross entropy loss, accuracy, macro precision, macro recall, macro F1, sample count, and a confusion matrix (rows are actual classes; columns are predicted classes). Macro averaging gives each class equal weight. Undefined per-class precision/recall is treated as zero; loss is weighted by sample count, including the last short batch.

Evaluation checks normalization metadata and held-out row IDs against the checkpoint, rejecting mismatched prepared splits. It does not regenerate data. If you move the project after training, pass `--split-dir data/split` to override the original saved absolute split path. `--config` chooses the default checkpoint path; saved model settings remain authoritative.

## Inference

```bash
python -m src.inference.predict
python -m src.inference.predict --features 1.7 0.0
python -m src.inference.predict --checkpoint checkpoints/best.pt --features -0.5 0.4
```

Pass **raw, unnormalized** coordinates in the same units as the CSV. `predict.py` restores the architecture and weights, applies checkpoint normalization, calls `model.eval()`, runs the forward pass inside `torch.no_grad()`, and returns class ID, class name, winning probability, and both class probabilities (class 0 then class 1). It never retrains. Softmax confidence is a model score, not a calibrated guarantee, especially far from the training distribution.

## Configuration

Edit `configs/config.yaml` to change sample generation, split fractions, hidden width, epochs, batch size, learning rate, weight decay, device, thread count, worker count, or paths. The two input dimensions and two output classes match this dataset. CPU and one worker process (`num_workers: 0`) are portable defaults. `device: cuda` requires a CUDA-enabled PyTorch installation and compatible hardware.

```bash
python -m src.training.train --config configs/config.yaml
python -m src.evaluation.evaluate --config configs/config.yaml
python -m src.inference.predict --config configs/config.yaml --features 0.7 0.0
```

## Automated tests and exploration

```bash
pytest
```

Tests use temporary dataset/checkpoint paths and a short three-epoch training run. They verify dataset lengths/dtypes, stratification, disjoint splits, repeatability, training-only normalization, input validation, forward shapes, finite loss/gradients, actual parameter updates, known metric values, validation leaving weights unchanged, best-epoch selection, checkpoint restoration, inference probabilities, final test evaluation, and rejection of mismatched test partitions. They do not depend on a preexisting checkpoint or assert an artificially guaranteed accuracy target.

The optional `notebooks/exploration.ipynb` loads training data for basic statistics and a scatter plot. Install `jupyterlab` and `matplotlib` if you want to run it; these are not required to train or test. It does not use the test set for development.

## PyTorch references

The implementation follows the documented [optimizer training loop](https://github.com/pytorch/pytorch/blob/main/docs/source/optim.md) and [module state dictionaries](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/modules.md). The central relationship is:

```text
DATA + MODEL + TRAINING CONFIGURATION
                 ↓
     FORWARD → LOSS → BACKPROPAGATION → OPTIMIZER
                 ↓
         LEARNED PARAMETERS
                 ↓
          SAVED TRAINED MODEL
```

## Verified run in this workspace

The prepared environment here is `.venv-intel` (Python 3.9.6, PyTorch 2.2.2, NumPy 1.26.4). To reuse it:

```bash
source .venv-intel/bin/activate
python -m src.training.train
python -m src.evaluation.evaluate
python -m src.inference.predict
pytest
```

A real 30-epoch CPU run completed. Its held-out test loss was 0.00145754; accuracy, macro precision, macro recall, and macro F1 were all 1.0 on 90 examples. The confusion matrix was `[[45, 0], [0, 45]]`. The input `[0.7, 0.0]` produced `inner_ring` with probability 0.99990034. All 7 tests passed. These are observed demo results, not promised scores for other data/configurations. The generated checkpoint and training history are present locally and excluded from version control.
