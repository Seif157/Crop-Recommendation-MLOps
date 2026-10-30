# Phase 1 Verification Record

## Status

Phase 1 is complete for the repository's stated ML-foundation scope: validate the
raw data, create a deterministic leakage-aware split, fit the baseline on training
data only, evaluate on the untouched test partition, and prove that the workflow
is reproducible in the locked environment.

Verification was performed on 2026-10-27. No model artifact was serialized, no
tracking service was introduced, and no legacy file or raw-data file was changed.

## Verified environment

| Item | Verified value |
| --- | --- |
| Operating system | Windows 11, AMD64 |
| Python | 3.12.13 |
| uv | 0.11.13 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| Dataset SHA-256 | `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e` |
| Split | stratified 80/20 holdout, `random_state=42` |
| Baseline | `RandomForestClassifier(n_estimators=100, random_state=42)` |

`uv.lock` resolved 18 packages during verification. The Git base revision was
`a81a7d3`; the Phase 1 changes were still in the working tree, so that commit alone
does not identify the verified source state.

## Commands and results

Run these commands from the repository root in Git Bash:

```bash
uv sync
uv lock --check --cache-dir .uv-cache
uv run validate-crop-data
uv run python -m pytest tests/test_end_to_end.py -q -p no:cacheprovider --basetemp=.test-tmp/phase1-focused
uv run python -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/phase1-full
uv run python -m pytest -q -m integration -p no:cacheprovider --basetemp=.test-tmp/phase1-integration
uv run python -m crop_recommendation.training.evaluate_command --json
```

The focused end-to-end gate passed. The complete suite passed with 112 tests, and
the integration selection passed with 7 tests. Dataset validation reported 2,200
rows, 7 features, 22 classes, no missing or infinite values, no duplicate rows,
and no validation errors or warnings.

## Reproducibility and integrity evidence

The complete validation, split, training, prediction, and evaluation workflow was
executed independently twice. The runs produced:

- identical train and test row indices and values;
- identical class distributions and class order;
- identical model metadata and predictions;
- identical scalar metrics, per-class metrics, and 22 by 22 confusion matrices;
- 1,760 training rows and 440 test rows, with disjoint source indices whose union
  contains all 2,200 source rows; and
- 80 training rows and 20 test rows for each class.

The end-to-end test independently recomputes accuracy, macro precision, macro
recall, macro F1, and the confusion matrix from the fixed predictions. It also
checks that confusion counts and per-class support both sum to the 440 evaluated
samples. Component tests separately prove that evaluation never calls `fit`, the
training API accepts no test partition, and training/evaluation do not mutate their
inputs.

Two independent invocations of the JSON evaluation command were also compared
byte for byte and were identical in this environment. Cross-version byte-for-byte
reproducibility is not claimed.

## Verified baseline result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.995454545455 |
| Macro precision | 0.995670995671 |
| Macro recall | 0.995454545455 |
| Macro F1 | 0.995451702769 |

The confusion matrix contains 438 correct predictions. One `blackgram` sample was
predicted as `maize`, and one `rice` sample was predicted as `jute`. The evaluation
emitted no undefined-metric warnings.

These figures describe only the fixed held-out partition. They are not model
selection evidence, external validation, or proof of agronomic suitability.

## Failure behavior and remaining limits

Tests cover missing, unexpected, duplicate, and reordered columns; non-numeric,
missing, infinite, and out-of-domain values; unknown or blank labels; infeasible
stratified splits; feature/target index mismatches; target leakage; invalid model
configuration; unfitted estimators; and prediction/class contract failures. These
conditions fail explicitly rather than silently changing the evaluation protocol.

The repository still has no model-selection workflow, serialized production
artifact, experiment tracker, dedicated data/model versioning, API, UI migration,
CI/CD, containers, monitoring, drift detection, or external validation dataset.
Those are later roadmap concerns and are not implied by Phase 1 completion.
