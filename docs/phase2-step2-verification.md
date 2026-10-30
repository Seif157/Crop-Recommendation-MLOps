# Phase 2, Step 2 Verification Record

## Scope and starting state

Phase 2, Step 2 implements DVC data and pipeline versioning only. Controlled
experiment comparison, model promotion, serving, containers, CI/CD, and
monitoring remain out of scope. Phase 2 is not complete; Step 3 is controlled
experiment comparison.

Verification was performed on 2026-10-29 on Windows 11 with Python 3.12.13,
DVC 3.67.1, MLflow 3.17.2, and Scikit-learn 1.7.2. The starting branch was
`main` at `a81a7d325604e3bd92af50612d78f300470d81a0`. The existing Phase 1 and
Phase 2, Step 1 work was uncommitted and preserved. The raw CSV was tracked by
Git at that revision, so it was migrated with the authorized index-only
`git rm --cached data/raw/Crop_recommendation.csv`; its working copy was never
removed. No unrelated path was staged and no commit was created.

Protected legacy checksums remained unchanged:

| File | SHA-256 |
| --- | --- |
| `legacy/train_model.py` | `ebed41784a76f21b2818dfef626cc667ff2a84b7c925ec302d4fce1a128754b1` |
| `legacy/app.py` | `a1566c875fd504ff5a10eae00c46b43a3df73241fd223e521f2cafb6799f83e6` |
| `legacy/requirements.txt` | `08171c8c25afa8614891da55ed9c166799c41cdf6c9258171cd1e8985ce4a6e4` |

## DVC architecture

Git records the pointer and pipeline definitions. DVC stores source and output
content in its content-addressed cache and the configured remote. MLflow remains
an explicit, separate experiment-tracking workflow.

```text
data/raw/Crop_recommendation.csv.dvc
                 |
                 v
             validate
                 |
                 v
               train
                 |
                 v
              evaluate
```

- Pointer: `data/raw/Crop_recommendation.csv.dvc`.
- Workspace data: ignored `data/raw/Crop_recommendation.csv` at the original
  package path.
- Cache: `.dvc/cache/`.
- Local remote: ignored `.dvc-remote/`, separate from the cache.
- Local-only configuration: ignored `.dvc/config.local` with default remote
  `local-storage`; no absolute path or credential is tracked.
- Configuration: `params.yaml` contains only the established split and Random
  Forest settings.
- Graph: `dvc.yaml` declares actual code, data, parameter, upstream-output, and
  output relationships for `validate`, `train`, and `evaluate`.
- State lock: DVC-generated `dvc.lock` records commands, dependency hashes,
  parameters, and output hashes. `uv.lock` instead locks Python packages.

The validation stage calls the existing data contract and emits deterministic
JSON. The train stage calls the established split and training APIs and receives
only `X_train` and `y_train`. It writes a Skops pipeline plus training metadata.
The evaluation stage reconstructs the same split, checks dataset, partition,
configuration, model parameters, and feature schema, then calls prediction-only
evaluation. It never calls `fit`.

The pipeline-only `artifacts/dvc/model.skops` is an intermediate DVC output. Its
loader permits only `sklearn.tree._tree.Tree` and verifies the fitted ordered
feature schema. It is not the MLflow logged model or a promoted production model.
Serialized files from untrusted sources must not be loaded.

## Dataset and storage results

| Property | Verified value |
| --- | --- |
| Workspace path | `data/raw/Crop_recommendation.csv` |
| Size | 150,034 bytes |
| SHA-256 before and after migration | `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e` |
| DVC MD5 | `5e743dd4cf49f84d23a904cd51c9b5b8` |
| Remote push | 6 files pushed |
| Isolated pull | 1 file fetched and added |
| Pulled SHA-256 | exact match to the source fingerprint |

The pull proof used a new ignored workspace under `.test-tmp/`, initialized as
an independent no-SCM DVC repository. It pulled the pointer from `.dvc-remote/`
without deleting the primary workspace data or clearing the primary cache. This
project-local remote does not provide cross-machine sharing by itself.

The first attempted isolated command combined `dvc init --no-scm` with
`--subdir` and passed a nonexistent project-directory option to later commands;
DVC 3.67.1 rejected those arguments before retrieval. The verification was
corrected by executing each DVC command with the isolated directory as its
working directory. No source data was affected.

## Pipeline and reproducibility results

The first `dvc repro` ran all three stages and generated `dvc.lock`. An unchanged
second run reported every stage as skipped and `Data and pipelines are up to
date.` `dvc status` reported the same clean pipeline state.

| Result | Phase 1 baseline | DVC pipeline |
| --- | ---: | ---: |
| Source rows | 2,200 | 2,200 |
| Train rows | 1,760 | 1,760 |
| Test rows | 440 | 440 |
| Accuracy | 0.995454545455 | 0.995454545455 |
| Macro precision | 0.995670995671 | 0.995670995671 |
| Macro recall | 0.995454545455 | 0.995454545455 |
| Macro F1 | 0.995451702769 | 0.995451702769 |

Repeated runs preserved the source fingerprint, split parameters and stable
index fingerprints, model configuration, training metadata, predictions,
prediction fingerprint, evaluation metrics, confusion matrix, and JSON bytes.
Binary model bytes are not claimed to be portable or deterministic across
environments. No test metric was used to select parameters.

## MLflow compatibility

`uv run track-crop-baseline --json` completed after the DVC migration. Run
`2557d2e56149426b994720c132fec006` logged model
`m-da8750f242124542a7221b4d56bc8fb5`, the unchanged dataset SHA-256, the same
five headline metrics, and a verified model/signature round trip. The local
server started against `.mlflow/mlflow.db` and `.mlflow/artifacts/`; `/health`
returned HTTP 200 with `OK`, after which the server was stopped normally.

DVC stages do not import or call the MLflow logging workflow and create no
implicit runs. A future explicit MLflow run can record the DVC content hash or a
committed Git/DVC revision as lineage without coupling deterministic pipeline
caching to side-effecting experiment creation.

## Verification commands and results

```bash
uv sync --cache-dir .uv-cache
uv lock --check --cache-dir .uv-cache
uv run dvc status
uv run dvc dag
uv run dvc repro
uv run dvc repro
uv run dvc push
uv run track-crop-baseline --json
uv run serve-crop-mlflow --port 5051
uv run python -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/phase2-step2-full
uv run python -m pytest -q -m integration -p no:cacheprovider --basetemp=.test-tmp/phase2-step2-integration
```

Results:

- dependency lock resolved 159 packages;
- 18 focused DVC tests passed;
- all 143 repository tests passed;
- the exact previous 125-test selection passed;
- all 9 integration tests passed; and
- the dataset, legacy files, Scikit-learn version, baseline behavior, and MLflow
  storage remained intact.
