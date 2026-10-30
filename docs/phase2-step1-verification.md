# Phase 2, Step 1 Verification Record

## Scope and status

MLflow experiment tracking is implemented for the established Random Forest
baseline. Phase 2 is not complete: DVC versioning, training-only candidate
comparison, and model promotion remain future work.

Verification was performed on 2026-10-27 with Python 3.12.13, MLflow 3.17.2,
Pandas 2.3.3, and Scikit-learn 1.7.2 on Windows 11. The Git worktree was dirty
because the Phase 1 and Phase 2 changes had not been committed; the recorded base
revision was `a81a7d325604e3bd92af50612d78f300470d81a0`.

## Storage architecture

- Backend store: repository-local SQLite at `.mlflow/mlflow.db`.
- Artifact store: repository-local filesystem at `.mlflow/artifacts/`.
- Tracking/UI server: `127.0.0.1:5000` by default.
- Experiment: `crop-recommendation-baseline`.
- Evaluation protocol tag: `fixed_holdout`.

Both storage paths are ignored by Git. The tracking command resolves absolute
paths from the installed module, so it does not depend on the caller's current
directory. The server uses the same backend and artifact locations, and the
experiment remained accessible after the logging process exited and when the
server started later.

## Verified real run

| Identity | Value |
| --- | --- |
| Experiment ID | `1` |
| Run ID | `41c1dcda8f224550863d6af0f7006577` |
| Logged model ID | `m-5a43796604cd4597a549254dab7a255a` |
| Model URI | `models:/m-5a43796604cd4597a549254dab7a255a` |
| Dataset SHA-256 | `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e` |

The run stored the actual Random Forest and split parameters, all explicit
classifier settings, dataset dimensions, ordered feature names, class count,
sample counts, Python/MLflow/Scikit-learn versions, Git state, and the tracking
identities. The JSON report contains the complete evaluation, per-class metrics,
and 22 by 22 confusion matrix.

| Metric | Stored value |
| --- | ---: |
| Accuracy | 0.995454545455 |
| Macro precision | 0.995670995671 |
| Macro recall | 0.995454545455 |
| Macro F1 | 0.995451702769 |
| Test samples | 440 |

The logged model signature contains the ordered inputs `N`, `P`, `K`,
`temperature`, `humidity`, `ph`, and `rainfall`. The example contains five rows
from the training partition. The model was loaded through its MLflow model URI and
reproduced the original predictions on that example.

## Verification commands

```bash
uv sync --cache-dir .uv-cache
uv lock --check --cache-dir .uv-cache
uv run track-crop-baseline --json
uv run serve-crop-mlflow --port 5050
uv run python -m pytest tests/test_mlflow_tracking.py -q -p no:cacheprovider --basetemp=.test-tmp/mlflow-focused
uv run python -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/phase2-step1
uv run python -m pytest -q -m integration -p no:cacheprovider --basetemp=.test-tmp/phase2-step1-integration
```

The localhost server returned HTTP 200 from `/health`, and its REST API retrieved
experiment ID `1` from the persistent SQLite backend. The server was then stopped
normally with `Ctrl+C`.

Verification results:

- dependency lock resolved 96 packages without changing Scikit-learn 1.7.2;
- 13 focused MLflow tracking tests passed;
- all 125 repository tests passed;
- all 8 integration tests passed; and
- the original 112 Phase 1 tests passed as a separate regression selection.

## Security and platform findings

MLflow 3.17's Scikit-learn flavor uses a security-aware serializer. Random Forest
requires the internal `sklearn.tree._tree.Tree` type, so the integration trusts
only that exact type while logging and round-trip loading the locally produced
model. Model files from untrusted sources must never be loaded.

MLflow reports that its optional job-execution backend is unsupported on Windows.
That does not affect the tracking server, UI, SQLite backend, report artifacts,
model logging, or model loading used in this step. The local server has no
authentication and is deliberately bound to `127.0.0.1`; it must not be exposed as
a production service.

MLflow successfully exported 90 locked project requirements through `uv`. Because
this `uv` environment does not install `pip`, MLflow warned that its generated
Conda environment includes `pip` without a version. The native MLflow model was
still loaded and verified in the locked project environment.
