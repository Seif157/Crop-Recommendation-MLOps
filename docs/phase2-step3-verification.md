# Phase 2, Step 3 Verification Record

## Scope and starting state

Phase 2, Step 3 implements controlled Random Forest comparison with
training-only stratified cross-validation and explicit MLflow logging. It does
not register, promote, retrain for production, serve, or deploy a model. Phase 2
is not complete; Step 4 is model registry and controlled promotion.

Verification completed on 2026-10-30 with Python 3.12.13, Scikit-learn 1.7.2,
MLflow 3.17.2, and DVC 3.67.1. The branch remained `main` at
`a81a7d325604e3bd92af50612d78f300470d81a0`. Earlier uncommitted work was
preserved, including the intentionally staged index-only removal of the original
CSV. No unrelated file was staged and no commit or push was made.

The source dataset retained SHA-256
`54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e`
and DVC MD5 `5e743dd4cf49f84d23a904cd51c9b5b8`. All protected legacy
checksums remained unchanged.

## Predetermined experiment design

The candidate list and selection policy were fixed before scores were observed:

| Candidate | `n_estimators` | Model seed |
| --- | ---: | ---: |
| RF-50 | 50 | 42 |
| RF-100 | 100 | 42 |
| RF-200 | 200 | 42 |

The comparison uses
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on only the
established 1,760-row outer training partition. Every fold fits on 1,408 rows and
validates on 352 rows. Every crop contributes 64 fit rows and 16 validation rows
per fold. Validation folds are disjoint and cover every outer-training row
exactly once.

Scoring records accuracy, macro precision, macro recall, and macro F1. Aggregate
dispersion is the population standard deviation. Ranking uses highest mean macro
F1. Scores within `1e-12` are treated as an effective tie and resolved in favor
of fewer trees. The historical outer-test score is not part of this policy.

## Measured fold results

The following values are measured training-only CV results, not outer-test
metrics:

| Candidate | Fold | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| RF-50 | 1 | 1.000000000000 | 1.000000000000 |
| RF-50 | 2 | 1.000000000000 | 1.000000000000 |
| RF-50 | 3 | 0.977272727273 | 0.976841687187 |
| RF-50 | 4 | 0.991477272727 | 0.991452214050 |
| RF-50 | 5 | 0.997159090909 | 0.997156313872 |
| RF-100 | 1 | 1.000000000000 | 1.000000000000 |
| RF-100 | 2 | 0.997159090909 | 0.997156313872 |
| RF-100 | 3 | 0.982954545455 | 0.982751888634 |
| RF-100 | 4 | 0.991477272727 | 0.991452214050 |
| RF-100 | 5 | 0.997159090909 | 0.997156313872 |
| RF-200 | 1 | 1.000000000000 | 1.000000000000 |
| RF-200 | 2 | 0.997159090909 | 0.997156313872 |
| RF-200 | 3 | 0.988636363636 | 0.988591800357 |
| RF-200 | 4 | 0.988636363636 | 0.988558015618 |
| RF-200 | 5 | 0.997159090909 | 0.997156313872 |

| Candidate | Mean accuracy | Mean macro precision | Mean macro recall | Mean macro F1 | Macro F1 population std. dev. |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF-50 | 0.993181818182 | 0.994320471410 | 0.993181818182 | 0.993090043022 | 0.008703115771 |
| RF-100 | 0.993750000000 | 0.994557338087 | 0.993750000000 | 0.993703346086 | 0.006139964810 |
| RF-200 | 0.994318181818 | 0.994940113206 | 0.994318181818 | 0.994292488744 | 0.004782482694 |

RF-200 has the best observed mean CV macro F1 and is therefore recommended by
the predetermined policy. The observed difference is small and does not prove
statistical superiority. The recommendation is configuration evidence, not
production approval.

## MLflow evidence

The separate experiment is `crop-recommendation-cv`, experiment ID `2`. The
first verified execution produced:

| Candidate | Run ID |
| --- | --- |
| RF-50 | `1621a3cfb75d456a9d852b89406a117b` |
| RF-100 | `3415a80942494bc4a6a06117046deefd` |
| RF-200 | `d55774ff7d6f4f9590c50ae99123f51b` |

A repeated execution produced the identical stable report and new run IDs:

| Candidate | Repeated run ID |
| --- | --- |
| RF-50 | `c03e4abddb944a949a4fd4bef6a8c54b` |
| RF-100 | `298b54efd72b416bafc48330d769304f` |
| RF-200 | `e29cf147c20841b78b46c546553b2caa` |

Every run contains candidate and CV parameters, means, population standard
deviations, all 20 fold-level metric values, source SHA-256, DVC MD5, the
training-partition fingerprint, Git/software metadata, and
`outer_holdout_used=false`. Each run stores candidate, fold, CV configuration,
and compact comparison JSON artifacts. No fitted model or registered model
version is created. After both executions, experiment ID `2` contained exactly
six runs—two for each candidate—and the MLflow backend contained zero registered
models.

The existing baseline command was also rerun successfully after implementation.
Baseline run `0192d7a4e1f84078b5a3d38e1e55e5f5` logged and round-tripped model
`m-e880c492837d449f8c04be7508a9b4c0` with the unchanged dataset fingerprint and
fixed-holdout metrics.

## Holdout isolation and DVC preservation

The core API accepts only `X_train` and `y_train`; it has no outer-test
parameters. The command loads the established outer split, extracts only those
two training objects, discards the split object, and passes the training inputs
to comparison. Tests inspect both API and orchestration boundaries. The 440-row
outer test partition is never evaluated during this step.

`params.yaml`, `dvc.yaml`, and baseline output metrics are unchanged. Registering
the `compare-crop-models` entry point changed `pyproject.toml`, which is a
declared baseline-stage dependency. The existing DVC integration gate therefore
reproduced the baseline and refreshed the corresponding dependency/output hashes
in `dvc.lock`; no comparison result or candidate configuration was added to DVC.
The Skops binary hash changed on reserialization, which is not treated as
cross-run binary determinism, while the training metadata, predictions,
evaluation report, and metrics remained identical. `dvc status` and a subsequent
unchanged `dvc repro` are up to date. Candidate runs record DVC data identity
without coupling MLflow side effects to DVC stage execution.

## Verification

```bash
uv sync --cache-dir .uv-cache
uv lock --check --cache-dir .uv-cache
uv run compare-crop-models --json
uv run compare-crop-models
uv run python -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/phase2-step3
uv run python -m pytest -q -m integration -p no:cacheprovider --basetemp=.test-tmp/phase2-step3-integration
uv run dvc status
uv run dvc repro
```

Results:

- 26 focused comparison tests passed with the real-data test deselected;
- the single focused real-data comparison integration test passed;
- all 170 repository tests passed;
- all previously reported 143 tests remained passing;
- the integration selection passed 10 tests;
- repeated stable comparison scores matched exactly; and
- one upstream MLflow SQLAlchemy deprecation warning was observed, with no test
  or persistence failure.
