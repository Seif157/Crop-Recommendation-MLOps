"""Explicit MLflow tracking for the established fixed-holdout baseline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from crop_recommendation.data.preprocessing import DataSplit, SplitConfig
from crop_recommendation.training.evaluate import EvaluationResult
from crop_recommendation.training.train import TrainingConfig, TrainingResult

DEFAULT_EXPERIMENT_NAME = "crop-recommendation-baseline"
DEFAULT_RUN_NAME = "random-forest-fixed-holdout"
MODEL_ARTIFACT_NAME = "baseline-pipeline"
EVALUATION_PROTOCOL = "fixed_holdout"
REPORT_ARTIFACT_PATH = "reports/evaluation.json"
SKOPS_TRUSTED_TYPES = ("sklearn.tree._tree.Tree",)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TrackingError(RuntimeError):
    """Raised when tracking inputs or the configured MLflow backend fail."""


def repository_root() -> Path:
    """Return the project root without depending on the caller's directory."""

    return Path(__file__).resolve().parents[3]


def default_storage_directory() -> Path:
    """Return the ignored directory used by local MLflow storage."""

    return repository_root() / ".mlflow"


def default_backend_path() -> Path:
    """Return the persistent local SQLite backend path."""

    return default_storage_directory() / "mlflow.db"


def default_artifact_root() -> Path:
    """Return the persistent local artifact root."""

    return default_storage_directory() / "artifacts"


def sqlite_uri(database_path: str | Path) -> str:
    """Build a cross-platform absolute SQLAlchemy SQLite URI."""

    path = Path(database_path).expanduser().resolve()
    return f"sqlite:///{path.as_posix()}"


def default_tracking_uri() -> str:
    """Return the repository-local SQLite tracking URI."""

    return sqlite_uri(default_backend_path())


def file_sha256(path: str | Path) -> str:
    """Calculate a file fingerprint without changing the source file."""

    resolved_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with resolved_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TrackingError(
            f"Could not fingerprint dataset '{resolved_path}': {error}"
        ) from error
    return digest.hexdigest()


@dataclass(frozen=True)
class MlflowTrackingConfig:
    """Local-first MLflow backend, artifact, experiment, and run settings."""

    tracking_uri: str = field(default_factory=default_tracking_uri)
    artifact_root: Path = field(default_factory=default_artifact_root)
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str = DEFAULT_RUN_NAME
    input_example_rows: int = 5

    def __post_init__(self) -> None:
        tracking_uri = self.tracking_uri.strip()
        experiment_name = self.experiment_name.strip()
        run_name = self.run_name.strip()
        if not tracking_uri:
            raise ValueError("tracking_uri must not be empty.")
        if not experiment_name:
            raise ValueError("experiment_name must not be empty.")
        if not run_name:
            raise ValueError("run_name must not be empty.")
        if isinstance(self.input_example_rows, bool) or not isinstance(
            self.input_example_rows, int
        ):
            raise TypeError("input_example_rows must be an integer.")
        if self.input_example_rows <= 0:
            raise ValueError("input_example_rows must be greater than 0.")

        object.__setattr__(self, "tracking_uri", tracking_uri)
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())
        object.__setattr__(self, "experiment_name", experiment_name)
        object.__setattr__(self, "run_name", run_name)

    @property
    def artifact_root_uri(self) -> str:
        """Return the local artifact root as an absolute file URI."""

        return self.artifact_root.as_uri()


@dataclass(frozen=True)
class TrackingRunResult:
    """Stable identities returned after a verified MLflow logging run."""

    experiment_name: str
    experiment_id: str
    run_id: str
    tracking_uri: str
    run_artifact_uri: str
    model_name: str
    model_uri: str
    model_id: str | None
    dataset_sha256: str

    def to_dict(self) -> dict[str, str | None]:
        """Return command-friendly, JSON-compatible tracking identities."""

        return asdict(self)


def _git_metadata() -> tuple[str, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unavailable", None
    return revision or "unavailable", bool(status.strip())


def _validate_tracking_inputs(
    training: TrainingResult,
    evaluation: EvaluationResult,
    split: DataSplit,
    dataset_path: Path,
    config: MlflowTrackingConfig,
) -> None:
    if not dataset_path.is_file():
        raise TrackingError(f"Dataset does not exist: {dataset_path}")
    if not isinstance(training.estimator, Pipeline):
        raise TrackingError("The tracked estimator must be a Scikit-learn Pipeline.")
    try:
        check_is_fitted(training.estimator)
    except (TypeError, ValueError) as error:
        raise TrackingError("The estimator must be fitted before tracking.") from error
    if len(split.X_train) < config.input_example_rows:
        raise TrackingError(
            "The training partition is smaller than input_example_rows."
        )
    if training.metadata.training_rows != split.train_rows:
        raise TrackingError("Training metadata does not match the training split.")
    if evaluation.test_sample_count != split.test_rows:
        raise TrackingError("Evaluation metadata does not match the test split.")
    if training.metadata.feature_names != split.feature_names:
        raise TrackingError("Training feature metadata does not match the split.")
    if evaluation.evaluation_metadata.feature_names != split.feature_names:
        raise TrackingError("Evaluation feature metadata does not match the split.")
    if evaluation.evaluation_metadata.class_count != training.metadata.class_count:
        raise TrackingError("Training and evaluation class counts do not match.")
    if evaluation.evaluation_metadata.sklearn_version != (
        training.metadata.sklearn_version
    ):
        raise TrackingError("Training and evaluation Scikit-learn versions differ.")

    baseline_split = SplitConfig()
    if split.config != baseline_split:
        raise TrackingError(
            "Only the established fixed split can be logged as the baseline."
        )
    baseline_training = TrainingConfig()
    if training.metadata.model_configuration != baseline_training.to_dict():
        raise TrackingError(
            "Only the established Random Forest configuration can be logged as "
            "the baseline."
        )
    classifier = training.estimator.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "get_params"):
        raise TrackingError("The pipeline must contain a classifier step.")
    estimator_parameters = classifier.get_params(deep=False)
    if (
        estimator_parameters.get("n_estimators") != baseline_training.n_estimators
        or estimator_parameters.get("random_state") != baseline_training.random_state
    ):
        raise TrackingError(
            "The fitted estimator does not match the established baseline metadata."
        )

    try:
        dataset = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError) as error:
        raise TrackingError(
            f"Could not read dataset '{dataset_path}': {error}"
        ) from error
    reconstructed = pd.concat(
        [
            split.X_train.assign(**{split.y_train.name: split.y_train}),
            split.X_test.assign(**{split.y_test.name: split.y_test}),
        ]
    ).sort_index()
    if not reconstructed.equals(dataset):
        raise TrackingError(
            "The fingerprinted dataset does not match the train/test partitions."
        )


def _parameter_value(value: Any) -> str | int | float | bool:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _logged_parameters(training: TrainingResult, split: DataSplit) -> dict[str, Any]:
    classifier = training.estimator.named_steps["classifier"]
    parameters: dict[str, Any] = {
        "model_type": training.metadata.model_type,
        "n_estimators": training.metadata.model_configuration["n_estimators"],
        "random_seed": training.metadata.random_seed,
        "split_test_size": split.config.test_size,
        "split_random_state": split.config.random_state,
    }
    parameters.update(
        {
            f"estimator__{name}": _parameter_value(value)
            for name, value in classifier.get_params(deep=False).items()
        }
    )
    return parameters


def _logged_metrics(evaluation: EvaluationResult) -> dict[str, float | int]:
    return {
        "accuracy": evaluation.accuracy,
        "macro_precision": evaluation.macro_precision,
        "macro_recall": evaluation.macro_recall,
        "macro_f1": evaluation.macro_f1,
        "test_sample_count": evaluation.test_sample_count,
    }


def _logged_tags(
    training: TrainingResult,
    evaluation: EvaluationResult,
    split: DataSplit,
    dataset_path: Path,
    dataset_sha256: str,
    run_id: str,
    experiment_id: str,
) -> dict[str, str]:
    git_revision, git_dirty = _git_metadata()
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "dataset_sha256": dataset_sha256,
        "dataset_name": dataset_path.name,
        "dataset_rows": str(split.source_rows),
        "dataset_columns": str(len(split.feature_names) + 1),
        "feature_names": json.dumps(list(split.feature_names)),
        "crop_class_count": str(training.metadata.class_count),
        "training_sample_count": str(training.metadata.training_rows),
        "test_sample_count": str(evaluation.test_sample_count),
        "scikit_learn_version": training.metadata.sklearn_version,
        "python_version": platform.python_version(),
        "mlflow_version": mlflow.__version__,
        "git_revision": git_revision,
        "git_worktree_dirty": (
            "unavailable" if git_dirty is None else str(git_dirty).lower()
        ),
        "tracking_run_id": run_id,
        "tracking_experiment_id": experiment_id,
        "baseline_role": "established_baseline",
    }


def _evaluation_report(
    training: TrainingResult,
    evaluation: EvaluationResult,
    split: DataSplit,
    dataset_path: Path,
    dataset_sha256: str,
    experiment_id: str,
    run_id: str,
) -> dict[str, object]:
    return {
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "rows": split.source_rows,
            "columns": len(split.feature_names) + 1,
            "feature_names": list(split.feature_names),
            "class_count": training.metadata.class_count,
        },
        "protocol": {
            "name": EVALUATION_PROTOCOL,
            "test_size": split.config.test_size,
            "random_state": split.config.random_state,
            "training_rows": split.train_rows,
            "test_rows": split.test_rows,
        },
        "training": training.metadata.to_dict(),
        "evaluation": evaluation.to_dict(),
        "tracking": {
            "experiment_id": experiment_id,
            "run_id": run_id,
        },
    }


def _get_or_create_experiment(
    client: MlflowClient, config: MlflowTrackingConfig
) -> str:
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        if experiment.artifact_location != config.artifact_root_uri:
            raise TrackingError(
                "Existing MLflow experiment uses a different artifact root: "
                f"{experiment.artifact_location}"
            )
        return experiment.experiment_id

    try:
        return client.create_experiment(
            config.experiment_name,
            artifact_location=config.artifact_root_uri,
            tags={"project": "crop-recommendation-mlops"},
        )
    except MlflowException:
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            raise
        return experiment.experiment_id


def log_baseline_experiment(
    training: TrainingResult,
    evaluation: EvaluationResult,
    split: DataSplit,
    dataset_path: str | Path,
    config: MlflowTrackingConfig | None = None,
) -> TrackingRunResult:
    """Log and round-trip the already-fitted fixed baseline without fitting.

    The input example and inferred signature use only the training partition.
    The held-out partition contributes previously calculated evaluation outputs
    but is never used to fit, select, or modify the model.
    """

    effective_config = config if config is not None else MlflowTrackingConfig()
    resolved_dataset_path = Path(dataset_path).expanduser().resolve()
    _validate_tracking_inputs(
        training, evaluation, split, resolved_dataset_path, effective_config
    )
    dataset_sha256 = file_sha256(resolved_dataset_path)
    if not _SHA256_PATTERN.fullmatch(dataset_sha256):
        raise TrackingError("The calculated dataset fingerprint is not SHA-256.")

    input_example = split.X_train.iloc[
        : effective_config.input_example_rows
    ].copy(deep=True)
    expected_predictions = training.estimator.predict(input_example.copy(deep=True))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
        )
        signature = infer_signature(input_example, expected_predictions)

    effective_config.artifact_root.mkdir(parents=True, exist_ok=True)
    if mlflow.active_run() is not None:
        raise TrackingError(
            "An MLflow run is already active; finish it before logging the baseline."
        )

    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(effective_config.tracking_uri)
        client = MlflowClient(tracking_uri=effective_config.tracking_uri)
        experiment_id = _get_or_create_experiment(client, effective_config)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=effective_config.run_name,
        ) as active_run:
            run_id = active_run.info.run_id
            mlflow.log_params(_logged_parameters(training, split))
            mlflow.log_metrics(_logged_metrics(evaluation))
            mlflow.set_tags(
                _logged_tags(
                    training,
                    evaluation,
                    split,
                    resolved_dataset_path,
                    dataset_sha256,
                    run_id,
                    experiment_id,
                )
            )
            mlflow.log_dict(
                _evaluation_report(
                    training,
                    evaluation,
                    split,
                    resolved_dataset_path,
                    dataset_sha256,
                    experiment_id,
                    run_id,
                ),
                REPORT_ARTIFACT_PATH,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Hint: Inferred schema contains integer column",
                    category=UserWarning,
                )
                model_info = mlflow.sklearn.log_model(
                    sk_model=training.estimator,
                    name=MODEL_ARTIFACT_NAME,
                    signature=signature,
                    input_example=input_example,
                    metadata={
                        "dataset_sha256": dataset_sha256,
                        "evaluation_protocol": EVALUATION_PROTOCOL,
                    },
                    skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
                )

            loaded_model = mlflow.sklearn.load_model(model_info.model_uri)
            loaded_predictions = loaded_model.predict(input_example.copy(deep=True))
            if list(loaded_predictions) != list(expected_predictions):
                raise TrackingError(
                    "The loaded MLflow model did not reproduce input-example "
                    "predictions."
                )

            run_artifact_uri = active_run.info.artifact_uri
            model_id = getattr(model_info, "model_id", None)
            model_uri = model_info.model_uri

        return TrackingRunResult(
            experiment_name=effective_config.experiment_name,
            experiment_id=experiment_id,
            run_id=run_id,
            tracking_uri=effective_config.tracking_uri,
            run_artifact_uri=run_artifact_uri,
            model_name=MODEL_ARTIFACT_NAME,
            model_uri=model_uri,
            model_id=model_id,
            dataset_sha256=dataset_sha256,
        )
    except TrackingError:
        raise
    except (MlflowException, OSError, RuntimeError, ValueError) as error:
        raise TrackingError(
            "MLflow tracking failed for backend "
            f"'{effective_config.tracking_uri}': {error}"
        ) from error
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
