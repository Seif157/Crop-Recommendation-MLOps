"""Explicit MLflow logging for completed training-only CV comparisons."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from crop_recommendation.experiments.comparison import (
    EXPERIMENT_PROTOCOL,
    METRIC_NAMES,
    CandidateResult,
    ComparisonResult,
)
from crop_recommendation.tracking.mlflow_tracking import (
    MlflowTrackingConfig,
    default_artifact_root,
    default_tracking_uri,
)

DEFAULT_CV_EXPERIMENT_NAME = "crop-recommendation-cv"
DEFAULT_COMPARISON_REPORT_PATH = (
    Path(__file__).resolve().parents[3]
    / ".mlflow"
    / "comparisons"
    / "latest.json"
)


class ComparisonTrackingError(RuntimeError):
    """Raised when candidate evidence cannot be persisted completely."""


@dataclass(frozen=True)
class CandidateRunRecord:
    """MLflow identity for one successfully logged candidate run."""

    candidate_name: str
    experiment_id: str
    run_id: str
    artifact_uri: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LoggedComparison:
    """Stable comparison evidence plus runtime-specific MLflow identities."""

    comparison: ComparisonResult
    experiment_name: str
    experiment_id: str
    tracking_uri: str
    runs: tuple[CandidateRunRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison": self.comparison.to_dict(),
            "tracking": {
                "experiment_name": self.experiment_name,
                "experiment_id": self.experiment_id,
                "tracking_uri": self.tracking_uri,
                "runs": [run.to_dict() for run in self.runs],
            },
        }


def default_cv_tracking_config() -> MlflowTrackingConfig:
    """Use the established backend/artifact locations in a separate experiment."""

    return MlflowTrackingConfig(
        tracking_uri=default_tracking_uri(),
        artifact_root=default_artifact_root(),
        experiment_name=DEFAULT_CV_EXPERIMENT_NAME,
        run_name="candidate",
    )


def _git_metadata() -> tuple[str, bool | None]:
    repository = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unavailable", None
    return revision or "unavailable", bool(status.strip())


def _get_or_create_experiment(
    client: MlflowClient, config: MlflowTrackingConfig
) -> str:
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        if experiment.artifact_location != config.artifact_root_uri:
            raise ComparisonTrackingError(
                "Existing CV experiment uses a different artifact root: "
                f"{experiment.artifact_location}"
            )
        return experiment.experiment_id
    try:
        return client.create_experiment(
            config.experiment_name,
            artifact_location=config.artifact_root_uri,
            tags={
                "project": "crop-recommendation-mlops",
                "protocol": EXPERIMENT_PROTOCOL,
            },
        )
    except MlflowException:
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            raise
        return experiment.experiment_id


def _parameters(
    comparison: ComparisonResult, candidate: CandidateResult
) -> dict[str, str | int | bool | float]:
    config = comparison.configuration
    return {
        "candidate_name": candidate.candidate.name,
        "n_estimators": candidate.candidate.n_estimators,
        "model_random_state": candidate.candidate.random_state,
        "cv_fold_count": config.cv_folds,
        "cv_split_seed": config.cv_random_state,
        "cv_shuffle": config.cv_shuffle,
        "primary_metric": config.primary_metric,
        "scoring_configuration": json.dumps(list(METRIC_NAMES)),
        "standard_deviation": "population",
        "selection_tolerance": config.tie_tolerance,
    }


def _metrics(candidate: CandidateResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in METRIC_NAMES:
        metrics[f"mean_cv_{metric}"] = candidate.mean_metrics[metric]
        metrics[f"std_cv_{metric}"] = candidate.standard_deviation_metrics[metric]
    for fold in candidate.folds:
        for metric, value in fold.metrics.items():
            metrics[f"fold_{fold.fold}_{metric}"] = value
    return metrics


def _tags(
    comparison: ComparisonResult,
    candidate: CandidateResult,
    experiment_id: str,
    run_id: str,
) -> dict[str, str]:
    revision, dirty = _git_metadata()
    return {
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "dataset_sha256": comparison.dataset.sha256,
        "dvc_dataset_md5": comparison.dataset.dvc_md5,
        "training_partition_sha256": str(
            comparison.training_partition["sha256"]
        ),
        "outer_holdout_used": "false",
        "outer_holdout_role": "untouched_final_evaluation",
        "feature_names": json.dumps(
            comparison.training_partition["feature_names"]
        ),
        "python_version": candidate.python_version,
        "scikit_learn_version": candidate.scikit_learn_version,
        "mlflow_version": mlflow.__version__,
        "git_revision": revision,
        "git_worktree_dirty": (
            "unavailable" if dirty is None else str(dirty).lower()
        ),
        "tracking_experiment_id": experiment_id,
        "tracking_run_id": run_id,
        "recommendation_status": (
            "recommended"
            if candidate.candidate.name == comparison.recommended_candidate
            else "not_recommended"
        ),
        "production_approval": "not_evaluated",
    }


def _fold_artifact(candidate: CandidateResult) -> dict[str, object]:
    return {
        "candidate_name": candidate.candidate.name,
        "folds": [fold.to_dict() for fold in candidate.folds],
    }


def log_comparison(
    comparison: ComparisonResult,
    config: MlflowTrackingConfig | None = None,
) -> LoggedComparison:
    """Log exactly one result-only MLflow run per completed candidate."""

    effective_config = config if config is not None else default_cv_tracking_config()
    if mlflow.active_run() is not None:
        raise ComparisonTrackingError(
            "An MLflow run is already active; finish it before logging comparison."
        )
    previous_uri = mlflow.get_tracking_uri()
    try:
        effective_config.artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(effective_config.tracking_uri)
        client = MlflowClient(tracking_uri=effective_config.tracking_uri)
        experiment_id = _get_or_create_experiment(client, effective_config)
        run_records: list[CandidateRunRecord] = []
        compact_comparison: dict[str, Any] = {
            "ranking": list(comparison.ranking),
            "best_observed_candidate": comparison.best_observed_candidate,
            "recommended_candidate": comparison.recommended_candidate,
            "selection_policy": {
                "primary_metric": comparison.configuration.primary_metric,
                "tie_tolerance": comparison.configuration.tie_tolerance,
                "tie_breaker": "fewer_estimators",
            },
        }
        for candidate in comparison.candidates:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=candidate.candidate.name,
            ) as active_run:
                run_id = active_run.info.run_id
                mlflow.log_params(_parameters(comparison, candidate))
                mlflow.log_metrics(_metrics(candidate))
                mlflow.set_tags(
                    _tags(comparison, candidate, experiment_id, run_id)
                )
                mlflow.log_dict(candidate.to_dict(), "reports/candidate.json")
                mlflow.log_dict(
                    comparison.configuration.to_dict(),
                    "reports/cv_configuration.json",
                )
                mlflow.log_dict(
                    _fold_artifact(candidate), "reports/fold_metrics.json"
                )
                mlflow.log_dict(
                    compact_comparison, "reports/comparison_summary.json"
                )
                run_records.append(
                    CandidateRunRecord(
                        candidate_name=candidate.candidate.name,
                        experiment_id=experiment_id,
                        run_id=run_id,
                        artifact_uri=active_run.info.artifact_uri,
                    )
                )
        return LoggedComparison(
            comparison=comparison,
            experiment_name=effective_config.experiment_name,
            experiment_id=experiment_id,
            tracking_uri=effective_config.tracking_uri,
            runs=tuple(run_records),
        )
    except ComparisonTrackingError:
        raise
    except (MlflowException, OSError, RuntimeError, ValueError) as error:
        raise ComparisonTrackingError(
            "MLflow comparison logging failed for backend "
            f"'{effective_config.tracking_uri}': {error}"
        ) from error
    finally:
        mlflow.set_tracking_uri(previous_uri)
