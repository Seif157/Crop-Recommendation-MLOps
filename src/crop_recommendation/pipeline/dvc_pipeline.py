"""Deterministic DVC stages built from the existing Phase 1 interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import sklearn
import skops.io as skops_io
import yaml
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from crop_recommendation.data.preprocessing import (
    DataSplit,
    SplitConfig,
    prepare_data_split,
)
from crop_recommendation.data.validation import (
    FEATURE_COLUMNS,
    default_dataset_path,
    repository_root,
    validate_dataset,
)
from crop_recommendation.training.evaluate import evaluate_model
from crop_recommendation.training.train import (
    TrainingConfig,
    TrainingResult,
    train_model,
)

PIPELINE_SCHEMA_VERSION = 1
ARTIFACT_FORMAT = "skops"
SKOPS_TRUSTED_TYPES = ("sklearn.tree._tree.Tree",)

DEFAULT_PARAMS_PATH = repository_root() / "params.yaml"
DEFAULT_ARTIFACT_DIRECTORY = repository_root() / "artifacts" / "dvc"
DEFAULT_VALIDATION_REPORT_PATH = DEFAULT_ARTIFACT_DIRECTORY / "validation.json"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIRECTORY / "model.skops"
DEFAULT_TRAINING_REPORT_PATH = DEFAULT_ARTIFACT_DIRECTORY / "training.json"
DEFAULT_EVALUATION_REPORT_PATH = DEFAULT_ARTIFACT_DIRECTORY / "evaluation.json"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIRECTORY / "metrics.json"


class PipelineConfigurationError(ValueError):
    """Raised when params.yaml is missing or violates the pipeline contract."""


class PipelineStageError(RuntimeError):
    """Raised when a deterministic pipeline stage cannot complete safely."""


class PipelineArtifactError(PipelineStageError):
    """Raised when a serialized pipeline artifact is missing or incompatible."""


@dataclass(frozen=True)
class DvcPipelineConfig:
    """Validated split and estimator configuration loaded from params.yaml."""

    split: SplitConfig
    model: TrainingConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "split": {
                "test_size": self.split.test_size,
                "random_state": self.split.random_state,
            },
            "model": self.model.to_dict(),
        }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PipelineConfigurationError(f"'{name}' must be a YAML mapping.")
    return value


def _require_exact_keys(
    mapping: Mapping[str, object], expected: set[str], name: str
) -> None:
    actual = {str(key) for key in mapping}
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing:
        raise PipelineConfigurationError(
            f"'{name}' is missing required key(s): {', '.join(missing)}."
        )
    if unexpected:
        raise PipelineConfigurationError(
            f"'{name}' contains unexpected key(s): {', '.join(unexpected)}."
        )


def load_pipeline_config(params_path: str | Path = DEFAULT_PARAMS_PATH) -> DvcPipelineConfig:
    """Load and validate the DVC pipeline's single parameter source."""

    path = Path(params_path).expanduser().resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PipelineConfigurationError(f"Parameter file does not exist: {path}") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PipelineConfigurationError(
            f"Could not read parameter file '{path}': {error}"
        ) from error

    root = _require_mapping(raw, "params")
    _require_exact_keys(root, {"split", "model"}, "params")
    split_values = _require_mapping(root["split"], "split")
    model_values = _require_mapping(root["model"], "model")
    _require_exact_keys(split_values, {"test_size", "random_state"}, "split")
    _require_exact_keys(model_values, {"n_estimators", "random_state"}, "model")

    try:
        split = SplitConfig(
            test_size=split_values["test_size"],  # type: ignore[arg-type]
            random_state=split_values["random_state"],  # type: ignore[arg-type]
        )
        model = TrainingConfig(
            n_estimators=model_values["n_estimators"],  # type: ignore[arg-type]
            random_state=model_values["random_state"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise PipelineConfigurationError(f"Invalid pipeline parameter: {error}") from error
    return DvcPipelineConfig(split=split, model=model)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PipelineStageError(f"Could not hash '{path}': {error}") from error
    return digest.hexdigest()


def _dataset_identity(dataset_path: Path) -> dict[str, str | int]:
    if not dataset_path.is_file():
        raise PipelineStageError(f"Dataset does not exist: {dataset_path}")
    return {
        "path": _portable_path(dataset_path),
        "size": dataset_path.stat().st_size,
        "sha256": _hash_file(dataset_path, "sha256"),
        "dvc_md5": _hash_file(dataset_path, "md5"),
    }


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return resolved.name


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(serialized, encoding="utf-8", newline="\n")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PipelineStageError(f"{description} does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineStageError(f"Could not read {description} '{path}': {error}") from error
    if not isinstance(value, dict):
        raise PipelineStageError(f"{description} must contain a JSON object.")
    return value


def _index_fingerprint(values: pd.Index) -> str:
    serialized = json.dumps(
        [int(value) if isinstance(value, int) else str(value) for value in values],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _prediction_fingerprint(predictions: Sequence[str]) -> str:
    serialized = json.dumps(list(predictions), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _partition_metadata(split: DataSplit) -> dict[str, object]:
    return {
        "source_rows": split.source_rows,
        "train_rows": split.train_rows,
        "test_rows": split.test_rows,
        "train_index_sha256": _index_fingerprint(split.X_train.index),
        "test_index_sha256": _index_fingerprint(split.X_test.index),
        "train_class_distribution": split.train_class_distribution,
        "test_class_distribution": split.test_class_distribution,
    }


def _validate_upstream_report(
    report: Mapping[str, Any], dataset_identity: Mapping[str, str | int]
) -> None:
    if report.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise PipelineStageError("Validation report schema version is incompatible.")
    if report.get("stage") != "validate":
        raise PipelineStageError("Expected a validation-stage report.")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or validation.get("validation_status") != "valid":
        raise PipelineStageError("The upstream validation report is not valid.")
    if report.get("dataset") != dataset_identity:
        raise PipelineStageError(
            "The validation report does not match the current dataset content."
        )


def run_validation_stage(
    dataset_path: str | Path = default_dataset_path(),
    output_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
) -> dict[str, object]:
    """Validate the source CSV and write a deterministic report."""

    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    report = validate_dataset(dataset)
    if not report.is_valid:
        details = "; ".join(report.validation_errors)
        raise PipelineStageError(f"Dataset validation failed: {details}")

    validation = report.to_dict()
    validation["dataset_path"] = _portable_path(dataset)
    payload: dict[str, object] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "validate",
        "dataset": _dataset_identity(dataset),
        "validation": validation,
    }
    _write_json(output, payload)
    return payload


def _dump_and_verify_model(
    training: TrainingResult, model_path: Path, input_example: pd.DataFrame
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        skops_io.dump(training.estimator, model_path)
    except (OSError, TypeError, ValueError) as error:
        raise PipelineArtifactError(f"Could not serialize model: {error}") from error

    loaded = load_trusted_pipeline(model_path, training.metadata.feature_names)
    expected = training.estimator.predict(input_example).tolist()
    actual = loaded.predict(input_example).tolist()
    if actual != expected:
        raise PipelineArtifactError(
            "The serialized model did not reproduce training-example predictions."
        )


def load_trusted_pipeline(
    model_path: str | Path,
    expected_features: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Load only the locally produced Skops pipeline with an exact schema."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise PipelineArtifactError(f"Model artifact does not exist: {path}")
    try:
        untrusted_types = tuple(sorted(skops_io.get_untrusted_types(file=path)))
    except (OSError, TypeError, ValueError) as error:
        raise PipelineArtifactError(f"Could not inspect model artifact: {error}") from error
    if untrusted_types != SKOPS_TRUSTED_TYPES:
        raise PipelineArtifactError(
            "Model artifact contains unexpected untrusted types: "
            + ", ".join(untrusted_types)
        )
    try:
        estimator = skops_io.load(path, trusted=list(SKOPS_TRUSTED_TYPES))
    except (OSError, TypeError, ValueError) as error:
        raise PipelineArtifactError(f"Could not load model artifact: {error}") from error
    if not isinstance(estimator, Pipeline):
        raise PipelineArtifactError("Model artifact is not a Scikit-learn Pipeline.")
    try:
        check_is_fitted(estimator)
    except (TypeError, ValueError) as error:
        raise PipelineArtifactError("Model artifact is not fitted.") from error
    fitted_features = tuple(str(value) for value in estimator.feature_names_in_)
    if fitted_features != tuple(expected_features):
        raise PipelineArtifactError(
            "Model feature schema does not match the expected ordered features."
        )
    return estimator


def run_training_stage(
    dataset_path: str | Path = default_dataset_path(),
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    validation_report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    training_report_path: str | Path = DEFAULT_TRAINING_REPORT_PATH,
) -> dict[str, object]:
    """Recreate the fixed split and fit only its training partition."""

    dataset = Path(dataset_path).expanduser().resolve()
    validation_report = Path(validation_report_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    training_report = Path(training_report_path).expanduser().resolve()
    config = load_pipeline_config(params_path)
    dataset_identity = _dataset_identity(dataset)
    _validate_upstream_report(
        _read_json(validation_report, "validation report"), dataset_identity
    )

    split = prepare_data_split(dataset, config.split)
    training = train_model(split.X_train, split.y_train, config.model)
    example = split.X_train.iloc[:5].copy(deep=True)
    _dump_and_verify_model(training, model, example)

    payload: dict[str, object] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "train",
        "dataset": dataset_identity,
        "configuration": config.to_dict(),
        "partition": _partition_metadata(split),
        "training": training.metadata.to_dict(),
        "model": {
            "path": _portable_path(model),
            "format": ARTIFACT_FORMAT,
            "feature_names": list(training.metadata.feature_names),
            "trusted_types": list(SKOPS_TRUSTED_TYPES),
        },
        "software": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },
    }
    _write_json(training_report, payload)
    return payload


def _validate_training_report(
    report: Mapping[str, Any],
    dataset_identity: Mapping[str, str | int],
    config: DvcPipelineConfig,
    split: DataSplit,
) -> None:
    if report.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise PipelineArtifactError("Training report schema version is incompatible.")
    if report.get("stage") != "train":
        raise PipelineArtifactError("Expected a training-stage report.")
    if report.get("dataset") != dataset_identity:
        raise PipelineArtifactError("Training report dataset identity is incompatible.")
    if report.get("configuration") != config.to_dict():
        raise PipelineArtifactError("Training report configuration is incompatible.")
    if report.get("partition") != _partition_metadata(split):
        raise PipelineArtifactError("Training report partition identity is incompatible.")
    model_metadata = report.get("model")
    if not isinstance(model_metadata, Mapping):
        raise PipelineArtifactError("Training report is missing model metadata.")
    if model_metadata.get("format") != ARTIFACT_FORMAT:
        raise PipelineArtifactError("Training report model format is incompatible.")
    if model_metadata.get("feature_names") != list(FEATURE_COLUMNS):
        raise PipelineArtifactError("Training report feature schema is incompatible.")


def run_evaluation_stage(
    dataset_path: str | Path = default_dataset_path(),
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    validation_report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    training_report_path: str | Path = DEFAULT_TRAINING_REPORT_PATH,
    evaluation_report_path: str | Path = DEFAULT_EVALUATION_REPORT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> tuple[dict[str, object], dict[str, float | int]]:
    """Load the trusted model and evaluate without fitting or mutation."""

    dataset = Path(dataset_path).expanduser().resolve()
    validation_report = Path(validation_report_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    training_report = Path(training_report_path).expanduser().resolve()
    evaluation_report = Path(evaluation_report_path).expanduser().resolve()
    metrics_output = Path(metrics_path).expanduser().resolve()
    config = load_pipeline_config(params_path)
    dataset_identity = _dataset_identity(dataset)
    _validate_upstream_report(
        _read_json(validation_report, "validation report"), dataset_identity
    )
    split = prepare_data_split(dataset, config.split)
    stored_training = _read_json(training_report, "training report")
    _validate_training_report(stored_training, dataset_identity, config, split)
    estimator = load_trusted_pipeline(model, split.feature_names)

    classifier = estimator.named_steps.get("classifier")
    if classifier is None:
        raise PipelineArtifactError("Model pipeline has no classifier step.")
    actual_parameters = {
        "n_estimators": classifier.get_params(deep=False).get("n_estimators"),
        "random_state": classifier.get_params(deep=False).get("random_state"),
    }
    if actual_parameters != config.model.to_dict():
        raise PipelineArtifactError("Model parameters do not match params.yaml.")

    evaluation = evaluate_model(estimator, split.X_test, split.y_test)
    predictions = [str(value) for value in estimator.predict(split.X_test)]
    metrics: dict[str, float | int] = {
        "accuracy": evaluation.accuracy,
        "macro_precision": evaluation.macro_precision,
        "macro_recall": evaluation.macro_recall,
        "macro_f1": evaluation.macro_f1,
        "test_sample_count": evaluation.test_sample_count,
    }
    report: dict[str, object] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "stage": "evaluate",
        "dataset": dataset_identity,
        "configuration": config.to_dict(),
        "partition": _partition_metadata(split),
        "evaluation": evaluation.to_dict(),
        "predictions": predictions,
        "prediction_sha256": _prediction_fingerprint(predictions),
    }
    _write_json(evaluation_report, report)
    _write_json(metrics_output, metrics)
    return report, metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one deterministic stage of the crop DVC pipeline."
    )
    parser.add_argument("stage", choices=("validate", "train", "evaluate"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.stage == "validate":
        run_validation_stage()
        print(f"Validation report: {DEFAULT_VALIDATION_REPORT_PATH}")
    elif arguments.stage == "train":
        run_training_stage()
        print(f"Model artifact: {DEFAULT_MODEL_PATH}")
        print(f"Training report: {DEFAULT_TRAINING_REPORT_PATH}")
    else:
        run_evaluation_stage()
        print(f"Evaluation report: {DEFAULT_EVALUATION_REPORT_PATH}")
        print(f"DVC metrics: {DEFAULT_METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
