"""Model training interfaces for crop recommendation."""

from crop_recommendation.training.evaluate import (
    ClassMetrics,
    EvaluationError,
    EvaluationMetadata,
    EvaluationResult,
    evaluate_model,
)
from crop_recommendation.training.train import (
    TrainingConfig,
    TrainingError,
    TrainingMetadata,
    TrainingResult,
    create_training_pipeline,
    train_model,
)

__all__ = [
    "ClassMetrics",
    "EvaluationError",
    "EvaluationMetadata",
    "EvaluationResult",
    "TrainingConfig",
    "TrainingError",
    "TrainingMetadata",
    "TrainingResult",
    "create_training_pipeline",
    "evaluate_model",
    "train_model",
]
