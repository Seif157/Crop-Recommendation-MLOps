"""Controlled, training-only experiment comparison interfaces."""

from crop_recommendation.experiments.comparison import (
    CandidateConfig,
    CandidateResult,
    ComparisonError,
    ComparisonResult,
    DatasetLineage,
    ExperimentConfig,
    ExperimentConfigurationError,
    FoldAssignment,
    FoldMetrics,
    build_cv_folds,
    compare_candidates,
    default_candidates,
    rank_candidates,
    training_partition_fingerprint,
    write_comparison_report,
)
from crop_recommendation.experiments.tracking import (
    CandidateRunRecord,
    ComparisonTrackingError,
    LoggedComparison,
    default_cv_tracking_config,
    log_comparison,
)

__all__ = [
    "CandidateConfig",
    "CandidateResult",
    "CandidateRunRecord",
    "ComparisonError",
    "ComparisonResult",
    "ComparisonTrackingError",
    "DatasetLineage",
    "ExperimentConfig",
    "ExperimentConfigurationError",
    "FoldAssignment",
    "FoldMetrics",
    "LoggedComparison",
    "build_cv_folds",
    "compare_candidates",
    "default_candidates",
    "default_cv_tracking_config",
    "log_comparison",
    "rank_candidates",
    "training_partition_fingerprint",
    "write_comparison_report",
]
