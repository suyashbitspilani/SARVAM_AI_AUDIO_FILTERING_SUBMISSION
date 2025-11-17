"""
Outlier detection modules for audio filtering pipeline.
"""

from .statistical import (
    modified_zscore,
    iqr_outlier_detection,
    detect_outliers_zscore,
    detect_outliers_iqr,
    compute_stratum_statistics
)

from .ensemble import (
    ensemble_outlier_score,
    voting_ensemble
)

__all__ = [
    'modified_zscore',
    'iqr_outlier_detection',
    'detect_outliers_zscore',
    'detect_outliers_iqr',
    'compute_stratum_statistics',
    'ensemble_outlier_score',
    'voting_ensemble'
]