"""
Ensemble methods for combining multiple outlier detection approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple


def ensemble_outlier_score(df: pd.DataFrame,
                          outlier_flags_dict: Dict[str, pd.DataFrame],
                          weights: Optional[Dict[str, float]] = None,
                          aggregation: str = 'weighted_mean') -> np.ndarray:
    """
    Combine multiple outlier detection methods using voting or weighted scoring.
    
    Args:
        df: DataFrame with samples
        outlier_flags_dict: Dict of {method_name: outlier_flags_df}
        weights: Dict of {method_name: weight} or None for equal weights
        aggregation: Method for aggregation ('weighted_mean', 'max', 'median')
    
    Returns:
        Array of ensemble scores [0, 1]
    """
    if len(outlier_flags_dict) == 0:
        return np.zeros(len(df))
    
    # Set default weights if not provided
    if weights is None:
        weights = {method: 1.0 for method in outlier_flags_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    else:
        weights = {k: 1.0/len(weights) for k in weights.keys()}
    
    # Initialize scores array
    outlier_scores = np.zeros(len(df), dtype=np.float64)
    method_scores = []
    
    # Calculate scores for each method
    for method, flags_df in outlier_flags_dict.items():
        if flags_df.empty:
            continue
            
        # Count outlier flags per sample across all metrics
        method_outlier_count = flags_df.sum(axis=1).values
        method_total_metrics = flags_df.shape[1]
        
        # Normalize to [0, 1] and ensure float64 type
        if method_total_metrics > 0:
            method_score = (method_outlier_count / method_total_metrics).astype(np.float64)
        else:
            method_score = np.zeros(len(df), dtype=np.float64)
        
        method_scores.append(method_score)
        
        if aggregation == 'weighted_mean':
            # Add weighted score with explicit float64 casting
            weight_value = float(weights.get(method, 0))
            outlier_scores += weight_value * method_score
    
    if aggregation == 'max':
        # Take maximum score across methods
        if method_scores:
            outlier_scores = np.max(method_scores, axis=0)
    elif aggregation == 'median':
        # Take median score across methods
        if method_scores:
            outlier_scores = np.median(method_scores, axis=0)
    
    return outlier_scores


def voting_ensemble(outlier_flags_dict: Dict[str, pd.DataFrame],
                   vote_threshold: float = 0.5,
                   metric_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple majority voting across methods.
    
    Args:
        outlier_flags_dict: Dict of {method_name: outlier_flags_df}
        vote_threshold: Fraction of methods that must agree for outlier
        metric_threshold: Fraction of metrics within a method to flag as outlier
    
    Returns:
        Tuple of (is_outlier boolean array, normalized vote scores)
    """
    if len(outlier_flags_dict) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    
    # Get number of samples from first method
    first_df = list(outlier_flags_dict.values())[0]
    n_samples = len(first_df)
    
    # Count how many methods flag each sample as outlier
    vote_counts = np.zeros(n_samples)
    
    for method_name, flags_df in outlier_flags_dict.items():
        if flags_df.empty:
            continue
            
        # Sample is outlier if sufficient metrics within method flag it
        method_outlier_ratio = flags_df.mean(axis=1)
        method_flags = method_outlier_ratio >= metric_threshold
        vote_counts += method_flags.values
    
    # Normalize vote counts
    n_methods = len(outlier_flags_dict)
    normalized_votes = vote_counts / max(n_methods, 1)
    
    # Flag as outlier if sufficient methods agree
    is_outlier = normalized_votes >= vote_threshold
    
    return is_outlier, normalized_votes


def adaptive_ensemble(df: pd.DataFrame,
                     outlier_flags_dict: Dict[str, pd.DataFrame],
                     metric_importance: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Adaptive ensemble that weights methods based on metric importance.
    
    Args:
        df: DataFrame with samples
        outlier_flags_dict: Dict of {method_name: outlier_flags_df}
        metric_importance: Dict of {metric_name: importance_weight}
    
    Returns:
        Array of ensemble scores [0, 1]
    """
    if len(outlier_flags_dict) == 0:
        return np.zeros(len(df))
    
    # Initialize scores
    outlier_scores = np.zeros(len(df))
    total_weight = 0
    
    for method_name, flags_df in outlier_flags_dict.items():
        if flags_df.empty:
            continue
        
        # Calculate weighted sum for this method
        method_score = np.zeros(len(df))
        
        for col in flags_df.columns:
            # Extract metric name from column (remove suffix like '_zscore_outlier')
            metric_name = col.split('_')[0]
            
            # Get importance weight for this metric
            if metric_importance and metric_name in metric_importance:
                weight = metric_importance[metric_name]
            else:
                weight = 1.0
            
            # Add weighted contribution
            method_score += flags_df[col].values * weight
            total_weight += weight
        
        outlier_scores += method_score
    
    # Normalize scores
    if total_weight > 0:
        outlier_scores /= total_weight
    
    return outlier_scores


def rank_based_ensemble(outlier_scores_dict: Dict[str, np.ndarray],
                       aggregation: str = 'mean') -> np.ndarray:
    """
    Combine outlier scores using rank-based aggregation.
    
    Args:
        outlier_scores_dict: Dict of {method_name: outlier_scores_array}
        aggregation: Method for rank aggregation ('mean', 'median', 'min')
    
    Returns:
        Combined outlier scores
    """
    if len(outlier_scores_dict) == 0:
        return np.zeros(0)
    
    # Get number of samples
    first_scores = list(outlier_scores_dict.values())[0]
    n_samples = len(first_scores)
    
    # Convert scores to ranks for each method
    ranks_list = []
    
    for method_name, scores in outlier_scores_dict.items():
        # Convert to ranks (higher score = higher rank)
        ranks = np.argsort(np.argsort(scores)) / (n_samples - 1)
        ranks_list.append(ranks)
    
    # Aggregate ranks
    ranks_array = np.array(ranks_list)
    
    if aggregation == 'mean':
        combined_ranks = np.mean(ranks_array, axis=0)
    elif aggregation == 'median':
        combined_ranks = np.median(ranks_array, axis=0)
    elif aggregation == 'min':
        combined_ranks = np.min(ranks_array, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return combined_ranks


def confidence_weighted_ensemble(df: pd.DataFrame,
                               outlier_flags_dict: Dict[str, pd.DataFrame],
                               confidence_scores: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Weight ensemble members by their confidence scores.
    
    Args:
        df: DataFrame with samples
        outlier_flags_dict: Dict of {method_name: outlier_flags_df}
        confidence_scores: Dict of {method_name: confidence_array}
    
    Returns:
        Confidence-weighted ensemble scores
    """
    if len(outlier_flags_dict) == 0:
        return np.zeros(len(df))
    
    # Initialize weighted scores
    weighted_scores = np.zeros(len(df))
    total_confidence = np.zeros(len(df))
    
    for method_name, flags_df in outlier_flags_dict.items():
        if flags_df.empty:
            continue
        
        # Get confidence for this method
        if method_name in confidence_scores:
            confidence = confidence_scores[method_name]
        else:
            confidence = np.ones(len(df))
        
        # Calculate outlier score for this method
        method_score = flags_df.mean(axis=1).values
        
        # Add weighted contribution
        weighted_scores += method_score * confidence
        total_confidence += confidence
    
    # Normalize by total confidence
    with np.errstate(divide='ignore', invalid='ignore'):
        ensemble_scores = np.where(
            total_confidence > 0,
            weighted_scores / total_confidence,
            0.0
        )
    
    return ensemble_scores


def isolation_forest_ensemble(df: pd.DataFrame,
                             metric_cols: List[str],
                             n_estimators: int = 100,
                             contamination: float = 0.1) -> np.ndarray:
    """
    Use Isolation Forest for anomaly detection.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns
        n_estimators: Number of trees in the forest
        contamination: Expected proportion of outliers
    
    Returns:
        Outlier scores (higher = more anomalous)
    """
    from sklearn.ensemble import IsolationForest
    
    # Get valid data
    valid_data = df[metric_cols].dropna()
    
    if len(valid_data) < 10:  # Need minimum samples
        return np.zeros(len(df))
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    
    # Get anomaly scores
    scores = np.zeros(len(df))
    
    # Predict returns -1 for outliers, 1 for inliers
    predictions = iso_forest.fit_predict(valid_data)
    
    # Get decision scores (negative = more anomalous)
    decision_scores = iso_forest.score_samples(valid_data)
    
    # Normalize to [0, 1] where 1 = most anomalous
    normalized_scores = 1 - (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
    
    scores[valid_data.index] = normalized_scores
    
    return scores