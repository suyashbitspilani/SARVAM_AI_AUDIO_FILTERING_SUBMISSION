"""
Statistical outlier detection methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def modified_zscore(data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified Z-score using Median Absolute Deviation (MAD).
    More robust than standard Z-score against outliers.
    
    Args:
        data: Array of values
        threshold: Z-score threshold for outlier detection
    
    Returns:
        Tuple of (is_outlier boolean array, modified z-scores)
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Avoid division by zero
    if mad == 0:
        mad = np.mean(np.abs(data - median))
        if mad == 0:
            # All values are the same
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    
    # Modified Z-score formula
    modified_z_scores = 0.6745 * (data - median) / mad
    
    # Flag outliers
    is_outlier = np.abs(modified_z_scores) > threshold
    
    return is_outlier, modified_z_scores


def iqr_outlier_detection(data: np.ndarray, 
                         iqr_multiplier: float = 1.5) -> Tuple[np.ndarray, float, float]:
    """
    IQR-based outlier detection using Tukey's method.
    
    Args:
        data: Array of values
        iqr_multiplier: Multiplier for IQR to determine outlier bounds
    
    Returns:
        Tuple of (is_outlier boolean array, lower_bound, upper_bound)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Calculate bounds
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    # Flag outliers
    is_outlier = (data < lower_bound) | (data > upper_bound)
    
    return is_outlier, lower_bound, upper_bound


def detect_outliers_zscore(df: pd.DataFrame, 
                          metric_cols: List[str],
                          stratum_cols: List[str] = ['lang', 'task_name'],
                          threshold: float = 3.5) -> pd.DataFrame:
    """
    Detect outliers using modified Z-score per stratum.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns to check
        stratum_cols: Columns to group by for stratified analysis
        threshold: Z-score threshold
    
    Returns:
        DataFrame with outlier flags for each metric
    """
    outlier_flags = pd.DataFrame(index=df.index)
    
    # Group by stratum
    for stratum, group in df.groupby(stratum_cols):
        if len(group) < 3:  # Need minimum samples for statistics
            continue
            
        for metric in metric_cols:
            if metric not in group.columns:
                continue
                
            col_name = f'{metric}_zscore_outlier'
            
            # Get non-null values
            valid_data = group[metric].dropna()
            if len(valid_data) < 3:
                outlier_flags.loc[group.index, col_name] = False
                continue
            
            # Detect outliers
            is_outlier, z_scores = modified_zscore(valid_data.values, threshold)
            
            # Store results
            outlier_flags.loc[valid_data.index, col_name] = is_outlier
            
            # Fill NaN values as False (not outliers)
            null_indices = group[group[metric].isna()].index
            outlier_flags.loc[null_indices, col_name] = False
    
    return outlier_flags


def detect_outliers_iqr(df: pd.DataFrame,
                        metric_cols: List[str],
                        stratum_cols: List[str] = ['lang', 'task_name'],
                        iqr_multiplier: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect outliers using IQR method per stratum.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns to check
        stratum_cols: Columns to group by for stratified analysis
        iqr_multiplier: IQR multiplier for bounds
    
    Returns:
        Tuple of (outlier flags DataFrame, bounds dictionary)
    """
    outlier_flags = pd.DataFrame(index=df.index)
    bounds = {}
    
    # Group by stratum
    for stratum, group in df.groupby(stratum_cols):
        if len(group) < 4:  # Need minimum samples for IQR
            continue
            
        # Convert stratum tuple to string for dict key
        stratum_key = '_'.join(map(str, stratum)) if isinstance(stratum, tuple) else str(stratum)
        bounds[stratum_key] = {}
        
        for metric in metric_cols:
            if metric not in group.columns:
                continue
                
            col_name = f'{metric}_iqr_outlier'
            
            # Get non-null values
            valid_data = group[metric].dropna()
            if len(valid_data) < 4:
                outlier_flags.loc[group.index, col_name] = False
                continue
            
            # Detect outliers
            is_outlier, lower, upper = iqr_outlier_detection(
                valid_data.values, 
                iqr_multiplier
            )
            
            # Store results
            outlier_flags.loc[valid_data.index, col_name] = is_outlier
            
            # Fill NaN values as False
            null_indices = group[group[metric].isna()].index
            outlier_flags.loc[null_indices, col_name] = False
            
            # Store bounds
            bounds[stratum_key][metric] = {
                'lower': float(lower),
                'upper': float(upper),
                'q1': float(np.percentile(valid_data, 25)),
                'q3': float(np.percentile(valid_data, 75)),
                'median': float(np.median(valid_data))
            }
    
    return outlier_flags, bounds


def compute_stratum_statistics(df: pd.DataFrame,
                              metric_cols: List[str],
                              stratum_cols: List[str] = ['lang', 'task_name']) -> Dict:
    """
    Compute robust statistics for each stratum.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns
        stratum_cols: Columns to group by
    
    Returns:
        Dictionary with statistics per stratum and metric
    """
    stats = {}
    
    grouped = df.groupby(stratum_cols)
    
    for stratum, group in grouped:
        # Convert stratum tuple to string for dict key
        stratum_key = '_'.join(map(str, stratum)) if isinstance(stratum, tuple) else str(stratum)
        stats[stratum_key] = {
            'n_samples': len(group)
        }
        
        for metric in metric_cols:
            if metric not in group.columns:
                continue
                
            valid_data = group[metric].dropna()
            
            if len(valid_data) == 0:
                continue
            
            # Compute various statistics
            stats[stratum_key][metric] = {
                'count': len(valid_data),
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'std': float(np.std(valid_data)),
                'mad': float(np.median(np.abs(valid_data - np.median(valid_data)))),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'q05': float(np.percentile(valid_data, 5)),
                'q25': float(np.percentile(valid_data, 25)),
                'q75': float(np.percentile(valid_data, 75)),
                'q95': float(np.percentile(valid_data, 95)),
                'iqr': float(np.percentile(valid_data, 75) - np.percentile(valid_data, 25)),
                'range': float(np.max(valid_data) - np.min(valid_data)),
                'cv': float(np.std(valid_data) / np.mean(valid_data)) if np.mean(valid_data) != 0 else 0.0
            }
    
    return stats


def compute_percentile_thresholds(df: pd.DataFrame,
                                 metric_cols: List[str],
                                 percentiles: List[float] = [1, 5, 10, 90, 95, 99]) -> Dict:
    """
    Compute percentile thresholds for each metric.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns
        percentiles: List of percentiles to compute
    
    Returns:
        Dictionary with percentile values for each metric
    """
    thresholds = {}
    
    for metric in metric_cols:
        if metric not in df.columns:
            continue
            
        valid_data = df[metric].dropna()
        
        if len(valid_data) == 0:
            continue
        
        thresholds[metric] = {}
        for p in percentiles:
            thresholds[metric][f'p{p:02d}'] = float(np.percentile(valid_data, p))
    
    return thresholds


def compute_robust_covariance(df: pd.DataFrame, 
                             metric_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute robust covariance matrix using Minimum Covariance Determinant.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns
    
    Returns:
        Tuple of (robust mean, robust covariance matrix)
    """
    from sklearn.covariance import MinCovDet
    
    # Get valid data
    valid_data = df[metric_cols].dropna()
    
    if len(valid_data) < len(metric_cols) + 1:
        # Not enough samples for robust covariance
        return np.mean(valid_data, axis=0), np.cov(valid_data.T)
    
    # Compute robust estimates
    mcd = MinCovDet(random_state=42)
    mcd.fit(valid_data)
    
    return mcd.location_, mcd.covariance_


def mahalanobis_outlier_detection(df: pd.DataFrame,
                                 metric_cols: List[str],
                                 threshold: float = 3.0) -> np.ndarray:
    """
    Detect multivariate outliers using Mahalanobis distance.
    
    Args:
        df: DataFrame with samples
        metric_cols: List of metric columns
        threshold: Chi-squared threshold (in standard deviations)
    
    Returns:
        Boolean array of outliers
    """
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import chi2
    
    # Get valid data
    valid_data = df[metric_cols].dropna()
    
    if len(valid_data) < len(metric_cols) + 1:
        return np.zeros(len(df), dtype=bool)
    
    # Compute robust mean and covariance
    robust_mean, robust_cov = compute_robust_covariance(df, metric_cols)
    
    # Compute Mahalanobis distance for each sample
    distances = np.zeros(len(valid_data))
    inv_cov = np.linalg.pinv(robust_cov)
    
    for i, row in enumerate(valid_data.values):
        diff = row - robust_mean
        distances[i] = np.sqrt(diff @ inv_cov @ diff)
    
    # Determine threshold using chi-squared distribution
    dof = len(metric_cols)  # Degrees of freedom
    chi2_threshold = chi2.ppf(1 - 0.01, dof)  # 99% confidence
    
    # Flag outliers
    outliers = np.zeros(len(df), dtype=bool)
    outliers[valid_data.index] = distances > np.sqrt(chi2_threshold)
    
    return outliers