"""Audio quality filtering rules."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def apply_hard_filters(df: pd.DataFrame, config: Dict) -> pd.Series:
    hard_reject = pd.Series(False, index=df.index)
    
    thresholds = config.get('hard_filters', {})
    
    if 'clipping_ratio' in df.columns:
        max_clipping = thresholds.get('max_clipping_ratio', 0.01)
        hard_reject |= df['clipping_ratio'] > max_clipping
    
    if 'duration' in df.columns:
        min_duration = thresholds.get('min_duration', 1.0)
        max_duration = thresholds.get('max_duration', 30.0)
        hard_reject |= (df['duration'] < min_duration) | (df['duration'] > max_duration)
    
    if 'snr' in df.columns:
        min_snr = thresholds.get('min_snr', -5.0)
        hard_reject |= df['snr'] < min_snr
    
    if 'silence_ratio' in df.columns:
        max_silence = thresholds.get('max_silence_ratio', 0.8)
        hard_reject |= df['silence_ratio'] > max_silence
    
    if 'active_speech_duration' in df.columns:
        min_speech = thresholds.get('min_active_speech', 0.5)
        hard_reject |= df['active_speech_duration'] < min_speech
    
    if 'rms_energy' in df.columns:
        min_energy = thresholds.get('min_rms_energy', 0.001)
        hard_reject |= df['rms_energy'] < min_energy
    
    if 'lang_confidence' in df.columns:
        min_lang_conf = thresholds.get('min_lang_confidence', 0.5)
        hard_reject |= df['lang_confidence'] < min_lang_conf
    
    if 'peak_amplitude' in df.columns:
        min_peak = thresholds.get('min_peak_amplitude', 0.01)
        hard_reject |= df['peak_amplitude'] < min_peak
    
    if 'spectral_centroid_mean' in df.columns:
        min_centroid = thresholds.get('min_spectral_centroid', 1000)
        max_centroid = thresholds.get('max_spectral_centroid', 8000)
        hard_reject |= (df['spectral_centroid_mean'] < min_centroid) | \
                      (df['spectral_centroid_mean'] > max_centroid)
    
    if 'pitch_mean' in df.columns:
        hard_reject |= (df['pitch_mean'] == 0) | df['pitch_mean'].isna()
    
    return hard_reject


def apply_soft_filters(df: pd.DataFrame, 
                      outlier_scores: np.ndarray,
                      config: Dict) -> pd.Series:
    """
    Apply soft filtering based on outlier scores.
    
    Args:
        df: DataFrame with samples
        outlier_scores: Array of outlier scores
        config: Configuration dictionary
    
    Returns:
        Boolean Series indicating samples to reject
    """
    # Get percentile threshold from config
    percentile_threshold = config.get('percentile_threshold', 70)
    
    # Language-specific overrides
    language_overrides = config.get('language_overrides', {})
    
    soft_reject = pd.Series(False, index=df.index)
    
    if 'lang' in df.columns:
        # Apply per-language thresholds
        for lang in df['lang'].unique():
            lang_mask = df['lang'] == lang
            lang_data = outlier_scores[lang_mask]
            
            if len(lang_data) == 0:
                continue
            
            # Get language-specific threshold or use default
            if lang in language_overrides:
                lang_threshold = language_overrides[lang].get('percentile_threshold', 
                                                              percentile_threshold)
            else:
                lang_threshold = percentile_threshold
            
            # Calculate threshold based on percentile
            score_threshold = np.percentile(lang_data, lang_threshold)
            
            # Flag samples above threshold
            soft_reject[lang_mask] = lang_data > score_threshold
    else:
        # Apply global threshold
        score_threshold = np.percentile(outlier_scores, percentile_threshold)
        soft_reject = pd.Series(outlier_scores > score_threshold, index=df.index)
    
    return soft_reject


def apply_contextual_filters(df: pd.DataFrame,
                            outlier_scores: np.ndarray,
                            config: Dict) -> pd.Series:
    """
    Apply contextual filtering based on speaker/session patterns.
    
    Args:
        df: DataFrame with samples
        outlier_scores: Array of outlier scores
        config: Configuration dictionary
    
    Returns:
        Boolean Series indicating samples to reject
    """
    contextual_reject = pd.Series(False, index=df.index)
    
    # Get thresholds from config
    speaker_outlier_threshold = config.get('speaker_outlier_threshold', 0.5)
    speaker_outlier_rate_threshold = config.get('speaker_outlier_rate_threshold', 0.5)
    district_outlier_threshold = config.get('district_outlier_threshold', 0.5)
    district_outlier_rate_threshold = config.get('district_outlier_rate_threshold', 0.7)
    min_samples_per_group = config.get('min_samples_per_group', 5)
    
    # Speaker-level check
    if 'speaker_id' in df.columns:
        speaker_scores = pd.Series(outlier_scores, index=df.index)
        
        # Calculate outlier rate per speaker
        def calc_speaker_rate(group):
            if len(group) >= min_samples_per_group:
                return (speaker_scores[group.index] > speaker_outlier_threshold).mean()
            return 0.0
        
        speaker_outlier_rates = df.groupby('speaker_id')['duration'].apply(calc_speaker_rate)
        
        # Flag speakers with high outlier rates
        bad_speakers = speaker_outlier_rates[
            speaker_outlier_rates > speaker_outlier_rate_threshold
        ].index
        
        contextual_reject |= df['speaker_id'].isin(bad_speakers)
    
    # District/location-level check
    if 'district' in df.columns:
        district_scores = pd.Series(outlier_scores, index=df.index)
        
        # Calculate outlier rate per district
        def calc_district_rate(group):
            if len(group) >= min_samples_per_group:
                return (district_scores[group.index] > district_outlier_threshold).mean()
            return 0.0
        
        district_outlier_rates = df.groupby('district')['duration'].apply(calc_district_rate)
        
        # Flag districts with high outlier rates
        bad_districts = district_outlier_rates[
            district_outlier_rates > district_outlier_rate_threshold
        ].index
        
        contextual_reject |= df['district'].isin(bad_districts)
    
    # Session-level check (if session IDs exist)
    if 'session_id' in df.columns:
        session_scores = pd.Series(outlier_scores, index=df.index)
        
        # Calculate outlier rate per session
        session_outlier_rates = df.groupby('session_id').apply(
            lambda x: (session_scores[x.index] > speaker_outlier_threshold).mean()
            if len(x) >= 3 else 0  # Minimum 3 samples per session
        )
        
        # Flag sessions with high outlier rates
        bad_sessions = session_outlier_rates[
            session_outlier_rates > 0.6  # Stricter for sessions
        ].index
        
        contextual_reject |= df['session_id'].isin(bad_sessions)
    
    # Age group check (optional)
    if 'age_group' in df.columns and config.get('check_age_groups', False):
        age_scores = pd.Series(outlier_scores, index=df.index)
        
        # Check for systematic issues in age groups
        age_outlier_rates = df.groupby('age_group').apply(
            lambda x: (age_scores[x.index] > 0.5).mean()
            if len(x) >= min_samples_per_group else 0
        )
        
        # Flag age groups with very high outlier rates
        bad_age_groups = age_outlier_rates[
            age_outlier_rates > 0.8
        ].index
        
        contextual_reject |= df['age_group'].isin(bad_age_groups)
    
    return contextual_reject


def make_filtering_decision(df: pd.DataFrame,
                           outlier_scores: np.ndarray,
                           config: Dict) -> pd.DataFrame:
    """
    Combine all filtering tiers to make final decision.
    
    Args:
        df: DataFrame with samples
        outlier_scores: Array of outlier scores
        config: Configuration dictionary
    
    Returns:
        DataFrame with filtering decisions and scores
    """
    # Apply different filtering tiers
    hard_reject = apply_hard_filters(df, config.get('filtering', {}))
    soft_reject = apply_soft_filters(df, outlier_scores, config.get('filtering', {}))
    contextual_reject = apply_contextual_filters(df, outlier_scores, 
                                                 config.get('filtering', {}))
    
    # Initialize decision column
    df = df.copy()
    df['filter_decision'] = 'KEEP'
    
    # Apply decisions in order of priority
    df.loc[contextual_reject & ~(hard_reject | soft_reject), 'filter_decision'] = 'REJECT_CONTEXTUAL'
    df.loc[soft_reject & ~hard_reject, 'filter_decision'] = 'REJECT_SOFT'
    df.loc[hard_reject, 'filter_decision'] = 'REJECT_HARD'
    
    # Add outlier score
    df['outlier_score'] = outlier_scores
    
    # Add detailed rejection reasons
    df['rejection_reasons'] = df.apply(
        lambda row: get_rejection_reason(row, hard_reject, soft_reject, 
                                        contextual_reject, config),
        axis=1
    )
    
    # Calculate quality score (inverse of outlier score)
    df['quality_score'] = 1 - outlier_scores
    
    # Add confidence level
    df['confidence_level'] = categorize_confidence(outlier_scores)
    
    return df


def get_rejection_reason(row: pd.Series,
                        hard_reject: pd.Series,
                        soft_reject: pd.Series,
                        contextual_reject: pd.Series,
                        config: Dict) -> List[str]:
    """
    Get detailed rejection reasons for a sample.
    
    Args:
        row: Sample row
        hard_reject: Hard rejection flags
        soft_reject: Soft rejection flags
        contextual_reject: Contextual rejection flags
        config: Configuration dictionary
    
    Returns:
        List of rejection reasons
    """
    reasons = []
    idx = row.name
    
    # Check hard filters
    if hard_reject[idx]:
        thresholds = config.get('filtering', {}).get('hard_filters', {})
        
        if 'clipping_ratio' in row and row['clipping_ratio'] > thresholds.get('max_clipping_ratio', 0.01):
            reasons.append(f"Excessive clipping ({row['clipping_ratio']:.1%})")
        
        if 'duration' in row:
            if row['duration'] < thresholds.get('min_duration', 1.0):
                reasons.append(f"Too short ({row['duration']:.1f}s)")
            elif row['duration'] > thresholds.get('max_duration', 30.0):
                reasons.append(f"Too long ({row['duration']:.1f}s)")
        
        if 'snr' in row and row['snr'] < thresholds.get('min_snr', -5.0):
            reasons.append(f"Low SNR ({row['snr']:.1f} dB)")
        
        if 'silence_ratio' in row and row['silence_ratio'] > thresholds.get('max_silence_ratio', 0.8):
            reasons.append(f"Excessive silence ({row['silence_ratio']:.1%})")
        
        if 'active_speech_duration' in row and row['active_speech_duration'] < thresholds.get('min_active_speech', 0.5):
            reasons.append(f"Insufficient speech ({row['active_speech_duration']:.1f}s)")
        
        if 'rms_energy' in row and row['rms_energy'] < thresholds.get('min_rms_energy', 0.001):
            reasons.append("Too quiet")
    
    # Check soft filters
    if soft_reject[idx] and not hard_reject[idx]:
        reasons.append(f"High outlier score ({row.get('outlier_score', 0):.2f})")
    
    # Check contextual filters
    if contextual_reject[idx] and not (hard_reject[idx] or soft_reject[idx]):
        if 'speaker_id' in row:
            reasons.append(f"Problematic speaker")
        if 'district' in row:
            reasons.append(f"Problematic recording location")
        if 'session_id' in row:
            reasons.append(f"Problematic session")
    
    return reasons


def categorize_confidence(outlier_scores: np.ndarray) -> np.ndarray:
    """
    Categorize samples by confidence level based on outlier scores.
    
    Args:
        outlier_scores: Array of outlier scores
    
    Returns:
        Array of confidence categories
    """
    confidence = np.empty(len(outlier_scores), dtype=object)
    
    confidence[outlier_scores < 0.2] = 'HIGH'
    confidence[(outlier_scores >= 0.2) & (outlier_scores < 0.4)] = 'MEDIUM-HIGH'
    confidence[(outlier_scores >= 0.4) & (outlier_scores < 0.6)] = 'MEDIUM'
    confidence[(outlier_scores >= 0.6) & (outlier_scores < 0.8)] = 'MEDIUM-LOW'
    confidence[outlier_scores >= 0.8] = 'LOW'
    
    return confidence


def apply_custom_rules(df: pd.DataFrame, rules: List[Dict]) -> pd.Series:
    """
    Apply custom user-defined filtering rules.
    
    Args:
        df: DataFrame with samples
        rules: List of rule dictionaries
    
    Returns:
        Boolean Series indicating samples to reject
    """
    custom_reject = pd.Series(False, index=df.index)
    
    for rule in rules:
        column = rule.get('column')
        operator = rule.get('operator')
        value = rule.get('value')
        
        if column not in df.columns:
            continue
        
        if operator == '>':
            custom_reject |= df[column] > value
        elif operator == '<':
            custom_reject |= df[column] < value
        elif operator == '>=':
            custom_reject |= df[column] >= value
        elif operator == '<=':
            custom_reject |= df[column] <= value
        elif operator == '==':
            custom_reject |= df[column] == value
        elif operator == '!=':
            custom_reject |= df[column] != value
        elif operator == 'in':
            custom_reject |= df[column].isin(value)
        elif operator == 'not_in':
            custom_reject |= ~df[column].isin(value)
    
    return custom_reject