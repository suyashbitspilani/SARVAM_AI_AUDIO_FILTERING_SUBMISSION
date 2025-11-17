"""
Data handling and storage utilities.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def save_metrics(results: List[Dict], language: str, output_dir: str) -> str:
    """
    Save computed metrics efficiently using Parquet format.
    
    Args:
        results: List of dictionaries with computed metrics
        language: Language identifier
        output_dir: Output directory
    
    Returns:
        Path to saved metrics file
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet (efficient columnar format)
    output_path = os.path.join(output_dir, f"{language}_metrics.parquet")
    df.to_parquet(output_path, compression='snappy', index=False)
    
    logger.info(f"Saved metrics for {language}: {len(df)} samples to {output_path}")
    
    # Also save basic info as JSON for quick inspection
    info = {
        'language': language,
        'n_samples': len(df),
        'n_successful': len(df[df['processing_status'] == 'success']),
        'n_failed': len(df[df['processing_status'] == 'failed']),
        'columns': list(df.columns),
        'success_rate': (df['processing_status'] == 'success').mean()
    }
    
    info_path = os.path.join(output_dir, f"{language}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return output_path


def load_metrics(language: str, metrics_dir: str) -> Optional[pd.DataFrame]:
    """
    Load precomputed metrics for a language.
    
    Args:
        language: Language identifier
        metrics_dir: Directory containing metrics files
    
    Returns:
        DataFrame with metrics or None if not found
    """
    metrics_path = os.path.join(metrics_dir, f"{language}_metrics.parquet")
    
    if os.path.exists(metrics_path):
        try:
            df = pd.read_parquet(metrics_path)
            logger.info(f"Loaded metrics for {language}: {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error loading metrics for {language}: {e}")
            return None
    else:
        logger.warning(f"Metrics file not found for {language}: {metrics_path}")
        return None


def save_filtered_manifests(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Save filtered manifests per language in JSONL format.
    
    Args:
        df: DataFrame with filtering decisions
        output_dir: Output directory
    
    Returns:
        Dictionary mapping language to manifest path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    manifest_paths = {}
    
    # Save complete dataset
    complete_path = os.path.join(output_dir, 'complete_filtered_dataset.parquet')
    df.to_parquet(complete_path, compression='snappy', index=False)
    logger.info(f"Saved complete dataset to {complete_path}")
    
    if 'lang' in df.columns:
        # Save per-language manifests
        for language in df['lang'].unique():
            lang_df = df[df['lang'] == language]
            
            # Keep only KEEP samples for manifest
            kept_df = lang_df[lang_df['filter_decision'] == 'KEEP']
            
            # Save filtered manifest
            manifest_path = os.path.join(output_dir, f"{language}_filtered_manifest.jsonl")
            kept_df.to_json(manifest_path, orient='records', lines=True)
            manifest_paths[language] = manifest_path
            
            # Save rejected samples for review
            rejected_df = lang_df[lang_df['filter_decision'] != 'KEEP']
            if len(rejected_df) > 0:
                rejected_path = os.path.join(output_dir, f"{language}_rejected_samples.jsonl")
                rejected_df.to_json(rejected_path, orient='records', lines=True)
            
            retention_rate = len(kept_df) / len(lang_df) if len(lang_df) > 0 else 0
            logger.info(f"{language}: {len(kept_df)}/{len(lang_df)} samples kept "
                       f"({retention_rate:.1%})")
    
    return manifest_paths


def save_statistics(stats: Dict, output_path: str) -> None:
    """
    Save computed statistics to file.
    
    Args:
        stats: Statistics dictionary
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    
    logger.info(f"Saved statistics to {output_path}")


def load_statistics(input_path: str) -> Optional[Dict]:
    """
    Load statistics from file.
    
    Args:
        input_path: Input file path
    
    Returns:
        Statistics dictionary or None if loading fails
    """
    if not os.path.exists(input_path):
        logger.warning(f"Statistics file not found: {input_path}")
        return None
    
    try:
        with open(input_path, 'rb') as f:
            stats = pickle.load(f)
        logger.info(f"Loaded statistics from {input_path}")
        return stats
    except Exception as e:
        logger.error(f"Error loading statistics: {e}")
        return None


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics for the dataset.
    
    Args:
        df: DataFrame with filtering results
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_samples': int(len(df)),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Filter decisions
    if 'filter_decision' in df.columns:
        decision_counts = df['filter_decision'].value_counts().to_dict()
        # Convert numpy int64 to regular int for JSON serialization
        summary['decision_counts'] = {k: int(v) for k, v in decision_counts.items()}
        summary['retention_rate'] = float(decision_counts.get('KEEP', 0) / len(df))
    
    # Language statistics
    if 'lang' in df.columns:
        lang_stats = {}
        for lang in df['lang'].unique():
            lang_df = df[df['lang'] == lang]
            kept = int((lang_df['filter_decision'] == 'KEEP').sum())
            lang_stats[lang] = {
                'total': int(len(lang_df)),
                'kept': kept,
                'retention_rate': float(kept / len(lang_df) if len(lang_df) > 0 else 0)
            }
        summary['language_stats'] = lang_stats
    
    # Duration statistics
    if 'duration' in df.columns:
        durations = df['duration'].dropna()
        if len(durations) > 0:
            summary['duration_stats'] = {
                'total_hours': float(durations.sum() / 3600),
                'mean_seconds': float(durations.mean()),
                'median_seconds': float(durations.median()),
                'min_seconds': float(durations.min()),
                'max_seconds': float(durations.max())
            }
    
    # Quality metrics for kept samples
    if 'filter_decision' in df.columns:
        kept_df = df[df['filter_decision'] == 'KEEP']
        if len(kept_df) > 0:
            quality_metrics = {}
            numeric_cols = kept_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in ['snr', 'rms_energy', 'pitch_mean', 'quality_score']:
                    values = kept_df[col].dropna()
                    if len(values) > 0:
                        quality_metrics[col] = {
                            'mean': float(values.mean()),
                            'median': float(values.median()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
            
            summary['quality_metrics'] = quality_metrics
    
    return summary


def export_for_training(df: pd.DataFrame, output_dir: str, 
                       split_ratios: Dict[str, float] = {'train': 0.8, 'val': 0.1, 'test': 0.1}) -> Dict[str, str]:
    """
    Export filtered data in format suitable for training.
    
    Args:
        df: DataFrame with kept samples
        output_dir: Output directory
        split_ratios: Dictionary with split ratios
    
    Returns:
        Dictionary mapping split names to file paths
    """
    # Filter to kept samples only
    kept_df = df[df['filter_decision'] == 'KEEP'].copy()
    
    if len(kept_df) == 0:
        logger.warning("No samples kept for training export")
        return {}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create stratified splits by language
    split_paths = {}
    
    if 'lang' in kept_df.columns:
        # Split per language to maintain balance
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for lang in kept_df['lang'].unique():
            lang_df = kept_df[kept_df['lang'] == lang].sample(frac=1, random_state=42)
            n_samples = len(lang_df)
            
            if n_samples < 3:
                # Too few samples, put all in train
                train_dfs.append(lang_df)
                continue
            
            # Calculate split sizes
            n_train = int(n_samples * split_ratios['train'])
            n_val = int(n_samples * split_ratios['val'])
            n_test = n_samples - n_train - n_val
            
            # Split
            train_dfs.append(lang_df.iloc[:n_train])
            val_dfs.append(lang_df.iloc[n_train:n_train + n_val])
            test_dfs.append(lang_df.iloc[n_train + n_val:])
        
        # Combine splits
        splits = {
            'train': pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(),
            'val': pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(),
            'test': pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        }
    else:
        # Simple random split
        shuffled = kept_df.sample(frac=1, random_state=42)
        n_samples = len(shuffled)
        
        n_train = int(n_samples * split_ratios['train'])
        n_val = int(n_samples * split_ratios['val'])
        
        splits = {
            'train': shuffled.iloc[:n_train],
            'val': shuffled.iloc[n_train:n_train + n_val],
            'test': shuffled.iloc[n_train + n_val:]
        }
    
    # Save splits
    for split_name, split_df in splits.items():
        if len(split_df) > 0:
            split_path = os.path.join(output_dir, f"{split_name}_manifest.jsonl")
            split_df.to_json(split_path, orient='records', lines=True)
            split_paths[split_name] = split_path
            
            logger.info(f"Saved {split_name} split: {len(split_df)} samples to {split_path}")
    
    return split_paths


def backup_checkpoint(checkpoint_file: str, backup_dir: str) -> str:
    """
    Create backup of checkpoint file.
    
    Args:
        checkpoint_file: Path to checkpoint file
        backup_dir: Backup directory
    
    Returns:
        Path to backup file
    """
    if not os.path.exists(checkpoint_file):
        return ""
    
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"checkpoint_backup_{timestamp}.json"
    backup_path = os.path.join(backup_dir, backup_name)
    
    # Copy checkpoint file
    import shutil
    shutil.copy2(checkpoint_file, backup_path)
    
    logger.info(f"Backed up checkpoint to {backup_path}")
    return backup_path