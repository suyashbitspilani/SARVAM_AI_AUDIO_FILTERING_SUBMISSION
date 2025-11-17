#!/usr/bin/env python3
"""Audio filtering pipeline for IndicVoices dataset."""

import argparse
import logging
import os
import sys
import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.audio_utils import process_single_audio_file
from utils.data_utils import save_metrics, load_metrics, save_filtered_manifests, create_summary_statistics
from utils.visualization import generate_analysis_reports
from outliers.statistical import detect_outliers_zscore, detect_outliers_iqr, compute_stratum_statistics
from outliers.ensemble import ensemble_outlier_score
from filtering.rules import make_filtering_decision


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('audio_filtering.log')
        ]
    )


def load_dataset_streaming(dataset_name: str, language: str, split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
    try:
        from datasets import load_dataset
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading {dataset_name} dataset for {language}")
        
        # Load in streaming mode
        dataset = load_dataset(dataset_name, language, split=split, streaming=True)
        
        # Convert to list (this will download the data)
        records = []
        for idx, item in enumerate(tqdm(dataset, desc=f"Loading {language}")):
            if max_samples and idx >= max_samples:
                logger.info(f"Reached max_samples limit of {max_samples}")
                break
                
            record = {
                'audio_filepath': item.get('audio_filepath', ''),
                'duration': item.get('duration', 0.0),
                'lang': language,
                'text': item.get('text', ''),
                'verbatim': item.get('verbatim', ''),
                'speaker_id': item.get('speaker_id', ''),
                'gender': item.get('gender', ''),
                'age_group': item.get('age_group', ''),
                'district': item.get('district', ''),
                'state': item.get('state', ''),
                'scenario': item.get('scenario', ''),
                'task_name': item.get('task_name', ''),
                'occupation': item.get('occupation', '')
            }
            records.append(record)
        
        logger.info(f"Loaded {len(records)} samples for {language}")
        return records
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading dataset for {language}: {e}")
        return []


def process_with_checkpointing(language: str, config: Dict) -> List[Dict]:
    import logging
    logger = logging.getLogger(__name__)
    
    checkpoint_dir = config['paths']['checkpoint_dir']
    batch_size = config['processing'].get('batch_size', 1000)
    dataset_name = config['dataset']['name']
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, f"{language}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint['last_processed_idx']
                all_results = checkpoint['results']
            logger.info(f"Resuming {language} from index {start_idx}")
        except Exception as e:
            logger.error(f"Error loading checkpoint for {language}: {e}")
            start_idx = 0
            all_results = []
    else:
        start_idx = 0
        all_results = []
    
    # Load dataset
    if start_idx == 0:
        max_samples = config['processing'].get('max_samples_per_language', None)
        records = load_dataset_streaming(dataset_name, language, max_samples=max_samples)
        if not records:
            logger.warning(f"No records loaded for {language}")
            return []
    else:
        # Load from checkpoint
        records = checkpoint.get('records', [])
        if not records:
            logger.error(f"No records in checkpoint for {language}")
            return []
    
    # Process in batches
    n_workers = config['processing'].get('n_workers', mp.cpu_count() - 1)
    
    for batch_start in range(start_idx, len(records), batch_size):
        batch_end = min(batch_start + batch_size, len(records))
        batch = records[batch_start:batch_end]
        
        logger.info(f"Processing {language} batch {batch_start}-{batch_end}")
        
        # Process batch in parallel
        with mp.Pool(processes=n_workers) as pool:
            process_fn = partial(process_single_audio_file, config=config)
            batch_results = list(tqdm(
                pool.imap(process_fn, batch, chunksize=10),
                total=len(batch),
                desc=f"Processing {language} batch"
            ))
        
        all_results.extend(batch_results)
        
        # Save checkpoint
        checkpoint = {
            'last_processed_idx': batch_end,
            'results': all_results,
            'language': language,
            'total_records': len(records)
        }
        
        # Only save records in first checkpoint to save space
        if start_idx == 0:
            checkpoint['records'] = records
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved for {language} at index {batch_end}")
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info(f"Checkpoint file removed for {language}")
    
    return all_results


def main(config_path: str) -> None:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting audio filtering pipeline")
    
    for dir_key in ['checkpoint_dir', 'metrics_dir', 'output_dir']:
        Path(config['paths'][dir_key]).mkdir(parents=True, exist_ok=True)
    
    languages = config['dataset'].get('languages', [
        "hindi", "bengali", "tamil", "telugu", "marathi", 
        "gujarati", "kannada", "malayalam", "punjabi", "odia"
    ])
    
    logger.info("=" * 80)
    logger.info("STEP 1: Computing Audio Quality Metrics")
    logger.info("=" * 80)
    
    for language in languages:
        logger.info(f"Processing language: {language}")
        
            existing_metrics = load_metrics(language, config['paths']['metrics_dir'])
        if existing_metrics is not None:
            logger.info(f"Metrics already computed for {language}, skipping...")
            continue
        
            results = process_with_checkpointing(language, config)
        
        if results:
                    save_metrics(results, language, config['paths']['metrics_dir'])
        else:
            logger.warning(f"No results for {language}")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading and Combining Metrics")
    logger.info("=" * 80)
    
    all_metrics = []
    for language in languages:
        df = load_metrics(language, config['paths']['metrics_dir'])
        if df is not None:
            all_metrics.append(df)
        else:
            logger.warning(f"Could not load metrics for {language}")
    
    if not all_metrics:
        logger.error("No metrics loaded. Exiting.")
        return
    
    combined_df = pd.concat(all_metrics, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} samples")
    
    successful_df = combined_df[combined_df['processing_status'] == 'success'].copy()
    logger.info(f"Successfully processed: {len(successful_df)} samples")
    
    if len(successful_df) == 0:
        logger.error("No successfully processed samples. Exiting.")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Computing Stratum Statistics")
    logger.info("=" * 80)
    
    metric_cols = [
        'snr', 'clipping_ratio', 'silence_ratio', 'rms_energy',
        'pitch_mean', 'spectral_centroid_mean', 'spectral_flatness'
    ]
    
    available_metrics = [col for col in metric_cols if col in successful_df.columns]
    logger.info(f"Available metrics: {available_metrics}")
    
    stratum_cols = config['stratification']['primary']
    stratum_stats = compute_stratum_statistics(successful_df, available_metrics, stratum_cols)
    
    stats_path = os.path.join(config['paths']['output_dir'], 'stratum_statistics.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stratum_stats, f)
    logger.info(f"Saved stratum statistics to {stats_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Detecting Outliers")
    logger.info("=" * 80)
    
    logger.info("Running modified Z-score outlier detection...")
    zscore_threshold = config['outlier_detection']['modified_zscore_threshold']
    zscore_flags = detect_outliers_zscore(
        successful_df, available_metrics, stratum_cols, zscore_threshold
    )
    
    logger.info("Running IQR outlier detection...")
    iqr_multiplier = config['outlier_detection']['iqr_multiplier']
    iqr_flags, bounds = detect_outliers_iqr(
        successful_df, available_metrics, stratum_cols, iqr_multiplier
    )
    
    logger.info("Computing ensemble outlier scores...")
    outlier_flags_dict = {
        'zscore': zscore_flags,
        'iqr': iqr_flags
    }
    
    ensemble_weights = config['outlier_detection']['ensemble_weights']
    outlier_scores = ensemble_outlier_score(
        successful_df, 
        outlier_flags_dict,
        weights=ensemble_weights
    )
    
    logger.info(f"Outlier score range: {outlier_scores.min():.3f} - {outlier_scores.max():.3f}")
    logger.info(f"Mean outlier score: {outlier_scores.mean():.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Making Filtering Decisions")
    logger.info("=" * 80)
    
    filtered_df = make_filtering_decision(successful_df, outlier_scores, config)
    
    logger.info("\nFiltering Summary:")
    decision_counts = filtered_df['filter_decision'].value_counts()
    for decision, count in decision_counts.items():
        percentage = count / len(filtered_df) * 100
        logger.info(f"  {decision}: {count:,} ({percentage:.1f}%)")
    
    retention_rate = (filtered_df['filter_decision'] == 'KEEP').mean()
    logger.info(f"\nOverall retention rate: {retention_rate:.2%}")
    
    if 'lang' in filtered_df.columns:
        logger.info("\nPer-Language Retention Rates:")
        retention_by_lang = filtered_df.groupby('lang').apply(
            lambda x: (x['filter_decision'] == 'KEEP').mean()
        ).sort_values()
        
        for lang, rate in retention_by_lang.items():
            kept = (filtered_df[filtered_df['lang'] == lang]['filter_decision'] == 'KEEP').sum()
            total = len(filtered_df[filtered_df['lang'] == lang])
            logger.info(f"  {lang}: {rate:.1%} ({kept:,}/{total:,})")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Saving Filtered Manifests")
    logger.info("=" * 80)
    
    manifest_paths = save_filtered_manifests(filtered_df, config['paths']['output_dir'])
    
    for lang, path in manifest_paths.items():
        logger.info(f"Saved manifest for {lang}: {path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Generating Analysis Reports")
    logger.info("=" * 80)
    
    if config.get('reporting', {}).get('generate_plots', True):
        generate_analysis_reports(filtered_df, config, config['paths']['output_dir'])
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Creating Final Summary")
    logger.info("=" * 80)
    
    summary = create_summary_statistics(filtered_df)
    summary_path = os.path.join(config['paths']['output_dir'], 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved pipeline summary to {summary_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Total samples processed: {len(filtered_df):,}")
    logger.info(f"Samples kept: {(filtered_df['filter_decision'] == 'KEEP').sum():,}")
    logger.info(f"Final retention rate: {retention_rate:.2%}")
    logger.info(f"Output directory: {config['paths']['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Audio Filtering Pipeline for IndicVoices Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py --config config.yaml
  
The pipeline will:
1. Load the IndicVoices dataset for specified languages
2. Compute audio quality metrics for each sample
3. Detect statistical outliers using multiple methods
4. Apply filtering rules to remove low-quality samples
5. Generate analysis reports and filtered manifests

Output files will be saved to the directory specified in config.yaml.
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--languages',
        type=str,
        nargs='*',
        help='Override languages to process (space-separated)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without processing'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Override languages if specified
    if args.languages:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['dataset']['languages'] = args.languages
        
        # Save temporary config
        temp_config = args.config.replace('.yaml', '_temp.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        args.config = temp_config
    
    if args.dry_run:
        print("Dry run mode - validating configuration...")
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            print("Configuration is valid!")
            print(f"Languages to process: {config['dataset']['languages']}")
            print(f"Output directory: {config['paths']['output_dir']}")
        except Exception as e:
            print(f"Configuration error: {e}")
            sys.exit(1)
    else:
        main(args.config)