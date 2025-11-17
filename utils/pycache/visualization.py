"""
Visualization and reporting utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_metric_distributions(df: pd.DataFrame, 
                              metrics_to_plot: List[str],
                              output_dir: str,
                              title_prefix: str = "Audio Quality") -> str:
    """
    Plot metric distributions before and after filtering.
    
    Args:
        df: DataFrame with samples and filter decisions
        metrics_to_plot: List of metrics to visualize
        output_dir: Output directory for plots
        title_prefix: Prefix for plot title
    
    Returns:
        Path to saved plot
    """
    # Calculate grid size
    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{metric}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric} (N/A)')
            continue
        
        # Get data
        all_data = df[metric].dropna()
        kept_data = df[df['filter_decision'] == 'KEEP'][metric].dropna()
        
        if len(all_data) == 0:
            ax.text(0.5, 0.5, f'{metric}\nNo Data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric} (No Data)')
            continue
        
        # Plot histograms
        bins = min(50, max(10, len(all_data) // 20))
        
        ax.hist(all_data, bins=bins, alpha=0.5, label='All', density=True, 
                color='lightblue', edgecolor='black')
        
        if len(kept_data) > 0:
            ax.hist(kept_data, bins=bins, alpha=0.7, label='Kept', density=True,
                    color='darkblue', edgecolor='black')
        
        # Add statistics
        ax.axvline(all_data.median(), color='lightblue', linestyle='--', 
                  label=f'All Median: {all_data.median():.2f}')
        
        if len(kept_data) > 0:
            ax.axvline(kept_data.median(), color='darkblue', linestyle='-', 
                      label=f'Kept Median: {kept_data.median():.2f}')
        
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(f'{metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metric distributions plot to {output_path}")
    return output_path


def plot_retention_by_language(df: pd.DataFrame, 
                              output_dir: str,
                              title: str = "Retention Rate by Language") -> str:
    """
    Plot retention rates by language.
    
    Args:
        df: DataFrame with samples and filter decisions
        output_dir: Output directory for plots
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    if 'lang' not in df.columns:
        logger.warning("No language column found, skipping language retention plot")
        return ""
    
    # Calculate retention rates
    retention_by_lang = df.groupby('lang').apply(
        lambda x: (x['filter_decision'] == 'KEEP').mean()
    ).sort_values()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(retention_by_lang)), retention_by_lang.values, 
                   color='steelblue', alpha=0.7)
    
    plt.xlabel('Language')
    plt.ylabel('Retention Rate')
    plt.title(title)
    plt.xticks(range(len(retention_by_lang)), retention_by_lang.index, 
               rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, retention_by_lang.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.1%}', ha='center', va='bottom')
    
    # Add average line
    avg_retention = (df['filter_decision'] == 'KEEP').mean()
    plt.axhline(avg_retention, color='red', linestyle='--', alpha=0.7,
               label=f'Overall: {avg_retention:.1%}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'retention_by_language.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved language retention plot to {output_path}")
    return output_path


def plot_rejection_reasons(df: pd.DataFrame, 
                          output_dir: str,
                          title: str = "Rejection Reasons Breakdown") -> str:
    """
    Plot breakdown of rejection reasons.
    
    Args:
        df: DataFrame with samples and filter decisions
        output_dir: Output directory for plots
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    # Count rejection types
    rejection_counts = df['filter_decision'].value_counts()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of overall decisions
    colors = ['green', 'orange', 'red', 'purple']
    wedges, texts, autotexts = ax1.pie(rejection_counts.values, 
                                       labels=rejection_counts.index,
                                       colors=colors[:len(rejection_counts)],
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Filtering Results')
    
    # Bar chart of rejection reasons
    rejection_only = rejection_counts[rejection_counts.index != 'KEEP']
    if len(rejection_only) > 0:
        bars = ax2.bar(range(len(rejection_only)), rejection_only.values,
                      color=colors[1:len(rejection_only)+1])
        ax2.set_xlabel('Rejection Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Breakdown of Rejections')
        ax2.set_xticks(range(len(rejection_only)))
        ax2.set_xticklabels(rejection_only.index, rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, rejection_only.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rejection_only.values) * 0.01,
                    f'{value}', ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No rejections\nto show', ha='center', va='center',
                transform=ax2.transAxes, fontsize=16)
        ax2.set_title('No Rejections')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'rejection_reasons.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved rejection reasons plot to {output_path}")
    return output_path


def plot_outlier_scores_distribution(df: pd.DataFrame,
                                    output_dir: str,
                                    title: str = "Outlier Score Distribution") -> str:
    """
    Plot distribution of outlier scores.
    
    Args:
        df: DataFrame with outlier scores
        output_dir: Output directory for plots
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    if 'outlier_score' not in df.columns:
        logger.warning("No outlier score column found")
        return ""
    
    plt.figure(figsize=(12, 8))
    
    # Plot distribution for each decision category
    decisions = df['filter_decision'].unique()
    colors = ['green', 'orange', 'red', 'purple']
    
    for i, decision in enumerate(decisions):
        subset = df[df['filter_decision'] == decision]['outlier_score']
        if len(subset) > 0:
            plt.hist(subset, bins=50, alpha=0.6, label=f'{decision} (n={len(subset)})',
                    color=colors[i % len(colors)], density=True)
    
    plt.xlabel('Outlier Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add threshold lines if available
    if hasattr(df, 'threshold_used'):
        plt.axvline(df.threshold_used, color='black', linestyle='--',
                   label=f'Threshold: {df.threshold_used:.2f}')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'outlier_scores_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved outlier scores plot to {output_path}")
    return output_path


def plot_quality_vs_duration(df: pd.DataFrame,
                            output_dir: str,
                            title: str = "Quality Score vs Duration") -> str:
    """
    Plot quality score vs duration scatter plot.
    
    Args:
        df: DataFrame with quality metrics
        output_dir: Output directory for plots
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    if 'duration' not in df.columns or 'quality_score' not in df.columns:
        logger.warning("Missing duration or quality_score columns")
        return ""
    
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot colored by decision
    decisions = df['filter_decision'].unique()
    colors = ['green', 'orange', 'red', 'purple']
    
    for i, decision in enumerate(decisions):
        subset = df[df['filter_decision'] == decision]
        if len(subset) > 0:
            plt.scatter(subset['duration'], subset['quality_score'],
                       alpha=0.6, label=f'{decision} (n={len(subset)})',
                       color=colors[i % len(colors)], s=20)
    
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Quality Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'quality_vs_duration.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved quality vs duration plot to {output_path}")
    return output_path


def generate_analysis_reports(df: pd.DataFrame, config: Dict, output_dir: str) -> None:
    """
    Generate comprehensive analysis reports and visualizations.
    
    Args:
        df: DataFrame with filtering results
        config: Configuration dictionary
        output_dir: Output directory for reports
    """
    # Create reports directory
    report_dir = os.path.join(output_dir, 'reports')
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating analysis reports in {report_dir}")
    
    # Define metrics to plot
    metrics_to_plot = [
        'snr', 'clipping_ratio', 'silence_ratio', 'rms_energy',
        'pitch_mean', 'spectral_centroid_mean', 'outlier_score'
    ]
    
    # Filter to existing metrics
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    try:
        # 1. Metric distributions
        plot_metric_distributions(df, available_metrics, report_dir)
        
        # 2. Language retention rates
        plot_retention_by_language(df, report_dir)
        
        # 3. Rejection reasons
        plot_rejection_reasons(df, report_dir)
        
        # 4. Outlier score distribution
        plot_outlier_scores_distribution(df, report_dir)
        
        # 5. Quality vs duration
        plot_quality_vs_duration(df, report_dir)
        
        # 6. Generate text summary
        generate_summary_report(df, config, report_dir)
        
        logger.info("Analysis reports generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating analysis reports: {e}")


def generate_summary_report(df: pd.DataFrame, config: Dict, output_dir: str) -> str:
    """
    Generate text summary report.
    
    Args:
        df: DataFrame with filtering results
        config: Configuration dictionary
        output_dir: Output directory
    
    Returns:
        Path to summary report
    """
    report_path = os.path.join(output_dir, 'summary_statistics.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AUDIO FILTERING PIPELINE - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        total_samples = len(df)
        kept_samples = (df['filter_decision'] == 'KEEP').sum()
        rejected_samples = total_samples - kept_samples
        retention_rate = kept_samples / total_samples if total_samples > 0 else 0
        
        f.write(f"Total samples processed: {total_samples:,}\n")
        f.write(f"Samples kept: {kept_samples:,}\n")
        f.write(f"Samples rejected: {rejected_samples:,}\n")
        f.write(f"Overall retention rate: {retention_rate:.2%}\n\n")
        
        # Rejection breakdown
        f.write("Rejection Breakdown:\n")
        f.write("-" * 30 + "\n")
        rejection_counts = df['filter_decision'].value_counts()
        for decision, count in rejection_counts.items():
            percentage = count / total_samples * 100
            f.write(f"  {decision}: {count:,} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Language-specific statistics
        if 'lang' in df.columns:
            f.write("Retention by Language:\n")
            f.write("-" * 30 + "\n")
            lang_stats = df.groupby('lang').agg({
                'filter_decision': lambda x: (x == 'KEEP').sum(),
                'lang': 'count'
            }).rename(columns={'filter_decision': 'kept', 'lang': 'total'})
            lang_stats['retention_rate'] = lang_stats['kept'] / lang_stats['total']
            lang_stats = lang_stats.sort_values('retention_rate')
            
            for lang, row in lang_stats.iterrows():
                f.write(f"  {lang}: {row['kept']:,}/{row['total']:,} "
                       f"({row['retention_rate']:.1%})\n")
            f.write("\n")
        
        # Quality metrics for kept samples
        kept_df = df[df['filter_decision'] == 'KEEP']
        if len(kept_df) > 0:
            f.write("Quality Metrics (Kept Samples Only):\n")
            f.write("-" * 40 + "\n")
            
            quality_metrics = ['snr', 'clipping_ratio', 'silence_ratio', 'rms_energy',
                             'pitch_mean', 'spectral_centroid_mean', 'quality_score']
            
            for metric in quality_metrics:
                if metric in kept_df.columns:
                    values = kept_df[metric].dropna()
                    if len(values) > 0:
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean: {values.mean():.4f}\n")
                        f.write(f"  Median: {values.median():.4f}\n")
                        f.write(f"  Std: {values.std():.4f}\n")
                        f.write(f"  Min: {values.min():.4f}\n")
                        f.write(f"  Max: {values.max():.4f}\n")
        
        # Duration statistics
        if 'duration' in df.columns:
            f.write("\n" + "Duration Statistics:\n")
            f.write("-" * 25 + "\n")
            
            all_durations = df['duration'].dropna()
            kept_durations = kept_df['duration'].dropna() if len(kept_df) > 0 else pd.Series()
            
            f.write(f"All samples - Total hours: {all_durations.sum() / 3600:.1f}\n")
            f.write(f"All samples - Mean duration: {all_durations.mean():.1f}s\n")
            f.write(f"All samples - Median duration: {all_durations.median():.1f}s\n")
            
            if len(kept_durations) > 0:
                f.write(f"Kept samples - Total hours: {kept_durations.sum() / 3600:.1f}\n")
                f.write(f"Kept samples - Mean duration: {kept_durations.mean():.1f}s\n")
                f.write(f"Kept samples - Median duration: {kept_durations.median():.1f}s\n")
        
        # Configuration used
        f.write("\n" + "Configuration Used:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Percentile threshold: {config.get('filtering', {}).get('percentile_threshold', 70)}%\n")
        
        hard_filters = config.get('filtering', {}).get('hard_filters', {})
        f.write(f"Hard filters:\n")
        for key, value in hard_filters.items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Summary report saved to {report_path}")
    return report_path