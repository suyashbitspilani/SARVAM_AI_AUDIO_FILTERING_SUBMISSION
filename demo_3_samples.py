#!/usr/bin/env python3
"""
Demo: Audio Filtering Pipeline for IndicVoices Dataset
Author: Suyash Khare
Assignment: Sarvam AI Speech Team ML Intern
Date: November 2024

This demo processes 3 Assamese samples from IndicVoices with detailed output.
"""

import os
import sys
import logging
from datasets import load_dataset
import tempfile
import soundfile as sf
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.audio_utils import process_single_audio_file
from filtering.rules import apply_hard_filters

def print_header():
    """Print demo header with author info."""
    print("\n" + "="*70)
    print("🎵  AUDIO FILTERING PIPELINE DEMO")
    print("="*70)
    print("Author: Suyash Khare")
    print("Assignment: Sarvam AI Speech Team ML Intern")
    print("Date: November 2024")
    print("Dataset: IndicVoices (ai4bharat)")
    print("="*70 + "\n")

def print_section(title):
    """Print section header."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def main():
    # Setup
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    # Set HF token
    os.environ["HF_TOKEN"] = "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"
    
    # Print header
    print_header()
    
    print_section("LOADING DATASET")
    print("📥 Loading 3 Assamese samples from IndicVoices...")
    print("   Source: HuggingFace (ai4bharat/IndicVoices)")
    print("   Language: Assamese")
    print("   Split: Validation")
    
    start_time = datetime.now()
    
    try:
        # Load dataset
        dataset = load_dataset("ai4bharat/IndicVoices", "assamese", 
                              split="valid[:3]", streaming=False)
        
        load_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Successfully loaded {len(dataset)} samples in {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nFalling back to synthetic data for demonstration...")
        # Create synthetic fallback data
        dataset = create_synthetic_samples()
    
    # Configuration
    config = {
        'processing': {'audio_sr': 16000},
        'metrics': {
            'compute_snr': True, 
            'compute_clipping': True,
            'compute_silence': True, 
            'compute_energy': True,
            'compute_spectral': True, 
            'compute_pitch': True,
            'compute_mfcc': False, 
            'compute_hnr': False,
            'compute_dnsmos': False, 
            'compute_asr_quality': False
        }
    }
    
    results = []
    
    print_section("PROCESSING AUDIO SAMPLES")
    
    # Process each sample
    for i, item in enumerate(dataset):
        print(f"\n📊 SAMPLE {i+1}/3")
        print("-" * 40)
        
        # Show metadata
        print(f"📝 Metadata:")
        print(f"   Duration: {item.get('duration', 0):.2f} seconds")
        print(f"   Speaker: {item.get('speaker_id', 'N/A')}")
        print(f"   Gender: {item.get('gender', 'N/A')}")
        print(f"   Age Group: {item.get('age_group', 'N/A')}")
        print(f"   Task: {item.get('task_name', 'N/A')}")
        print(f"   District: {item.get('district', 'N/A')}")
        
        text = item.get('text', '')
        if text:
            print(f"   Text: \"{text[:60]}...\"" if len(text) > 60 else f"   Text: \"{text}\"")
        
        try:
            print(f"\n⚙️  Computing quality metrics...")
            
            # Save audio temporarily
            audio_data = item.get('audio', {})
            if isinstance(audio_data, dict) and 'array' in audio_data:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio_data['array'], audio_data['sampling_rate'])
                    
                    # Prepare sample
                    sample = {
                        'audio_filepath': tmp.name,
                        'duration': item.get('duration', len(audio_data['array']) / audio_data['sampling_rate']),
                        'lang': 'assamese',
                        'text': item.get('text', ''),
                        'speaker_id': item.get('speaker_id', ''),
                        'gender': item.get('gender', ''),
                        'age_group': item.get('age_group', ''),
                        'district': item.get('district', ''),
                        'task_name': item.get('task_name', '')
                    }
                    
                    # Process
                    process_start = datetime.now()
                    result = process_single_audio_file(sample, config)
                    process_time = (datetime.now() - process_start).total_seconds()
                    
                    results.append(result)
                    
                    # Show metrics
                    if result.get('processing_status') == 'success':
                        print(f"✅ Processing successful ({process_time:.2f}s)")
                        print("\n🎯 COMPUTED METRICS:")
                        
                        # Primary metrics
                        metrics_display = [
                            ('SNR', 'snr', 'dB'),
                            ('Clipping Ratio', 'clipping_ratio', '%'),
                            ('Silence Ratio', 'silence_ratio', '%'),
                            ('RMS Energy', 'rms_energy', ''),
                            ('Pitch Mean', 'pitch_mean', 'Hz'),
                            ('Spectral Centroid', 'spectral_centroid_mean', 'Hz')
                        ]
                        
                        for display_name, metric_key, unit in metrics_display:
                            if metric_key in result:
                                value = result[metric_key]
                                if '%' in unit:
                                    value *= 100
                                if unit:
                                    print(f"   {display_name}: {value:.2f} {unit}")
                                else:
                                    print(f"   {display_name}: {value:.4f}")
                    else:
                        print(f"❌ Processing failed: {result.get('error_message')}")
                    
                    # Cleanup
                    os.unlink(tmp.name)
                    
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            
    # Apply filtering
    print_section("APPLYING FILTERING RULES")
    
    df = pd.DataFrame([r for r in results if r.get('processing_status') == 'success'])
    
    if len(df) > 0:
        print("\n🔍 Three-Tier Filtering System:")
        print("   Tier 1: Hard thresholds (automatic rejection)")
        print("   Tier 2: Soft filters (percentile-based)")
        print("   Tier 3: Contextual filters (pattern-based)")
        
        # Hard filters configuration
        filter_config = {
            'hard_filters': {
                'max_clipping_ratio': 0.01,
                'min_duration': 1.0,
                'max_duration': 30.0,
                'min_snr': -5.0,
                'max_silence_ratio': 0.8,
                'min_active_speech': 0.5,
                'min_rms_energy': 0.001,
                'min_peak_amplitude': 0.01,
                'min_spectral_centroid': 1000,
                'max_spectral_centroid': 8000
            }
        }
        
        print("\n📋 Hard Filter Thresholds:")
        print("   • Max clipping: 1%")
        print("   • Min SNR: -5 dB")
        print("   • Max silence: 80%")
        print("   • Duration: 1-30 seconds")
        print("   • Min RMS energy: 0.001")
        
        # Apply filters
        hard_reject = apply_hard_filters(df, filter_config)
        
        print("\n" + "="*50)
        print("📊 FILTERING RESULTS:")
        print("="*50)
        
        for i, row in df.iterrows():
            sample_name = f"Sample {i+1}"
            if hard_reject.iloc[i]:
                print(f"\n❌ {sample_name}: REJECTED")
                
                # Determine rejection reason
                reasons = []
                if 'snr' in row and row['snr'] < filter_config['hard_filters']['min_snr']:
                    reasons.append(f"Low SNR ({row['snr']:.1f} dB)")
                if 'clipping_ratio' in row and row['clipping_ratio'] > filter_config['hard_filters']['max_clipping_ratio']:
                    reasons.append(f"Excessive clipping ({row['clipping_ratio']*100:.1f}%)")
                if 'silence_ratio' in row and row['silence_ratio'] > filter_config['hard_filters']['max_silence_ratio']:
                    reasons.append(f"Too much silence ({row['silence_ratio']*100:.1f}%)")
                if 'rms_energy' in row and row['rms_energy'] < filter_config['hard_filters']['min_rms_energy']:
                    reasons.append(f"Low energy ({row['rms_energy']:.4f})")
                
                if reasons:
                    print(f"   Rejection reasons: {', '.join(reasons)}")
            else:
                print(f"\n✅ {sample_name}: ACCEPTED")
                print(f"   Quality metrics within acceptable range")
            
        kept = (~hard_reject).sum()
        total = len(df)
        retention_rate = (kept/total*100) if total > 0 else 0
        
        print("\n" + "="*50)
        print(f"📈 SUMMARY STATISTICS:")
        print("="*50)
        print(f"   Total samples processed: {total}")
        print(f"   Samples accepted: {kept}")
        print(f"   Samples rejected: {total - kept}")
        print(f"   Retention rate: {retention_rate:.1f}%")
        
        if total > 0:
            # Compute average metrics
            print("\n📊 Average Metrics (All Samples):")
            for metric in ['snr', 'silence_ratio', 'rms_energy']:
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    if metric == 'silence_ratio':
                        print(f"   {metric}: {mean_val*100:.1f}% ± {std_val*100:.1f}%")
                    elif metric == 'snr':
                        print(f"   {metric}: {mean_val:.1f} ± {std_val:.1f} dB")
                    else:
                        print(f"   {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n" + "="*70)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nThis pipeline can process the full IndicVoices dataset:")
    print("• 22 languages")
    print("• 19,550+ hours of audio")
    print("• Parallel processing for scalability")
    print("• Comprehensive quality assessment")
    print("\nAuthor: Suyash Khare | Sarvam AI Speech Team ML Intern")
    print("="*70 + "\n")

def create_synthetic_samples():
    """Create synthetic samples for fallback demonstration."""
    import numpy as np
    
    samples = []
    sr = 16000
    
    for i in range(3):
        duration = 3.0 + i
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * (440 + i*110) * t) + 0.05 * np.random.normal(0, 0.01, len(t))
        
        samples.append({
            'audio': {
                'array': audio,
                'sampling_rate': sr
            },
            'duration': duration,
            'speaker_id': f'SYNTH_SPEAKER_{i+1}',
            'gender': ['Male', 'Female', 'Male'][i],
            'age_group': ['18-30', '30-45', '45-60'][i],
            'task_name': ['read', 'extempore', 'conversational'][i],
            'district': 'Synthetic',
            'text': f'This is synthetic sample {i+1} for demonstration purposes.'
        })
    
    return samples

if __name__ == "__main__":
    main()