"""
Audio processing utility functions.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Optional, Tuple, Union
import logging
import os
from pathlib import Path

from metrics.signal_metrics import (
    compute_snr, detect_clipping, compute_silence_metrics,
    compute_energy_metrics, compute_speaking_rate
)
from metrics.spectral_metrics import (
    compute_spectral_features, compute_spectral_flatness,
    compute_pitch_metrics, compute_mfcc_features
)
from metrics.perceptual_metrics import (
    compute_hnr, compute_dnsmos, verify_language, asr_quality_check
)

logger = logging.getLogger(__name__)


def load_audio_safe(file_path: str, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    """
    Safely load audio file with error handling.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate for resampling
    
    Returns:
        Tuple of (audio array, sample rate, error message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return None, None, f"File not found: {file_path}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None, None, "File is empty"
        
        # Load audio using librosa (handles multiple formats)
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        except Exception as e:
            # Fallback to soundfile
            try:
                audio, sr = sf.read(file_path)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)  # Convert to mono
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
            except Exception as e2:
                return None, None, f"Failed to load audio: {e2}"
        
        # Validate audio data
        if len(audio) == 0:
            return None, None, "Audio contains no samples"
        
        # Check for invalid values
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            logger.warning(f"Fixed infinite/NaN values in {file_path}")
        
        # Normalize to prevent clipping issues
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio, sr, "success"
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None, str(e)


def validate_audio_file(file_path: str, min_duration: float = 0.5, 
                       max_duration: float = 600.0) -> Tuple[bool, str]:
    """
    Validate audio file before processing.
    
    Args:
        file_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file extension
        valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff'}
        if Path(file_path).suffix.lower() not in valid_extensions:
            return False, "Unsupported audio format"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Empty file"
        
        # Check if file is too large (> 100 MB)
        if file_size > 100 * 1024 * 1024:
            return False, "File too large"
        
        # Load audio to check duration and content
        audio, sr, error = load_audio_safe(file_path)
        if audio is None:
            return False, error
        
        duration = len(audio) / sr
        
        # Check duration bounds
        if duration < min_duration:
            return False, f"Duration too short ({duration:.1f}s)"
        
        if duration > max_duration:
            return False, f"Duration too long ({duration:.1f}s)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio signal array
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal.
    
    Args:
        audio: Audio signal array
        method: Normalization method ('peak', 'rms', 'lufs')
    
    Returns:
        Normalized audio array
    """
    if method == 'peak':
        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        else:
            return audio
            
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 0.1  # Target RMS level
        if rms > 0:
            return audio * (target_rms / rms)
        else:
            return audio
            
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_all_metrics(audio: np.ndarray, sr: int, audio_path: str, 
                       config: Dict, metadata: Dict = None) -> Dict:
    """
    Compute all audio quality metrics for a single file.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        audio_path: Path to audio file (for some metrics)
        config: Configuration dictionary
        metadata: Additional metadata
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Basic audio properties
    metrics['duration'] = len(audio) / sr
    metrics['sample_rate'] = sr
    metrics['n_samples'] = len(audio)
    
    # Add metadata if provided
    if metadata:
        metrics.update(metadata)
    
    try:
        # Signal quality metrics
        if config.get('metrics', {}).get('compute_snr', True):
            metrics['snr'] = compute_snr(audio, sr)
        
        if config.get('metrics', {}).get('compute_clipping', True):
            clipping_metrics = detect_clipping(audio)
            metrics.update(clipping_metrics)
        
        if config.get('metrics', {}).get('compute_silence', True):
            silence_metrics = compute_silence_metrics(audio, sr)
            metrics.update(silence_metrics)
        
        if config.get('metrics', {}).get('compute_energy', True):
            energy_metrics = compute_energy_metrics(audio)
            metrics.update(energy_metrics)
        
        if config.get('metrics', {}).get('compute_speaking_rate', True):
            rate_metrics = compute_speaking_rate(audio, sr)
            metrics.update(rate_metrics)
        
        # Spectral metrics
        if config.get('metrics', {}).get('compute_spectral', True):
            spectral_metrics = compute_spectral_features(audio, sr)
            metrics.update(spectral_metrics)
            
            metrics['spectral_flatness'] = compute_spectral_flatness(audio, sr)
        
        if config.get('metrics', {}).get('compute_pitch', True):
            pitch_metrics = compute_pitch_metrics(audio, sr)
            if pitch_metrics:
                metrics.update(pitch_metrics)
        
        if config.get('metrics', {}).get('compute_mfcc', False):
            mfcc_metrics = compute_mfcc_features(audio, sr)
            metrics.update(mfcc_metrics)
        
        # Perceptual metrics (optional, may require additional libraries)
        if config.get('metrics', {}).get('compute_hnr', False):
            hnr = compute_hnr(audio_path)
            if hnr is not None:
                metrics['hnr'] = hnr
        
        if config.get('metrics', {}).get('compute_dnsmos', False):
            dnsmos_metrics = compute_dnsmos(audio_path)
            metrics.update(dnsmos_metrics)
        
        # Language verification
        if config.get('metrics', {}).get('compute_lang_id', False):
            expected_lang = metadata.get('lang', 'unknown') if metadata else 'unknown'
            lang_metrics = verify_language(audio_path, expected_lang)
            metrics.update(lang_metrics)
        
        # ASR quality check (expensive)
        if config.get('metrics', {}).get('compute_asr_quality', False):
            ground_truth = metadata.get('text', None) if metadata else None
            asr_metrics = asr_quality_check(audio_path, ground_truth)
            metrics.update(asr_metrics)
        
        metrics['processing_status'] = 'success'
        
    except Exception as e:
        logger.error(f"Error computing metrics for {audio_path}: {e}")
        metrics['processing_status'] = 'failed'
        metrics['processing_error'] = str(e)
    
    return metrics


def process_single_audio_file(audio_record: Dict, config: Dict) -> Dict:
    """
    Process a single audio file and compute all metrics.
    
    Args:
        audio_record: Dictionary with audio_filepath and metadata
        config: Configuration dictionary
    
    Returns:
        Dictionary with computed metrics and metadata
    """
    try:
        audio_path = audio_record['audio_filepath']
        
        # Validate file first
        is_valid, validation_error = validate_audio_file(audio_path)
        if not is_valid:
            return {
                **audio_record,
                'processing_status': 'failed',
                'processing_error': f"Validation failed: {validation_error}"
            }
        
        # Load audio
        target_sr = config.get('processing', {}).get('audio_sr', 16000)
        audio, sr, load_error = load_audio_safe(audio_path, target_sr)
        
        if audio is None:
            return {
                **audio_record,
                'processing_status': 'failed',
                'processing_error': f"Loading failed: {load_error}"
            }
        
        # Compute metrics
        metrics = compute_all_metrics(audio, sr, audio_path, config, audio_record)
        
        # Combine with original record
        result = {**audio_record, **metrics}
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {audio_record.get('audio_filepath', 'unknown')}: {e}")
        return {
            **audio_record,
            'processing_status': 'failed',
            'processing_error': str(e)
        }


def estimate_processing_time(file_path: str) -> float:
    """
    Estimate processing time for a single file.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Estimated processing time in seconds
    """
    try:
        # Get file duration
        duration = librosa.get_duration(filename=file_path)
        
        # Rough estimate: 0.1-0.5 seconds per second of audio
        # depending on metrics computed
        processing_time = duration * 0.3
        
        # Add overhead for file I/O
        processing_time += 0.5
        
        return processing_time
        
    except:
        # Default estimate
        return 2.0