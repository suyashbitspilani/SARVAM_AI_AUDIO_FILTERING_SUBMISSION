"""
Spectral and frequency domain metrics for audio filtering.
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple


def compute_pitch_metrics(audio: np.ndarray, sr: int,
                         fmin: float = 65, fmax: float = 2093) -> Optional[Dict[str, float]]:
    """
    Compute fundamental frequency (F0) statistics.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        fmin: Minimum frequency in Hz (default: C2 ~65 Hz)
        fmax: Maximum frequency in Hz (default: C7 ~2093 Hz)
    
    Returns:
        Dictionary with pitch metrics or None if no voiced speech
    """
    # Use PYIN algorithm for robust pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=fmin,
        fmax=fmax,
        sr=sr
    )
    
    # Filter out unvoiced frames
    f0_voiced = f0[voiced_flag]
    
    if len(f0_voiced) > 0:
        # Remove NaN values
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        
        if len(f0_voiced) > 0:
            # Compute pitch statistics
            pitch_mean = np.mean(f0_voiced)
            pitch_median = np.median(f0_voiced)
            pitch_std = np.std(f0_voiced)
            pitch_min = np.min(f0_voiced)
            pitch_max = np.max(f0_voiced)
            pitch_range = pitch_max - pitch_min
            
            # Coefficient of variation (normalized variability)
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0.0
            
            # Voiced ratio
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
            
            # Pitch confidence
            mean_confidence = np.mean(voiced_probs[voiced_flag]) if np.any(voiced_flag) else 0.0
            
            return {
                'pitch_mean': float(pitch_mean),
                'pitch_median': float(pitch_median),
                'pitch_std': float(pitch_std),
                'pitch_min': float(pitch_min),
                'pitch_max': float(pitch_max),
                'pitch_range': float(pitch_range),
                'pitch_cv': float(pitch_cv),
                'voiced_ratio': float(voiced_ratio),
                'pitch_confidence': float(mean_confidence)
            }
    
    # No voiced speech detected
    return {
        'pitch_mean': 0.0,
        'pitch_median': 0.0,
        'pitch_std': 0.0,
        'pitch_min': 0.0,
        'pitch_max': 0.0,
        'pitch_range': 0.0,
        'pitch_cv': 0.0,
        'voiced_ratio': 0.0,
        'pitch_confidence': 0.0
    }


def compute_spectral_features(audio: np.ndarray, sr: int,
                             n_fft: int = 2048,
                             hop_length: Optional[int] = None) -> Dict[str, float]:
    """
    Compute spectral characteristics.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        Dictionary with spectral features
    """
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Compute spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                          n_fft=n_fft, 
                                                          hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr,
                                                           n_fft=n_fft,
                                                           hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        roll_percent=0.85)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length)
    
    # Compute statistics
    features = {
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'spectral_centroid_std': float(np.std(spectral_centroid)),
        'spectral_centroid_min': float(np.min(spectral_centroid)),
        'spectral_centroid_max': float(np.max(spectral_centroid)),
        
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
        
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_rolloff_std': float(np.std(spectral_rolloff)),
        
        'spectral_contrast_mean': float(np.mean(spectral_contrast)),
        'spectral_contrast_std': float(np.std(spectral_contrast))
    }
    
    # Add spectral slope (trend in spectrum)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Compute spectral slope per frame
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    slopes = []
    
    for frame in magnitude.T:
        if np.sum(frame) > 0:
            # Linear regression for spectral slope
            coeffs = np.polyfit(freqs, frame, 1)
            slopes.append(coeffs[0])
    
    if slopes:
        features['spectral_slope_mean'] = float(np.mean(slopes))
        features['spectral_slope_std'] = float(np.std(slopes))
    else:
        features['spectral_slope_mean'] = 0.0
        features['spectral_slope_std'] = 0.0
    
    return features


def compute_spectral_flatness(audio: np.ndarray, sr: int = None) -> float:
    """
    Compute spectral flatness (measure of noise-likeness).
    
    Args:
        audio: Audio signal array
        sr: Sample rate (optional, not used but kept for consistency)
    
    Returns:
        Mean spectral flatness value
    """
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    
    # Return mean flatness across all frames
    return float(np.mean(flatness))


def compute_mfcc_features(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, float]:
    """
    Compute MFCC (Mel-frequency cepstral coefficients) features.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        Dictionary with MFCC statistics
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    features = {}
    
    # Statistics for each MFCC coefficient
    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
    
    # Delta and delta-delta MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Add delta statistics for first few coefficients
    for i in range(min(5, n_mfcc)):
        features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
        features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
    
    return features


def compute_formants(audio: np.ndarray, sr: int, n_formants: int = 3) -> Dict[str, float]:
    """
    Estimate formant frequencies using LPC analysis.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_formants: Number of formants to extract
    
    Returns:
        Dictionary with formant frequencies
    """
    # Frame the signal
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms
    
    frames = librosa.util.frame(audio, frame_length=frame_length, 
                                hop_length=hop_length)
    
    formants = {f'f{i+1}_mean': 0.0 for i in range(n_formants)}
    formants.update({f'f{i+1}_std': 0.0 for i in range(n_formants)})
    
    formant_values = [[] for _ in range(n_formants)]
    
    for frame in frames.T:
        # Apply window
        windowed = frame * np.hanning(len(frame))
        
        # LPC analysis
        lpc_order = 2 * n_formants + 2
        
        try:
            # Compute LPC coefficients
            a = librosa.lpc(windowed, order=lpc_order)
            
            # Find roots
            roots = np.roots(a)
            
            # Convert to frequencies
            angles = np.angle(roots)
            freqs = sorted(angles * (sr / (2 * np.pi)))
            
            # Keep only positive frequencies
            freqs = [f for f in freqs if f > 0 and f < sr/2]
            
            # Extract formants
            for i, f in enumerate(freqs[:n_formants]):
                if i < n_formants:
                    formant_values[i].append(f)
        except:
            # LPC failed for this frame
            continue
    
    # Compute statistics
    for i in range(n_formants):
        if formant_values[i]:
            formants[f'f{i+1}_mean'] = float(np.mean(formant_values[i]))
            formants[f'f{i+1}_std'] = float(np.std(formant_values[i]))
    
    return formants