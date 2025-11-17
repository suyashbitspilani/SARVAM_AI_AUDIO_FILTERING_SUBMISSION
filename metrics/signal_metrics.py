"""Audio signal quality metrics."""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict, Optional, Tuple


def compute_snr(audio: np.ndarray, sr: int, method: str = 'energy') -> float:
    if method == 'energy':
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                    hop_length=hop_length)[0]
        
        energy_db = librosa.power_to_db(energy**2, ref=np.max)
        
        vad_threshold = np.percentile(energy_db, 30)  # Bottom 30% as noise
        
        noise_frames = energy_db[energy_db <= vad_threshold]
        speech_frames = energy_db[energy_db > vad_threshold]
        
        if len(noise_frames) > 0 and len(speech_frames) > 0:
            noise_power = np.mean(10**(noise_frames/10))
            speech_power = np.mean(10**(speech_frames/10))
            
            if noise_power > 0:
                snr = 10 * np.log10(speech_power / noise_power)
            else:
                snr = 40.0  # Max SNR if no noise detected
        else:
            snr = 0.0  # Unable to compute SNR
            
    elif method == 'wada':
        snr = compute_snr(audio, sr, method='energy')
    else:
        raise ValueError(f"Unknown SNR method: {method}")
    
    return float(snr)


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> Dict[str, float]:
    max_val = threshold
    
    clipping_ratio = np.sum(np.abs(audio) >= max_val) / len(audio)
    peak_amplitude = np.max(np.abs(audio))
    
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        crest_factor = peak_amplitude / rms
    else:
        crest_factor = 0.0
    
    clipped = np.abs(audio) >= max_val
    consecutive_clips = []
    
    if np.any(clipped):
        diff = np.diff(np.concatenate(([0], clipped.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        consecutive_clips = ends - starts
        max_consecutive = np.max(consecutive_clips) if len(consecutive_clips) > 0 else 0
    else:
        max_consecutive = 0
    
    return {
        'clipping_ratio': float(clipping_ratio),
        'peak_amplitude': float(peak_amplitude),
        'crest_factor': float(crest_factor),
        'max_consecutive_clips': int(max_consecutive),
        'num_clipping_events': len(consecutive_clips)
    }


def compute_silence_metrics(audio: np.ndarray, sr: int, 
                           silence_threshold: float = -40,
                           frame_length_ms: int = 25,
                           hop_length_ms: int = 10) -> Dict[str, float]:
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                hop_length=hop_length)[0]
    
    energy_db = librosa.power_to_db(energy**2, ref=np.max)
    
    voice_frames = energy_db > silence_threshold
    
    total_frames = len(voice_frames)
    if total_frames == 0:
        return {
            'leading_silence': 0.0,
            'trailing_silence': 0.0,
            'active_speech_duration': 0.0,
            'silence_ratio': 1.0,
            'mean_silence_duration': 0.0,
            'max_silence_duration': 0.0
        }
    
    first_voice = np.argmax(voice_frames) if np.any(voice_frames) else total_frames
    leading_silence = first_voice * hop_length / sr
    
    last_voice = total_frames - np.argmax(voice_frames[::-1]) if np.any(voice_frames) else 0
    trailing_silence = (total_frames - last_voice) * hop_length / sr
    
    active_frames = np.sum(voice_frames)
    active_speech_duration = active_frames * hop_length / sr
    
    total_duration = len(audio) / sr
    
    silence_ratio = 1 - (active_speech_duration / total_duration) if total_duration > 0 else 1.0
    
    silence_segments = []
    if not np.all(voice_frames):
        silence = ~voice_frames
        diff = np.diff(np.concatenate(([0], silence.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            duration = (end - start) * hop_length / sr
            if duration > 0.1:
                silence_segments.append(duration)
    
    mean_silence_duration = np.mean(silence_segments) if silence_segments else 0.0
    max_silence_duration = np.max(silence_segments) if silence_segments else 0.0
    
    return {
        'leading_silence': float(leading_silence),
        'trailing_silence': float(trailing_silence),
        'active_speech_duration': float(active_speech_duration),
        'silence_ratio': float(silence_ratio),
        'mean_silence_duration': float(mean_silence_duration),
        'max_silence_duration': float(max_silence_duration)
    }


def compute_energy_metrics(audio: np.ndarray) -> Dict[str, float]:
    rms_energy = np.sqrt(np.mean(audio**2))
    
    peak_amplitude = np.max(np.abs(audio))
    
    if rms_energy > 0:
        dynamic_range_db = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
    else:
        dynamic_range_db = 0.0
    
    abs_audio = np.abs(audio)
    energy_p10 = np.percentile(abs_audio, 10)
    energy_p50 = np.percentile(abs_audio, 50)
    energy_p90 = np.percentile(abs_audio, 90)
    
    energy_variance = np.var(audio)
    
    return {
        'rms_energy': float(rms_energy),
        'peak_amplitude': float(peak_amplitude),
        'dynamic_range_db': float(dynamic_range_db),
        'energy_p10': float(energy_p10),
        'energy_p50': float(energy_p50),
        'energy_p90': float(energy_p90),
        'energy_variance': float(energy_variance)
    }


def compute_speaking_rate(audio: np.ndarray, sr: int) -> Dict[str, float]:
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    spectral_flux_mean = np.mean(onset_env)
    spectral_flux_std = np.std(onset_env)
    
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, 
                                   pre_avg=3, post_avg=5, delta=0.5, 
                                   wait=10)
    
    duration = len(audio) / sr
    if duration > 0:
        syllable_rate = len(peaks) / duration
    else:
        syllable_rate = 0.0
    
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    return {
        'zcr_mean': float(zcr_mean),
        'zcr_std': float(zcr_std),
        'spectral_flux_mean': float(spectral_flux_mean),
        'spectral_flux_std': float(spectral_flux_std),
        'syllable_rate': float(syllable_rate),
        'tempo_bpm': float(tempo)
    }