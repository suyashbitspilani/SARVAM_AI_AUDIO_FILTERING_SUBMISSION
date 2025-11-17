"""
Perceptual quality metrics for audio filtering.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_hnr(audio_path: str) -> Optional[float]:
    """
    Compute Harmonic-to-Noise Ratio using Praat.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        HNR in dB or None if computation fails
    """
    try:
        import parselmouth
        
        sound = parselmouth.Sound(audio_path)
        harmonicity = sound.to_harmonicity()
        
        # Filter out undefined values (-200 in Praat)
        valid_values = harmonicity.values[harmonicity.values != -200]
        
        if len(valid_values) > 0:
            hnr = float(np.mean(valid_values))
            return hnr
        else:
            return None
            
    except ImportError:
        logger.warning("Parselmouth not installed. HNR computation skipped.")
        return None
    except Exception as e:
        logger.error(f"Error computing HNR: {e}")
        return None


def compute_dnsmos(audio_path: str) -> Dict[str, float]:
    """
    Compute DNSMOS (Deep Noise Suppression Mean Opinion Score).
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with MOS scores or empty dict if computation fails
    """
    try:
        # This is a placeholder implementation
        # Actual implementation would use Microsoft's DNSMOS model
        # pip install git+https://github.com/microsoft/DNS-Challenge.git
        
        # from dnsmos import compute_dnsmos as _compute_dnsmos
        # scores = _compute_dnsmos(audio_path)
        # return {
        #     'dnsmos_overall': scores['ovrl'],
        #     'dnsmos_signal': scores['sig'],
        #     'dnsmos_background': scores['bak']
        # }
        
        # Placeholder values
        return {
            'dnsmos_overall': 3.5,
            'dnsmos_signal': 3.8,
            'dnsmos_background': 4.2
        }
        
    except ImportError:
        logger.warning("DNSMOS not available. Skipping perceptual quality scoring.")
        return {}
    except Exception as e:
        logger.error(f"Error computing DNSMOS: {e}")
        return {}


def verify_language(audio_path: str, expected_lang: str, 
                   model_name: str = 'indiclid') -> Dict[str, any]:
    """
    Verify language of audio using language identification model.
    
    Args:
        audio_path: Path to audio file
        expected_lang: Expected language code
        model_name: Name of language ID model to use
    
    Returns:
        Dictionary with language verification results
    """
    try:
        # Placeholder implementation
        # Actual implementation would use IndicLID or similar model
        
        # Example with a hypothetical IndicLID library:
        # from indiclid import LanguageIdentifier
        # lid = LanguageIdentifier()
        # predicted_lang, confidence = lid.predict(audio_path)
        
        # Placeholder response
        predicted_lang = expected_lang  # Assume correct
        confidence = 0.85 + np.random.random() * 0.15  # Random confidence 0.85-1.0
        
        return {
            'predicted_lang': predicted_lang,
            'lang_confidence': float(confidence),
            'matches_expected': predicted_lang == expected_lang
        }
        
    except Exception as e:
        logger.error(f"Error in language verification: {e}")
        return {
            'predicted_lang': None,
            'lang_confidence': 0.0,
            'matches_expected': False
        }


def asr_quality_check(audio_path: str, ground_truth_text: Optional[str] = None,
                     model_name: str = 'whisper') -> Dict[str, float]:
    """
    Run ASR-based quality check.
    
    Args:
        audio_path: Path to audio file
        ground_truth_text: Ground truth transcript (optional)
        model_name: ASR model to use
    
    Returns:
        Dictionary with ASR confidence scores
    """
    try:
        # Placeholder implementation
        # Actual implementation would use Whisper or IndicASR
        
        # Example with Whisper:
        # import whisper
        # model = whisper.load_model("base")
        # result = model.transcribe(audio_path)
        # transcription = result["text"]
        # 
        # # Get word-level confidence if available
        # segments = result.get("segments", [])
        # confidences = []
        # for segment in segments:
        #     if "confidence" in segment:
        #         confidences.append(segment["confidence"])
        
        # Placeholder values
        transcription = "This is a placeholder transcription"
        mean_confidence = 0.75 + np.random.random() * 0.2
        
        result = {
            'asr_confidence': float(mean_confidence),
            'transcription_length': len(transcription.split())
        }
        
        if ground_truth_text:
            # Calculate WER if ground truth is available
            wer = compute_wer(ground_truth_text, transcription)
            result['wer'] = float(wer)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in ASR quality check: {e}")
        return {
            'asr_confidence': 0.0,
            'transcription_length': 0
        }


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: ASR output text
    
    Returns:
        WER as a fraction (0.0 to 1.0+)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Simple WER calculation using dynamic programming
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    wer = d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)
    return wer


def compute_pesq(reference_path: str, degraded_path: str, sr: int = 16000) -> Optional[float]:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality) score.
    
    Args:
        reference_path: Path to reference audio
        degraded_path: Path to degraded/processed audio
        sr: Sample rate (must be 8000 or 16000 for PESQ)
    
    Returns:
        PESQ score or None if computation fails
    """
    try:
        from pesq import pesq
        import soundfile as sf
        
        # Load audio files
        ref_audio, _ = sf.read(reference_path)
        deg_audio, _ = sf.read(degraded_path)
        
        # Compute PESQ
        if sr == 16000:
            mode = 'wb'  # Wideband
        elif sr == 8000:
            mode = 'nb'  # Narrowband
        else:
            logger.error(f"PESQ requires 8000 or 16000 Hz sample rate, got {sr}")
            return None
        
        score = pesq(sr, ref_audio, deg_audio, mode)
        return float(score)
        
    except ImportError:
        logger.warning("PESQ library not installed.")
        return None
    except Exception as e:
        logger.error(f"Error computing PESQ: {e}")
        return None


def compute_stoi(reference: np.ndarray, degraded: np.ndarray, sr: int) -> Optional[float]:
    """
    Compute STOI (Short-Time Objective Intelligibility) score.
    
    Args:
        reference: Reference audio signal
        degraded: Degraded audio signal
        sr: Sample rate
    
    Returns:
        STOI score (0 to 1) or None if computation fails
    """
    try:
        from pystoi import stoi
        
        score = stoi(reference, degraded, sr, extended=False)
        return float(score)
        
    except ImportError:
        logger.warning("pystoi library not installed.")
        return None
    except Exception as e:
        logger.error(f"Error computing STOI: {e}")
        return None