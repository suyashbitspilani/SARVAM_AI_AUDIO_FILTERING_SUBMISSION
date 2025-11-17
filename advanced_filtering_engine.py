#!/usr/bin/env python3
"""
Advanced Statistical Filtering Engine for IndicVoices Dataset
============================================================

Based on comprehensive analysis of real IndicVoices features, this implements
statistically optimal filtering methods with clever preprocessing.

Features leveraged:
- 5 Text features for linguistic quality analysis
- 6 Speaker demographic features for consistency validation  
- 3 Geographic features for environment-based filtering
- 2 Quality indicators for meta-quality assessment
- Rich categorical and temporal features for context-aware filtering
"""

import pandas as pd
import numpy as np
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedTextQualityAnalyzer:
    """
    Advanced text quality analysis leveraging all text fields in IndicVoices.
    """
    
    def __init__(self, language='hindi'):
        self.language = language
        self.script_ranges = {
            'hindi': [(0x0900, 0x097F)],    # Devanagari
            'bengali': [(0x0980, 0x09FF)],  # Bengali  
            'tamil': [(0x0B80, 0x0BFF)],    # Tamil
            'telugu': [(0x0C00, 0x0C7F)],   # Telugu
            'gujarati': [(0x0A80, 0x0AFF)], # Gujarati
            'kannada': [(0x0C80, 0x0CFF)],  # Kannada
            'malayalam': [(0x0D00, 0x0D7F)], # Malayalam
            'punjabi': [(0x0A00, 0x0A7F)],  # Gurmukhi
            'odia': [(0x0B00, 0x0B7F)],     # Odia
            'marathi': [(0x0900, 0x097F)],  # Devanagari
        }
    
    def analyze_comprehensive_text_quality(self, sample: Dict) -> Dict[str, float]:
        """
        Comprehensive text quality analysis using all available text fields.
        """
        text = sample.get('text', '')
        verbatim = sample.get('verbatim', '')
        normalized = sample.get('normalized', '')
        unsanitized_verbatim = sample.get('unsanitized_verbatim', '')
        unsanitized_normalized = sample.get('unsanitized_normalized', '')
        duration = sample.get('duration', 0)
        
        quality_metrics = {}
        
        # 1. Script purity and consistency
        quality_metrics['script_purity'] = self._calculate_script_purity(text)
        quality_metrics['script_consistency'] = self._calculate_script_consistency([
            text, verbatim, normalized
        ])
        
        # 2. Multi-level text alignment analysis
        quality_metrics['text_verbatim_alignment'] = self._calculate_alignment_score(text, verbatim)
        quality_metrics['verbatim_normalized_alignment'] = self._calculate_alignment_score(verbatim, normalized)
        quality_metrics['sanitization_quality'] = self._calculate_sanitization_quality(
            unsanitized_verbatim, verbatim, unsanitized_normalized, normalized
        )
        
        # 3. Content density and speech timing analysis
        quality_metrics['content_density'] = self._calculate_content_density(text, duration)
        quality_metrics['speech_rate_consistency'] = self._calculate_speech_rate_consistency(
            text, verbatim, duration
        )
        
        # 4. Linguistic complexity and appropriateness
        quality_metrics['linguistic_complexity'] = self._calculate_linguistic_complexity(text)
        quality_metrics['text_completeness'] = self._calculate_text_completeness([
            text, verbatim, normalized
        ])
        
        # 5. Language-specific quality patterns
        quality_metrics['language_pattern_quality'] = self._calculate_language_patterns(text)
        
        return quality_metrics
    
    def _calculate_script_purity(self, text: str) -> float:
        """Calculate percentage of characters in expected script."""
        if not text or self.language not in self.script_ranges:
            return 0.5
            
        script_ranges = self.script_ranges[self.language]
        text_chars = [c for c in text if not c.isspace() and not c.ispunctuation()]
        
        if not text_chars:
            return 0.0
            
        script_chars = 0
        for char in text_chars:
            char_code = ord(char)
            for start, end in script_ranges:
                if start <= char_code <= end:
                    script_chars += 1
                    break
        
        return script_chars / len(text_chars)
    
    def _calculate_script_consistency(self, text_fields: List[str]) -> float:
        """Calculate consistency of script usage across text fields."""
        purities = [self._calculate_script_purity(text) for text in text_fields if text]
        if not purities:
            return 0.0
        return 1.0 - np.std(purities)  # Higher consistency = lower std deviation
    
    def _calculate_alignment_score(self, text1: str, text2: str) -> float:
        """Calculate semantic alignment between two text versions."""
        if not text1 or not text2:
            return 0.5
            
        # Word-level Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_score = len(intersection) / len(union)
        
        # Length similarity penalty
        len1, len2 = len(text1), len(text2)
        length_similarity = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        return 0.7 * jaccard_score + 0.3 * length_similarity
    
    def _calculate_sanitization_quality(self, unsanitized_verbatim: str, verbatim: str,
                                       unsanitized_normalized: str, normalized: str) -> float:
        """Assess quality of text sanitization process."""
        scores = []
        
        # Verbatim sanitization quality
        if unsanitized_verbatim and verbatim:
            verbatim_quality = self._assess_sanitization(unsanitized_verbatim, verbatim)
            scores.append(verbatim_quality)
            
        # Normalized sanitization quality  
        if unsanitized_normalized and normalized:
            normalized_quality = self._assess_sanitization(unsanitized_normalized, normalized)
            scores.append(normalized_quality)
            
        return np.mean(scores) if scores else 0.5
    
    def _assess_sanitization(self, unsanitized: str, sanitized: str) -> float:
        """Assess individual sanitization quality."""
        # Sanitized should be cleaner (fewer special chars, more standard)
        unsanitized_special = len(re.findall(r'[^\w\s]', unsanitized))
        sanitized_special = len(re.findall(r'[^\w\s]', sanitized))
        
        # Good sanitization reduces special characters
        if len(unsanitized) == 0:
            return 0.0
        special_char_reduction = 1.0 - (sanitized_special / max(unsanitized_special, 1))
        
        # Content preservation (shouldn't remove too much content)
        content_preservation = len(sanitized) / max(len(unsanitized), 1)
        content_preservation = min(content_preservation, 1.0)
        
        return 0.6 * special_char_reduction + 0.4 * content_preservation
    
    def _calculate_content_density(self, text: str, duration: float) -> float:
        """Calculate optimal content density for speech."""
        if duration <= 0 or not text:
            return 0.0
            
        # Characters per second (language-specific optimal rates)
        chars_per_second = len(text.replace(' ', '')) / duration
        
        # Language-specific optimal rates (chars/second)
        optimal_rates = {
            'hindi': (2.5, 5.5),
            'bengali': (2.0, 5.0),
            'tamil': (2.8, 6.0),
            'telugu': (2.5, 5.5),
            'default': (2.0, 6.0)
        }
        
        optimal_min, optimal_max = optimal_rates.get(self.language, optimal_rates['default'])
        
        if optimal_min <= chars_per_second <= optimal_max:
            return 1.0
        elif chars_per_second < optimal_min:
            return chars_per_second / optimal_min
        else:
            return optimal_max / chars_per_second
    
    def _calculate_speech_rate_consistency(self, text: str, verbatim: str, duration: float) -> float:
        """Calculate consistency between text variants and speech timing."""
        if duration <= 0:
            return 0.0
            
        densities = []
        for txt in [text, verbatim]:
            if txt:
                density = len(txt.replace(' ', '')) / duration
                densities.append(density)
        
        if len(densities) < 2:
            return 0.5
            
        # Consistent speech rate = similar densities
        density_std = np.std(densities)
        density_mean = np.mean(densities)
        
        # Normalize by mean to get coefficient of variation
        cv = density_std / max(density_mean, 0.1)
        return max(0.0, 1.0 - cv)
    
    def _calculate_linguistic_complexity(self, text: str) -> float:
        """Calculate linguistic complexity score."""
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        # Word complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        vocab_diversity = len(set(words)) / len(words)  # Type-token ratio
        
        # Sentence structure complexity
        sentences = len(re.findall(r'[.!?।]', text))  # Include Devanagari full stop
        avg_sentence_length = len(words) / max(sentences, 1)
        
        # Normalize to 0-1 scale
        word_complexity = min(avg_word_length / 8, 1.0)
        sentence_complexity = min(avg_sentence_length / 15, 1.0)
        
        return 0.4 * word_complexity + 0.3 * vocab_diversity + 0.3 * sentence_complexity
    
    def _calculate_text_completeness(self, text_fields: List[str]) -> float:
        """Calculate completeness across text fields."""
        non_empty_fields = sum(1 for text in text_fields if text and text.strip())
        total_fields = len([text for text in text_fields if text is not None])
        
        if total_fields == 0:
            return 0.0
            
        return non_empty_fields / total_fields
    
    def _calculate_language_patterns(self, text: str) -> float:
        """Calculate language-specific pattern quality."""
        if not text:
            return 0.0
            
        # Language-specific patterns
        if self.language == 'hindi':
            return self._analyze_hindi_patterns(text)
        elif self.language == 'bengali':
            return self._analyze_bengali_patterns(text)
        elif self.language == 'tamil':
            return self._analyze_tamil_patterns(text)
        else:
            return 0.5  # Default score for other languages
    
    def _analyze_hindi_patterns(self, text: str) -> float:
        """Analyze Hindi-specific linguistic patterns."""
        # Check for proper Devanagari conjunct consonants
        conjunct_pattern = r'[\u0915-\u0939][\u094D][\u0915-\u0939]'
        conjuncts = len(re.findall(conjunct_pattern, text))
        
        # Check for proper vowel marks
        vowel_marks = len(re.findall(r'[\u093E-\u094C]', text))
        
        # Calculate pattern density
        total_chars = len([c for c in text if ord(c) >= 0x0900 and ord(c) <= 0x097F])
        if total_chars == 0:
            return 0.0
            
        pattern_density = (conjuncts + vowel_marks) / total_chars
        return min(pattern_density * 3, 1.0)  # Normalize
    
    def _analyze_bengali_patterns(self, text: str) -> float:
        """Analyze Bengali-specific patterns."""
        # Bengali script pattern analysis
        bengali_chars = len([c for c in text if ord(c) >= 0x0980 and ord(c) <= 0x09FF])
        total_chars = len(text.replace(' ', ''))
        
        return bengali_chars / max(total_chars, 1)
    
    def _analyze_tamil_patterns(self, text: str) -> float:
        """Analyze Tamil-specific patterns."""
        # Tamil script pattern analysis
        tamil_chars = len([c for c in text if ord(c) >= 0x0B80 and ord(c) <= 0x0BFF])
        total_chars = len(text.replace(' ', ''))
        
        return tamil_chars / max(total_chars, 1)

class SpeakerIntelligenceEngine:
    """
    Advanced speaker-based quality analysis and filtering.
    """
    
    def __init__(self):
        self.speaker_profiles = defaultdict(lambda: {
            'samples': [],
            'quality_scores': [],
            'demographic_consistency': [],
            'recording_contexts': []
        })
    
    def analyze_speaker_quality_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze speaker-level quality patterns for intelligent filtering.
        """
        logger.info("Analyzing speaker quality patterns...")
        
        speaker_analysis = {}
        
        for speaker_id in df['speaker_id'].unique():
            speaker_data = df[df['speaker_id'] == speaker_id]
            
            # Skip speakers with too few samples
            if len(speaker_data) < 2:
                continue
                
            analysis = self._analyze_individual_speaker(speaker_data)
            speaker_analysis[speaker_id] = analysis
        
        return speaker_analysis
    
    def _analyze_individual_speaker(self, speaker_data: pd.DataFrame) -> Dict:
        """Analyze individual speaker quality patterns."""
        analysis = {
            'sample_count': len(speaker_data),
            'quality_consistency': 0.0,
            'demographic_consistency': 0.0,
            'context_diversity': 0.0,
            'reliability_score': 0.0
        }
        
        # Quality consistency across samples
        if 'snr' in speaker_data.columns:
            snr_values = speaker_data['snr'].dropna()
            if len(snr_values) > 1:
                analysis['quality_consistency'] = 1.0 / (1.0 + np.std(snr_values))
        
        # Demographic consistency
        analysis['demographic_consistency'] = self._assess_demographic_consistency(speaker_data)
        
        # Recording context diversity
        analysis['context_diversity'] = self._assess_context_diversity(speaker_data)
        
        # Overall reliability score
        analysis['reliability_score'] = (
            0.4 * analysis['quality_consistency'] +
            0.3 * analysis['demographic_consistency'] +
            0.3 * analysis['context_diversity']
        )
        
        return analysis
    
    def _assess_demographic_consistency(self, speaker_data: pd.DataFrame) -> float:
        """Assess consistency of demographic information."""
        consistency_scores = []
        
        # Check consistency across demographic fields
        demo_fields = ['gender', 'age_group', 'occupation', 'qualification']
        
        for field in demo_fields:
            if field in speaker_data.columns:
                unique_values = speaker_data[field].nunique()
                consistency = 1.0 if unique_values <= 1 else 1.0 / unique_values
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _assess_context_diversity(self, speaker_data: pd.DataFrame) -> float:
        """Assess diversity of recording contexts (good for robustness)."""
        diversity_scores = []
        
        context_fields = ['scenario', 'task_name', 'district']
        
        for field in context_fields:
            if field in speaker_data.columns:
                unique_values = speaker_data[field].nunique()
                total_samples = len(speaker_data)
                diversity = min(unique_values / total_samples, 1.0)
                diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.5

class GeographicIntelligenceEngine:
    """
    Geographic-based quality analysis and environmental inference.
    """
    
    def __init__(self):
        self.location_profiles = defaultdict(lambda: {
            'quality_scores': [],
            'recording_conditions': [],
            'demographic_patterns': []
        })
    
    def analyze_geographic_quality_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze location-based quality patterns.
        """
        logger.info("Analyzing geographic quality patterns...")
        
        geographic_analysis = {}
        
        # Analyze by district
        if 'district' in df.columns:
            geographic_analysis['district_analysis'] = self._analyze_location_level(
                df, 'district'
            )
        
        # Analyze by state
        if 'state' in df.columns:
            geographic_analysis['state_analysis'] = self._analyze_location_level(
                df, 'state'
            )
            
        # Analyze by area type
        if 'area' in df.columns:
            geographic_analysis['area_analysis'] = self._analyze_location_level(
                df, 'area'
            )
        
        return geographic_analysis
    
    def _analyze_location_level(self, df: pd.DataFrame, location_field: str) -> Dict:
        """Analyze quality patterns at specific geographic level."""
        location_analysis = {}
        
        for location in df[location_field].unique():
            if pd.isna(location):
                continue
                
            location_data = df[df[location_field] == location]
            
            if len(location_data) < 3:  # Skip locations with too few samples
                continue
                
            analysis = {
                'sample_count': len(location_data),
                'avg_quality': 0.0,
                'quality_variance': 0.0,
                'demographic_diversity': 0.0,
                'recording_condition_score': 0.0
            }
            
            # Quality metrics
            if 'snr' in location_data.columns:
                snr_values = location_data['snr'].dropna()
                if len(snr_values) > 0:
                    analysis['avg_quality'] = float(snr_values.mean())
                    analysis['quality_variance'] = float(snr_values.var())
            
            # Demographic diversity (good indicator of recording setup quality)
            analysis['demographic_diversity'] = self._calculate_demographic_diversity(location_data)
            
            # Infer recording conditions
            analysis['recording_condition_score'] = self._infer_recording_conditions(location_data)
            
            location_analysis[location] = analysis
        
        return location_analysis
    
    def _calculate_demographic_diversity(self, location_data: pd.DataFrame) -> float:
        """Calculate demographic diversity score for location."""
        diversity_scores = []
        
        demo_fields = ['gender', 'age_group', 'occupation']
        
        for field in demo_fields:
            if field in location_data.columns:
                value_counts = location_data[field].value_counts()
                total_samples = len(location_data)
                
                # Shannon diversity index
                proportions = value_counts / total_samples
                shannon_diversity = -sum(p * np.log2(p) for p in proportions if p > 0)
                
                # Normalize by maximum possible diversity
                max_diversity = np.log2(len(value_counts))
                normalized_diversity = shannon_diversity / max_diversity if max_diversity > 0 else 0
                
                diversity_scores.append(normalized_diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.5
    
    def _infer_recording_conditions(self, location_data: pd.DataFrame) -> float:
        """Infer recording condition quality from patterns."""
        condition_indicators = []
        
        # Quality consistency (good equipment/environment = consistent quality)
        if 'snr' in location_data.columns:
            snr_values = location_data['snr'].dropna()
            if len(snr_values) > 1:
                consistency = 1.0 / (1.0 + np.std(snr_values))
                condition_indicators.append(consistency)
        
        # Task diversity (good setup can handle various tasks)
        if 'task_name' in location_data.columns:
            task_diversity = location_data['task_name'].nunique() / len(location_data)
            condition_indicators.append(min(task_diversity, 1.0))
        
        return np.mean(condition_indicators) if condition_indicators else 0.5

class HierarchicalOutlierDetector:
    """
    Multi-level hierarchical outlier detection engine.
    """
    
    def __init__(self):
        self.outlier_thresholds = {
            'global': 3.0,
            'language': 2.5, 
            'speaker': 2.0,
            'location': 2.0,
            'task': 2.5
        }
    
    def detect_hierarchical_outliers(self, df: pd.DataFrame, 
                                   metric_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers at multiple hierarchical levels.
        """
        logger.info("Performing hierarchical outlier detection...")
        
        outlier_flags = {}
        
        # Global level outliers
        outlier_flags['global'] = self._detect_global_outliers(df, metric_cols)
        
        # Language level outliers
        if 'lang' in df.columns:
            outlier_flags['language'] = self._detect_grouped_outliers(
                df, metric_cols, 'lang', self.outlier_thresholds['language']
            )
        
        # Speaker level outliers
        if 'speaker_id' in df.columns:
            outlier_flags['speaker'] = self._detect_speaker_level_outliers(df, metric_cols)
        
        # Location level outliers
        if 'district' in df.columns:
            outlier_flags['location'] = self._detect_grouped_outliers(
                df, metric_cols, 'district', self.outlier_thresholds['location']
            )
        
        # Task level outliers
        if 'task_name' in df.columns:
            outlier_flags['task'] = self._detect_grouped_outliers(
                df, metric_cols, 'task_name', self.outlier_thresholds['task']
            )
        
        return outlier_flags
    
    def _detect_global_outliers(self, df: pd.DataFrame, 
                              metric_cols: List[str]) -> pd.DataFrame:
        """Detect global dataset outliers."""
        outlier_flags = pd.DataFrame(False, index=df.index, columns=metric_cols)
        
        for col in metric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    # Modified Z-score method
                    median_val = values.median()
                    mad = np.median(np.abs(values - median_val))
                    modified_z_scores = 0.6745 * (df[col] - median_val) / (mad + 1e-6)
                    outlier_flags[col] = np.abs(modified_z_scores) > self.outlier_thresholds['global']
        
        return outlier_flags
    
    def _detect_grouped_outliers(self, df: pd.DataFrame, metric_cols: List[str],
                               group_col: str, threshold: float) -> pd.DataFrame:
        """Detect outliers within groups."""
        outlier_flags = pd.DataFrame(False, index=df.index, columns=metric_cols)
        
        for group_value in df[group_col].unique():
            if pd.isna(group_value):
                continue
                
            group_mask = df[group_col] == group_value
            group_data = df[group_mask]
            
            if len(group_data) < 3:  # Need minimum samples for meaningful outlier detection
                continue
                
            for col in metric_cols:
                if col in df.columns:
                    values = group_data[col].dropna()
                    if len(values) > 2:
                        median_val = values.median()
                        mad = np.median(np.abs(values - median_val))
                        
                        if mad > 0:
                            modified_z_scores = 0.6745 * (group_data[col] - median_val) / mad
                            group_outliers = np.abs(modified_z_scores) > threshold
                            outlier_flags.loc[group_mask, col] = group_outliers
        
        return outlier_flags
    
    def _detect_speaker_level_outliers(self, df: pd.DataFrame,
                                     metric_cols: List[str]) -> pd.DataFrame:
        """Detect outliers at speaker level with special logic."""
        outlier_flags = pd.DataFrame(False, index=df.index, columns=metric_cols)
        
        for speaker_id in df['speaker_id'].unique():
            speaker_mask = df['speaker_id'] == speaker_id
            speaker_data = df[speaker_mask]
            
            if len(speaker_data) < 2:
                continue
                
            # For speakers, look for samples that are inconsistent with their pattern
            for col in metric_cols:
                if col in df.columns and len(speaker_data) > 2:
                    values = speaker_data[col].dropna()
                    if len(values) > 1:
                        # Use smaller threshold for speaker consistency
                        speaker_mean = values.mean()
                        speaker_std = values.std()
                        
                        if speaker_std > 0:
                            z_scores = (speaker_data[col] - speaker_mean) / speaker_std
                            speaker_outliers = np.abs(z_scores) > self.outlier_thresholds['speaker']
                            outlier_flags.loc[speaker_mask, col] = speaker_outliers
        
        return outlier_flags

def main():
    """Demonstrate the advanced filtering engine."""
    print("🚀 Advanced Statistical Filtering Engine")
    print("Based on comprehensive IndicVoices dataset analysis")
    print("=" * 80)
    
    # This would be integrated into the main pipeline
    # For now, just show the component structure
    
    text_analyzer = AdvancedTextQualityAnalyzer('hindi')
    speaker_engine = SpeakerIntelligenceEngine()
    geo_engine = GeographicIntelligenceEngine()
    outlier_detector = HierarchicalOutlierDetector()
    
    print("✅ Advanced components initialized:")
    print("  📝 Text Quality Analyzer - 10+ metrics")
    print("  👤 Speaker Intelligence Engine - Consistency analysis")
    print("  🌍 Geographic Intelligence Engine - Environment inference")
    print("  📊 Hierarchical Outlier Detector - Multi-level analysis")
    print("\n🎯 Ready for integration into main pipeline!")

if __name__ == "__main__":
    main()