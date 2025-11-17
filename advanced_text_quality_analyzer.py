
import re
import unicodedata
from collections import Counter

class TextQualityAnalyzer:
    """Advanced text quality analysis for IndicVoices data."""
    
    def __init__(self, language='hindi'):
        self.language = language
        self.script_ranges = {
            'hindi': [(0x0900, 0x097F)],  # Devanagari
            'bengali': [(0x0980, 0x09FF)],  # Bengali
            'tamil': [(0x0B80, 0x0BFF)],   # Tamil
        }
    
    def analyze_text_quality(self, sample):
        """Comprehensive text quality analysis."""
        text = sample.get('text', '')
        verbatim = sample.get('verbatim', '')
        normalized = sample.get('normalized', '')
        
        quality_scores = {}
        
        # 1. Script purity analysis
        quality_scores['script_purity'] = self._calculate_script_purity(text)
        
        # 2. Text-verbatim alignment
        quality_scores['text_alignment'] = self._calculate_text_alignment(text, verbatim)
        
        # 3. Normalization consistency
        quality_scores['normalization_quality'] = self._calculate_normalization_quality(
            verbatim, normalized)
        
        # 4. Text complexity and readability
        quality_scores['text_complexity'] = self._calculate_text_complexity(text)
        
        # 5. Language-specific patterns
        quality_scores['language_pattern_score'] = self._calculate_language_patterns(text)
        
        # 6. Content density (text length vs expected speech time)
        duration = sample.get('duration', 0)
        quality_scores['content_density'] = self._calculate_content_density(text, duration)
        
        return quality_scores
    
    def _calculate_script_purity(self, text):
        """Calculate script purity (% of characters in expected script)."""
        if not text or self.language not in self.script_ranges:
            return 0.5
        
        script_ranges = self.script_ranges[self.language]
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return 0.0
        
        script_chars = 0
        for char in text:
            char_code = ord(char)
            for start, end in script_ranges:
                if start <= char_code <= end:
                    script_chars += 1
                    break
        
        return script_chars / total_chars
    
    def _calculate_text_alignment(self, text, verbatim):
        """Calculate similarity between text and verbatim versions."""
        if not text or not verbatim:
            return 0.5
        
        # Simple Jaccard similarity on words
        text_words = set(text.lower().split())
        verbatim_words = set(verbatim.lower().split())
        
        if not text_words and not verbatim_words:
            return 1.0
        
        intersection = text_words.intersection(verbatim_words)
        union = text_words.union(verbatim_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_normalization_quality(self, verbatim, normalized):
        """Assess quality of text normalization."""
        if not verbatim or not normalized:
            return 0.5
        
        # Check if normalization makes sense
        # (normalized should be cleaner, shorter, more standard)
        verbatim_clean = re.sub(r'[^\w\s]', '', verbatim.lower())
        normalized_clean = re.sub(r'[^\w\s]', '', normalized.lower())
        
        # Normalization should not be too different
        similarity = len(set(verbatim_clean.split()).intersection(
            set(normalized_clean.split()))) / max(
            len(set(verbatim_clean.split())), 1)
        
        return similarity
    
    def _calculate_text_complexity(self, text):
        """Calculate text complexity score."""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence complexity (punctuation patterns)
        sentence_complexity = len(re.findall(r'[.!?]', text)) / max(len(words), 1)
        
        # Normalize to 0-1 scale
        complexity_score = (avg_word_length / 10) + sentence_complexity
        return min(complexity_score, 1.0)
    
    def _calculate_language_patterns(self, text):
        """Detect language-specific patterns and quality."""
        if not text:
            return 0.0
        
        # Language-specific pattern detection
        # For Hindi: Check for proper Devanagari patterns
        if self.language == 'hindi':
            # Check for proper conjunct consonants, vowel marks, etc.
            devanagari_chars = re.findall(r'[ऀ-ॿ]', text)
            return len(devanagari_chars) / max(len(text.replace(' ', '')), 1)
        
        return 0.5  # Default for other languages
    
    def _calculate_content_density(self, text, duration):
        """Calculate content density (chars per second of speech)."""
        if duration <= 0 or not text:
            return 0.0
        
        # Typical speech rate: 3-5 characters per second for Indian languages
        chars_per_second = len(text.replace(' ', '')) / duration
        
        # Normalize to expected range
        optimal_range = (2, 6)  # characters per second
        if optimal_range[0] <= chars_per_second <= optimal_range[1]:
            return 1.0
        elif chars_per_second < optimal_range[0]:
            return chars_per_second / optimal_range[0]
        else:
            return optimal_range[1] / chars_per_second
