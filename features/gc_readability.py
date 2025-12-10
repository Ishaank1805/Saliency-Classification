from .gc_base import BaseFeatureExtractor


class ReadabilityFeatureExtractor(BaseFeatureExtractor):
    """Extract readability scores"""
    
    def extract(self, text):
        """Extract readability features"""
        features = {}
        
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        if sentences > 0 and len(words) > 0:
            # Average words per sentence
            features['avg_words_per_sentence'] = len(words) / sentences
            
            # Average syllables per word
            syllable_count = sum(self._count_syllables(word) for word in words)
            features['avg_syllables_per_word'] = syllable_count / len(words)
            
            # Flesch Reading Ease
            features['flesch_reading_ease'] = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllable_count / len(words))
            
            # Gunning Fog Index
            complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
            features['gunning_fog_index'] = 0.4 * ((len(words) / sentences) + 100 * (complex_words / len(words)))
        else:
            features['avg_words_per_sentence'] = 0
            features['avg_syllables_per_word'] = 0
            features['flesch_reading_ease'] = 0
            features['gunning_fog_index'] = 0
        
        return features
    
    def _count_syllables(self, word):
        """Simple syllable counter"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count