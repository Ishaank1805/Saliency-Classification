import re
from .gc_base import BaseFeatureExtractor


class PronounFeatureExtractor(BaseFeatureExtractor):
    """Extract pronoun usage features"""
    
    def __init__(self):
        super().__init__()
        self.pronouns = {
            'first_singular': re.compile(r'\b(I|me|my|mine|myself)\b', re.IGNORECASE),
            'first_plural': re.compile(r'\b(we|us|our|ours|ourselves)\b', re.IGNORECASE),
            'second_person': re.compile(r'\b(you|your|yours|yourself|yourselves)\b', re.IGNORECASE),
            'third_singular': re.compile(r'\b(he|him|his|himself|she|her|hers|herself|it|its|itself)\b', re.IGNORECASE),
            'third_plural': re.compile(r'\b(they|them|their|theirs|themselves)\b', re.IGNORECASE)
        }
    
    def extract(self, text):
        """Extract pronoun features"""
        features = {}
        total_words = len(text.split())
        
        # Count each pronoun type
        for pronoun_type, pattern in self.pronouns.items():
            matches = pattern.findall(text)
            features[f'pronoun_{pronoun_type}_count'] = len(matches)
            features[f'pronoun_{pronoun_type}_ratio'] = len(matches) / (total_words + 1)
        
        # Overall pronoun density
        total_pronouns = sum(len(pattern.findall(text)) for pattern in self.pronouns.values())
        features['pronoun_total_ratio'] = total_pronouns / (total_words + 1)
        
        # First vs third person ratio
        first_person = len(self.pronouns['first_singular'].findall(text)) + \
                      len(self.pronouns['first_plural'].findall(text))
        third_person = len(self.pronouns['third_singular'].findall(text)) + \
                      len(self.pronouns['third_plural'].findall(text))
        features['first_third_person_ratio'] = first_person / (third_person + 1)
        
        return features