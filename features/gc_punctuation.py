import re
from .gc_base import BaseFeatureExtractor


class PunctuationFeatureExtractor(BaseFeatureExtractor):
    """Extract punctuation-based features"""
    
    def __init__(self):
        super().__init__()
        self.exclamation_pattern = re.compile(r'!')
        self.question_pattern = re.compile(r'\?')
        self.ellipsis_pattern = re.compile(r'\.\.\.')
    
    def extract(self, text):
        """Extract punctuation features"""
        features = {}
        total_words = len(text.split())
        
        # Exclamations
        exclamations = self.exclamation_pattern.findall(text)
        features['exclamation_count'] = len(exclamations)
        
        # Questions
        questions = self.question_pattern.findall(text)
        features['question_count'] = len(questions)
        
        # Ellipses
        ellipses = self.ellipsis_pattern.findall(text)
        features['ellipsis_count'] = len(ellipses)
        
        # Sentence endings ratio
        total_sentences = text.count('.') + text.count('!') + text.count('?')
        if total_sentences > 0:
            features['exclamation_sentence_ratio'] = len(exclamations) / total_sentences
            features['question_sentence_ratio'] = len(questions) / total_sentences
        else:
            features['exclamation_sentence_ratio'] = 0
            features['question_sentence_ratio'] = 0
        
        # Comma density
        features['comma_density'] = text.count(',') / (total_words + 1)
        
        # Semicolon and colon usage
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        
        return features