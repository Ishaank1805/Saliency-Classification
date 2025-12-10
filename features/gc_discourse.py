import re
from .gc_base import BaseFeatureExtractor


class DiscourseFeatureExtractor(BaseFeatureExtractor):
    """Extract discourse marker features"""
    
    def __init__(self):
        super().__init__()
        self.causal_markers = re.compile(
            r'\b(because|since|as|therefore|thus|hence|consequently|so|'
            r'accordingly|due to|owing to|as a result|for this reason)\b', 
            re.IGNORECASE
        )
        self.contrast_markers = re.compile(
            r'\b(but|however|although|though|despite|nevertheless|nonetheless|'
            r'yet|still|whereas|while|on the other hand|in contrast|conversely)\b', 
            re.IGNORECASE
        )
        self.addition_markers = re.compile(
            r'\b(and|also|moreover|furthermore|additionally|besides|in addition|'
            r'as well as|not only|similarly|likewise)\b', 
            re.IGNORECASE
        )
    
    def extract(self, text):
        """Extract discourse features"""
        features = {}
        total_words = len(text.split())
        
        # Causal markers
        causal_matches = self.causal_markers.findall(text)
        features['causal_marker_count'] = len(causal_matches)
        features['causal_marker_ratio'] = len(causal_matches) / (total_words + 1)
        
        # Contrast markers
        contrast_matches = self.contrast_markers.findall(text)
        features['contrast_marker_count'] = len(contrast_matches)
        features['contrast_marker_ratio'] = len(contrast_matches) / (total_words + 1)
        
        # Addition markers
        addition_matches = self.addition_markers.findall(text)
        features['addition_marker_count'] = len(addition_matches)
        features['addition_marker_ratio'] = len(addition_matches) / (total_words + 1)
        
        # Total discourse markers
        total_discourse = len(causal_matches) + len(contrast_matches) + len(addition_matches)
        features['discourse_marker_total_ratio'] = total_discourse / (total_words + 1)
        
        return features