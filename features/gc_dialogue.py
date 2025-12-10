import re
from .gc_base import BaseFeatureExtractor


class DialogueFeatureExtractor(BaseFeatureExtractor):
    """Extract dialogue-related features"""
    
    def __init__(self):
        super().__init__()
        self.dialogue_markers = re.compile(r'["\'""'']')
        self.direct_speech = re.compile(r'["\'""''].*?["\'""'']', re.DOTALL)
        self.speech_verbs = re.compile(
            r'\b(said|says|saying|asked|asks|asking|replied|replies|replying|'
            r'whispered|whispers|whispering|shouted|shouts|shouting|muttered|'
            r'mutters|muttering|exclaimed|exclaims|exclaiming|answered|answers|'
            r'answering|told|tells|telling|cried|cries|crying|spoke|speaks|'
            r'speaking|declared|declares|declaring|announced|announces|announcing|'
            r'remarked|remarks|remarking|continued|continues|continuing|added|'
            r'adds|adding|interrupted|interrupts|interrupting|suggested|suggests|'
            r'suggesting|demanded|demands|demanding|insisted|insists|insisting|'
            r'explained|explains|explaining|admitted|admits|admitting|agreed|'
            r'agrees|agreeing|argued|argues|arguing|yelled|yells|yelling)\b', 
            re.IGNORECASE
        )
        self.speech_tags = re.compile(r',\s*["\'""'']|["\'""'']\s*,')
    
    def extract(self, text):
        """Extract dialogue features"""
        features = {}
        total_words = len(text.split())
        
        # Direct speech occurrences
        direct_speeches = self.direct_speech.findall(text)
        features['dialogue_count'] = len(direct_speeches)
        features['dialogue_ratio'] = len(direct_speeches) / (total_words + 1)
        
        # Speech verbs
        speech_verb_matches = self.speech_verbs.findall(text)
        features['speech_verb_count'] = len(speech_verb_matches)
        features['speech_verb_ratio'] = len(speech_verb_matches) / (total_words + 1)
        
        # Speech tags
        speech_tags = self.speech_tags.findall(text)
        features['speech_tag_count'] = len(speech_tags)
        
        # Dialogue diversity
        unique_speech_verbs = len(set([v.lower() for v in speech_verb_matches]))
        features['speech_verb_diversity'] = unique_speech_verbs / (len(speech_verb_matches) + 1)
        
        return features