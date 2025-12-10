import re
from .gc_base import BaseFeatureExtractor


class NarrativeFeatureExtractor(BaseFeatureExtractor):
    """Extract narrative-specific features"""
    
    def __init__(self):
        super().__init__()
        self.action_verbs = re.compile(
            r'\b(ran|walked|jumped|looked|turned|grabbed|pushed|pulled|opened|'
            r'closed|moved|stepped|reached|touched|lifted|dropped|threw|caught|'
            r'climbed|fell|slipped|danced|kicked|punched|kissed|hugged|smiled|'
            r'frowned|nodded|shook|waved|pointed|leaned|bent|stretched|twisted|'
            r'rolled|spun|crawled|leaped|bounced|slid|swung|rushed|hurried|'
            r'strolled|marched|tip-toed|stumbled|tripped|collapsed)\b', 
            re.IGNORECASE
        )
        self.sensory_words = re.compile(
            r'\b(saw|heard|felt|smelled|tasted|touched|looked|sounded|seemed|'
            r'appeared|watched|listened|observed|noticed|perceived|glimpsed|'
            r'spotted|detected|sensed|recognized|tasted like|smelled like|'
            r'felt like|looked like|sounded like)\b', 
            re.IGNORECASE
        )
        self.emotion_words = re.compile(
            r'\b(happy|sad|angry|afraid|excited|worried|surprised|disappointed|'
            r'confused|frustrated|anxious|nervous|scared|delighted|pleased|upset|'
            r'furious|terrified|thrilled|depressed|miserable|cheerful|content|'
            r'annoyed|irritated|embarrassed|ashamed|proud|jealous|envious|lonely|'
            r'bored|curious|suspicious|disgusted|relieved|grateful|hopeful|'
            r'desperate|calm|relaxed|tense|stressed)\b', 
            re.IGNORECASE
        )
        self.character_names = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b')
        self.place_indicators = re.compile(
            r'\b(in|at|on|near|beside|behind|above|below|between|through|across|'
            r'along|around|inside|outside|within)\s+(?:the\s+)?[A-Z][a-z]+\b'
        )
    
    def extract(self, text):
        """Extract narrative features"""
        features = {}
        total_words = len(text.split())
        
        # Action verbs
        action_matches = self.action_verbs.findall(text)
        features['action_verb_count'] = len(action_matches)
        features['action_verb_ratio'] = len(action_matches) / (total_words + 1)
        features['action_verb_diversity'] = len(set([v.lower() for v in action_matches])) / (len(action_matches) + 1)
        
        # Sensory words
        sensory_matches = self.sensory_words.findall(text)
        features['sensory_word_count'] = len(sensory_matches)
        features['sensory_word_ratio'] = len(sensory_matches) / (total_words + 1)
        
        # Emotion words
        emotion_matches = self.emotion_words.findall(text)
        features['emotion_word_count'] = len(emotion_matches)
        features['emotion_word_ratio'] = len(emotion_matches) / (total_words + 1)
        
        # Character names
        potential_names = self.character_names.findall(text)
        common_non_names = {'The', 'This', 'That', 'These', 'Those', 'Many', 'Some', 'All', 'New', 'Old'}
        names = [n for n in potential_names if n.split()[0] not in common_non_names]
        features['character_name_count'] = len(names)
        features['character_name_ratio'] = len(names) / (total_words + 1)
        
        # Place indicators
        place_matches = self.place_indicators.findall(text)
        features['place_indicator_count'] = len(place_matches)
        
        return features