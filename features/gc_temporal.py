import re
from .gc_base import BaseFeatureExtractor


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract temporal and tense-related features"""
    
    def __init__(self):
        super().__init__()
        self.past_tense_verbs = re.compile(
            r'\b(was|were|had|did|went|came|saw|said|thought|felt|knew|took|'
            r'made|got|gave|found|told|asked|became|left|brought|began|kept|'
            r'held|stood|heard|let|meant|set|met|ran|paid|sat|spoke|lay|led|'
            r'read|grew|lost|fell|sent|built|understood|broke|spent|drove|'
            r'wrote|beat|bought|caught|taught|sold|fought|flew|drew|chose|'
            r'rose|threw|dealt)\b', 
            re.IGNORECASE
        )
        self.present_tense_verbs = re.compile(
            r'\b(is|are|am|has|have|does|do|goes|go|comes|come|sees|see|says|'
            r'say|thinks|think|feels|feel|knows|know|takes|take|makes|make|'
            r'gets|get|gives|give|finds|find|tells|tell|asks|ask|becomes|'
            r'become|leaves|leave|brings|bring|begins|begin|keeps|keep|holds|'
            r'hold|stands|stand)\b', 
            re.IGNORECASE
        )
        self.time_adverbs = re.compile(
            r'\b(yesterday|today|tomorrow|now|then|later|earlier|soon|recently|'
            r'lately|afterwards|meanwhile|suddenly|immediately|eventually|finally|'
            r'initially|subsequently|presently|formerly|currently|momentarily)\b', 
            re.IGNORECASE
        )
    
    def extract(self, text):
        """Extract temporal features"""
        features = {}
        total_words = len(text.split())
        
        # Past vs present tense
        past_matches = self.past_tense_verbs.findall(text)
        present_matches = self.present_tense_verbs.findall(text)
        
        features['past_tense_count'] = len(past_matches)
        features['present_tense_count'] = len(present_matches)
        features['past_present_ratio'] = len(past_matches) / (len(present_matches) + 1)
        
        # Time adverbs
        time_adverb_matches = self.time_adverbs.findall(text)
        features['time_adverb_count'] = len(time_adverb_matches)
        features['time_adverb_ratio'] = len(time_adverb_matches) / (total_words + 1)
        
        return features