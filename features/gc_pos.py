from collections import defaultdict
from .gc_base import BaseFeatureExtractor


class POSFeatureExtractor(BaseFeatureExtractor):
    """Extract Part-of-Speech ratio features"""
    
    def extract(self, text):
        """Extract POS ratio features"""
        self.initialize_nlp()
        doc = self.nlp_stanza(text)
        
        pos_counts = defaultdict(int)
        
        for sent in doc.sentences:
            for word in sent.words:
                pos_counts[word.upos] += 1
        
        # Calculate ratios with add-one smoothing
        ratios = {
            'adverb_noun': pos_counts['ADV'] / (pos_counts['NOUN'] + 1),
            'adverb_pronoun': pos_counts['ADV'] / (pos_counts['PRON'] + 1),
            'adjective_verb': pos_counts['ADJ'] / (pos_counts['VERB'] + 1),
            'noun_verb': pos_counts['NOUN'] / (pos_counts['VERB'] + 1),
            'verb_pronoun': pos_counts['VERB'] / (pos_counts['PRON'] + 1),
            'adverb_adjective': pos_counts['ADV'] / (pos_counts['ADJ'] + 1),
            'adjective_pronoun': pos_counts['ADJ'] / (pos_counts['PRON'] + 1),
            'noun_pronoun': pos_counts['NOUN'] / (pos_counts['PRON'] + 1)
        }
        
        return ratios