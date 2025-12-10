import numpy as np
from .gc_base import BaseFeatureExtractor


class BasicFeatureExtractor(BaseFeatureExtractor):
    """Extract basic text statistics"""
    
    def __init__(self):
        super().__init__()
        self.content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
        self.function_pos = {'DET', 'ADP', 'PRON', 'CONJ', 'SCONJ', 'AUX', 'CCONJ', 'PART'}
    
    def extract(self, text):
        """Extract basic features: sentence length and lexical density"""
        self.initialize_nlp()
        doc = self.nlp_stanza(text)
        
        features = {}
        
        # Sentence length statistics
        sentence_lengths = [len(sent.words) for sent in doc.sentences]
        if sentence_lengths:
            features['avg_sen_len'] = np.mean(sentence_lengths)
            features['std_sen_len'] = np.std(sentence_lengths)
        else:
            features['avg_sen_len'] = 0
            features['std_sen_len'] = 0
        
        # Lexical density
        content_words = 0
        function_words = 0
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos in self.content_pos:
                    content_words += 1
                elif word.upos in self.function_pos:
                    function_words += 1
        
        total_words = content_words + function_words
        features['lex_den'] = content_words / total_words if total_words > 0 else 0
        
        return features