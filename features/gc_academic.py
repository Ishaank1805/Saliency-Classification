import re
from .gc_base import BaseFeatureExtractor


class AcademicFeatureExtractor(BaseFeatureExtractor):
    """Extract academic and non-fiction specific features"""
    
    def __init__(self):
        super().__init__()
        self.citation_patterns = re.compile(
            r'\([A-Z][^)]+,?\s*\d{4}\)|[\[\(]\d+[\]\)]|\b\d{4}[a-z]?\b|'
            r'\bet\s+al\.|ibid\.|op\.\s*cit\.|cf\.|see\s+also|cited\s+in'
        )
        self.numbers_pattern = re.compile(r'\b\d+(\.\d+)?%?\b')
        self.technical_terms = re.compile(r'\b[A-Z]{2,}\b')
        self.parentheticals = re.compile(r'\([^)]+\)')
    
    def extract(self, text):
        """Extract academic features"""
        features = {}
        total_words = len(text.split())
        
        # Citations and references
        citation_matches = self.citation_patterns.findall(text)
        features['citation_count'] = len(citation_matches)
        features['citation_ratio'] = len(citation_matches) / (total_words + 1)
        
        # Numbers and statistics
        number_matches = self.numbers_pattern.findall(text)
        features['number_count'] = len(number_matches)
        features['number_ratio'] = len(number_matches) / (total_words + 1)
        
        # Technical terms
        technical_matches = self.technical_terms.findall(text)
        # Filter out common words
        technical_matches = [t for t in technical_matches if len(t) > 1 and 
                           t not in ['I', 'A', 'THE', 'AND', 'OR', 'BUT']]
        features['technical_term_count'] = len(technical_matches)
        features['technical_term_ratio'] = len(technical_matches) / (total_words + 1)
        
        # Parentheticals
        parenthetical_matches = self.parentheticals.findall(text)
        features['parenthetical_count'] = len(parenthetical_matches)
        features['parenthetical_ratio'] = len(parenthetical_matches) / (total_words + 1)
        
        return features