import numpy as np
from collections import defaultdict
from .gc_base import BaseFeatureExtractor
from huggingface_hub import hf_hub_download


class PolarityFeatureExtractor(BaseFeatureExtractor):
    """Extract sentiment polarity features using Distributional Polarity Lexicon"""
    
    def __init__(self, lexicon_path=None):
        """
        Initialize with path to polarity lexicon file.
        
        Parameters:
        -----------
        lexicon_path : str, optional
            Path to the DPLp-EN lexicon file (lemma::pos format).
            If None, downloads from HuggingFace Hub.
        """
        super().__init__()
        
        # Download from HuggingFace if no path provided
        if lexicon_path is None:
            print("Downloading polarity lexicon from HuggingFace Hub...")
            lexicon_path = hf_hub_download(
                repo_id="Ishaank18/screenplay-lexicons",
                filename="DPLp-EN_lrec2016.txt",
                repo_type="dataset"
            )
            print(f"✓ Downloaded to: {lexicon_path}")
        
        self.polarity_lexicon = self._load_lexicon(lexicon_path)
    
    def _load_lexicon(self, path):
        """Load polarity lexicon from file"""
        lexicon = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        word = parts[0].strip()
                        scores = parts[1].strip().split(',')
                        if len(scores) == 3:
                            try:
                                pos_score = float(scores[0])
                                neg_score = float(scores[1])
                                neu_score = float(scores[2])
                                lexicon[word] = {
                                    'positive': pos_score,
                                    'negative': neg_score,
                                    'neutral': neu_score
                                }
                            except ValueError:
                                continue
            print(f"Loaded {len(lexicon)} entries from polarity lexicon")
        except FileNotFoundError:
            print(f"Warning: Polarity lexicon file not found at {path}")
            print("Polarity features will be set to 0")
        except Exception as e:
            print(f"Error loading polarity lexicon: {e}")
        
        return lexicon
    
    def _get_pos_tag_mapping(self, upos):
        """Map Universal POS tags to lexicon format (::j, ::n, ::v, ::r)"""
        mapping = {
            'ADJ': 'j',
            'NOUN': 'n',
            'PROPN': 'n',
            'VERB': 'v',
            'ADV': 'r'
        }
        return mapping.get(upos, None)
    
    def extract(self, text):
        """Extract polarity features from text"""
        self.initialize_nlp()
        doc = self.nlp_stanza(text)
        
        features = {}
        
        # Collect polarity scores for all words
        positive_scores = []
        negative_scores = []
        neutral_scores = []
        polarity_coverage = 0
        total_content_words = 0
        
        # Track polarity by POS
        pos_polarity = defaultdict(lambda: {'pos': [], 'neg': [], 'neu': []})
        
        for sent in doc.sentences:
            for word in sent.words:
                # Get lemma and POS tag
                lemma = word.lemma.lower()
                upos = word.upos
                
                # Only process content words
                if upos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                    total_content_words += 1
                    
                    # Map to lexicon format
                    pos_code = self._get_pos_tag_mapping(upos)
                    if pos_code:
                        lexicon_key = f"{lemma}::{pos_code}"
                        
                        # Look up in lexicon
                        if lexicon_key in self.polarity_lexicon:
                            scores = self.polarity_lexicon[lexicon_key]
                            positive_scores.append(scores['positive'])
                            negative_scores.append(scores['negative'])
                            neutral_scores.append(scores['neutral'])
                            polarity_coverage += 1
                            
                            # Track by POS
                            pos_polarity[upos]['pos'].append(scores['positive'])
                            pos_polarity[upos]['neg'].append(scores['negative'])
                            pos_polarity[upos]['neu'].append(scores['neutral'])
        
        # Overall polarity statistics
        features['pol_positive_mean'] = np.mean(positive_scores) if positive_scores else 0
        features['pol_positive_std'] = np.std(positive_scores) if positive_scores else 0
        features['pol_positive_max'] = np.max(positive_scores) if positive_scores else 0
        features['pol_positive_sum'] = np.sum(positive_scores) if positive_scores else 0
        
        features['pol_negative_mean'] = np.mean(negative_scores) if negative_scores else 0
        features['pol_negative_std'] = np.std(negative_scores) if negative_scores else 0
        features['pol_negative_max'] = np.max(negative_scores) if negative_scores else 0
        features['pol_negative_sum'] = np.sum(negative_scores) if negative_scores else 0
        
        features['pol_neutral_mean'] = np.mean(neutral_scores) if neutral_scores else 0
        features['pol_neutral_std'] = np.std(neutral_scores) if neutral_scores else 0
        
        # Polarity ratios
        if positive_scores and negative_scores:
            features['pol_pos_neg_ratio'] = np.mean(positive_scores) / (np.mean(negative_scores) + 0.01)
        else:
            features['pol_pos_neg_ratio'] = 0
        
        # Coverage
        features['pol_coverage'] = polarity_coverage / (total_content_words + 1)
        
        # Polarity by POS type
        for pos_type in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            if pos_polarity[pos_type]['pos']:
                features[f'pol_{pos_type.lower()}_positive_mean'] = np.mean(pos_polarity[pos_type]['pos'])
                features[f'pol_{pos_type.lower()}_negative_mean'] = np.mean(pos_polarity[pos_type]['neg'])
            else:
                features[f'pol_{pos_type.lower()}_positive_mean'] = 0
                features[f'pol_{pos_type.lower()}_negative_mean'] = 0
        
        # Sentiment orientation (dominance of positive vs negative)
        if positive_scores or negative_scores:
            features['pol_sentiment_orientation'] = (np.sum(positive_scores) - np.sum(negative_scores)) / (len(positive_scores) + len(negative_scores) + 1)
        else:
            features['pol_sentiment_orientation'] = 0
        
        # Count highly polarized words
        features['pol_highly_positive_count'] = sum(1 for s in positive_scores if s > 0.7)
        features['pol_highly_negative_count'] = sum(1 for s in negative_scores if s > 0.7)
        
        return features