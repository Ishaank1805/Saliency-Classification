import numpy as np
from collections import defaultdict
from .gc_base import BaseFeatureExtractor
from huggingface_hub import hf_hub_download


class ConcretenessFeatureExtractor(BaseFeatureExtractor):
    """Extract concreteness features using Brysbaert concreteness ratings"""
    
    def __init__(self, lexicon_path=None):
        """
        Initialize with path to concreteness ratings file.
        
        Parameters:
        -----------
        lexicon_path : str, optional
            Path to the Brysbaert concreteness ratings file.
            If None, downloads from HuggingFace Hub.
        """
        super().__init__()
        
        # Download from HuggingFace if no path provided
        if lexicon_path is None:
            print("Downloading concreteness lexicon from HuggingFace Hub...")
            lexicon_path = hf_hub_download(
                repo_id="Ishaank18/screenplay-lexicons",
                filename="Concreteness_ratings_Brysbaert_et_al_BRM.txt",
                repo_type="dataset"
            )
            print(f"✓ Downloaded to: {lexicon_path}")
        
        self.concreteness_lexicon = self._load_lexicon(lexicon_path)
    
    def _load_lexicon(self, path):
        """Load concreteness lexicon from file"""
        lexicon = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Skip header line
                header = f.readline()
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    # Format: Word, Bigram, Conc.M, Conc.SD, Unknown, Total, Percent_known, SUBTLEX, Dom_Pos
                    if len(parts) >= 9:
                        word = parts[0].strip().lower()
                        bigram = parts[1].strip()
                        
                        try:
                            conc_mean = float(parts[2])
                            conc_sd = float(parts[3])
                            
                            lexicon[word] = {
                                'mean': conc_mean,
                                'sd': conc_sd,
                                'bigram': bigram
                            }
                        except ValueError:
                            continue
            
            print(f"Loaded {len(lexicon)} entries from concreteness lexicon")
        except FileNotFoundError:
            print(f"Warning: Concreteness lexicon file not found at {path}")
            print("Concreteness features will be set to 0")
        except Exception as e:
            print(f"Error loading concreteness lexicon: {e}")
        
        return lexicon
    
    def extract(self, text):
        """Extract concreteness features from text"""
        self.initialize_nlp()
        doc = self.nlp_stanza(text)
        
        features = {}
        
        # Collect concreteness scores
        concreteness_scores = []
        concreteness_sds = []
        coverage_count = 0
        total_words = 0
        
        # Track by POS
        pos_concreteness = defaultdict(list)
        
        for sent in doc.sentences:
            for word in sent.words:
                lemma = word.lemma.lower()
                upos = word.upos
                
                # Only process content words
                if upos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                    total_words += 1
                    
                    # Look up in lexicon (try both lemma and original word)
                    word_text = word.text.lower()
                    
                    conc_data = None
                    if lemma in self.concreteness_lexicon:
                        conc_data = self.concreteness_lexicon[lemma]
                    elif word_text in self.concreteness_lexicon:
                        conc_data = self.concreteness_lexicon[word_text]
                    
                    if conc_data:
                        concreteness_scores.append(conc_data['mean'])
                        concreteness_sds.append(conc_data['sd'])
                        coverage_count += 1
                        pos_concreteness[upos].append(conc_data['mean'])
        
        # Overall concreteness statistics
        features['conc_mean'] = np.mean(concreteness_scores) if concreteness_scores else 0
        features['conc_std'] = np.std(concreteness_scores) if concreteness_scores else 0
        features['conc_median'] = np.median(concreteness_scores) if concreteness_scores else 0
        features['conc_max'] = np.max(concreteness_scores) if concreteness_scores else 0
        features['conc_min'] = np.min(concreteness_scores) if concreteness_scores else 0
        features['conc_range'] = features['conc_max'] - features['conc_min']
        
        # Average standard deviation (lexical ambiguity indicator)
        features['conc_sd_mean'] = np.mean(concreteness_sds) if concreteness_sds else 0
        
        # Coverage
        features['conc_coverage'] = coverage_count / (total_words + 1)
        
        # Count highly concrete/abstract words
        if concreteness_scores:
            features['conc_highly_concrete_count'] = sum(1 for s in concreteness_scores if s >= 4.5)
            features['conc_highly_abstract_count'] = sum(1 for s in concreteness_scores if s <= 2.5)
            features['conc_highly_concrete_ratio'] = features['conc_highly_concrete_count'] / len(concreteness_scores)
            features['conc_highly_abstract_ratio'] = features['conc_highly_abstract_count'] / len(concreteness_scores)
        else:
            features['conc_highly_concrete_count'] = 0
            features['conc_highly_abstract_count'] = 0
            features['conc_highly_concrete_ratio'] = 0
            features['conc_highly_abstract_ratio'] = 0
        
        # Concreteness by POS type
        for pos_type in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            if pos_concreteness[pos_type]:
                features[f'conc_{pos_type.lower()}_mean'] = np.mean(pos_concreteness[pos_type])
                features[f'conc_{pos_type.lower()}_std'] = np.std(pos_concreteness[pos_type])
            else:
                features[f'conc_{pos_type.lower()}_mean'] = 0
                features[f'conc_{pos_type.lower()}_std'] = 0
        
        # Percentile features
        if concreteness_scores:
            features['conc_25percentile'] = np.percentile(concreteness_scores, 25)
            features['conc_75percentile'] = np.percentile(concreteness_scores, 75)
        else:
            features['conc_25percentile'] = 0
            features['conc_75percentile'] = 0
        
        return features