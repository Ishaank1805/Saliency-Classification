"""
Discriminative N-gram Features for CCG Surface Realization
Based on White & Rajkumar (2009) "Perceptron Reranking for CCG Realization"

Key insight: Count occurrences of each n-gram rather than using 
log-probability as a single feature value.
"""

from collections import defaultdict
from typing import List, Dict, Tuple
import re
import pandas as pd
import nltk
from tqdm.auto import tqdm


class DiscriminativeNgramExtractor:
    """
    Extracts discriminative n-gram features as described in Section 3.
    
    Unlike generative language models that compute log-probabilities,
    discriminative n-gram features count the occurrences of each specific
    n-gram in the candidate realization.
    
    The paper uses a factored language model over:
    - Words
    - Named entity classes (semantic classes)
    - Part-of-speech tags
    - Supertags (CCG categories)
    """
    
    def __init__(self, n: int = 3):
        """
        Args:
            n: Maximum n-gram order (paper uses trigrams, n=3)
        """
        self.n = n
    
    def extract_ngrams(self, sequence: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract all n-grams of order n from a sequence."""
        if n > len(sequence):
            return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def extract_features(self, 
                        words: List[str],
                        pos_tags: List[str] = None,
                        ne_classes: List[str] = None,
                        supertags: List[str] = None) -> Dict[str, int]:
        """
        Extract discriminative n-gram features from a candidate realization.
        
        Args:
            words: List of word tokens
            pos_tags: Part-of-speech tags (optional)
            ne_classes: Named entity classes/semantic classes (optional)
            supertags: CCG lexical categories (optional)
            
        Returns:
            Dictionary mapping feature names to counts
            
        Example:
            words = ["He", "has", "a", "point"]
            pos = ["PRP", "VBZ", "DT", "NN"]
            
            Features include:
            - word_unigram_He: 1
            - word_bigram_He_has: 1
            - word_trigram_He_has_a: 1
            - pos_unigram_PRP: 1
            - pos_bigram_PRP_VBZ: 1
            etc.
        """
        features = defaultdict(int)
        
        # Extract word n-grams (unigrams, bigrams, trigrams)
        for order in range(1, min(self.n + 1, len(words) + 1)):
            ngrams = self.extract_ngrams(words, order)
            for ngram in ngrams:
                feature_name = f"word_{order}gram_{'_'.join(ngram)}"
                features[feature_name] += 1
        
        # Extract POS tag n-grams if provided
        if pos_tags:
            for order in range(1, min(self.n + 1, len(pos_tags) + 1)):
                ngrams = self.extract_ngrams(pos_tags, order)
                for ngram in ngrams:
                    feature_name = f"pos_{order}gram_{'_'.join(ngram)}"
                    features[feature_name] += 1
        
        # Extract named entity class n-grams if provided
        # (Used to handle data sparsity for rare words in same semantic class)
        if ne_classes:
            for order in range(1, min(self.n + 1, len(ne_classes) + 1)):
                ngrams = self.extract_ngrams(ne_classes, order)
                for ngram in ngrams:
                    feature_name = f"ne_{order}gram_{'_'.join(ngram)}"
                    features[feature_name] += 1
        
        # Extract supertag n-grams if provided
        # (CCG lexical categories as fine-grained syntactic labels)
        if supertags:
            for order in range(1, min(self.n + 1, len(supertags) + 1)):
                ngrams = self.extract_ngrams(supertags, order)
                for ngram in ngrams:
                    # Sanitize supertags for feature names (contain special chars)
                    sanitized = '_'.join(self._sanitize(t) for t in ngram)
                    feature_name = f"supertag_{order}gram_{sanitized}"
                    features[feature_name] += 1
        
        return dict(features)
    
    def _sanitize(self, text: str) -> str:
        """Sanitize text for use in feature names."""
        return re.sub(r'[^a-zA-Z0-9]', '', text)


class PerceptronReranker:
    """
    Averaged perceptron model for reranking candidate realizations.
    
    Combines three types of features (as in the paper):
    1. N-gram log probabilities (from generative LM) - baseline features
    2. Syntactic features (from CCG derivation)
    3. Discriminative n-gram features (THIS MODULE)
    """
    
    def __init__(self):
        self.weights = defaultdict(float)
        self.weight_sum = defaultdict(float)  # For averaging
        self.updates = 0
        
    def score(self, features: Dict[str, float]) -> float:
        """Compute score for a candidate using current weights."""
        return sum(self.weights[f] * v for f, v in features.items())
    
    def predict(self, candidates: List[Dict[str, float]]) -> int:
        """Return index of highest-scoring candidate."""
        scores = [self.score(feat) for feat in candidates]
        return max(range(len(scores)), key=lambda i: scores[i])
    
    def update(self, correct_features: Dict[str, float], 
               predicted_features: Dict[str, float]):
        """
        Perceptron update rule:
        α = α + Φ(x_i, y_i) - Φ(x_i, z_i)
        
        where y_i is oracle-best and z_i is predicted
        """
        # Add weight for correct features
        for feat, val in correct_features.items():
            self.weights[feat] += val
            self.weight_sum[feat] += val
            
        # Subtract weight for incorrect prediction
        for feat, val in predicted_features.items():
            self.weights[feat] -= val
            self.weight_sum[feat] -= val
            
        self.updates += 1
    
    def average_weights(self):
        """Compute averaged weights (after training)."""
        if self.updates > 0:
            self.weights = {f: self.weight_sum[f] / self.updates 
                          for f in self.weight_sum}


def add_ngram_features(df: pd.DataFrame, text_col: str = "scene_text", n: int = 3) -> pd.DataFrame:
    """
    Add discriminative n-gram features to a DataFrame.
    
    This function extracts word n-grams and POS tag n-grams from text and
    aggregates them into summary statistics suitable for regression/classification.
    
    Args:
        df: Input DataFrame
        text_col: Column containing text to extract features from
        n: Maximum n-gram order (default: 3 for trigrams)
    
    Returns:
        DataFrame with added n-gram feature columns
    """
    df = df.copy()
    extractor = DiscriminativeNgramExtractor(n=n)
    
    # Extract features for each text
    unique_unigrams_list = []
    unique_bigrams_list = []
    unique_trigrams_list = []
    total_unigrams_list = []
    total_bigrams_list = []
    total_trigrams_list = []
    
    for text in tqdm(df[text_col].tolist(), desc="N-gram features"):
        # Tokenize
        words = nltk.word_tokenize(text.lower())
        
        # Extract features
        features = extractor.extract_features(words)
        
        # Count unique and total n-grams of each order
        unigrams = {k: v for k, v in features.items() if k.startswith("word_1gram_")}
        bigrams = {k: v for k, v in features.items() if k.startswith("word_2gram_")}
        trigrams = {k: v for k, v in features.items() if k.startswith("word_3gram_")}
        
        unique_unigrams_list.append(len(unigrams))
        unique_bigrams_list.append(len(bigrams))
        unique_trigrams_list.append(len(trigrams))
        total_unigrams_list.append(sum(unigrams.values()))
        total_bigrams_list.append(sum(bigrams.values()))
        total_trigrams_list.append(sum(trigrams.values()))
    
    # Add as columns
    df["ngram_unique_unigrams"] = unique_unigrams_list
    df["ngram_unique_bigrams"] = unique_bigrams_list
    df["ngram_unique_trigrams"] = unique_trigrams_list
    df["ngram_total_unigrams"] = total_unigrams_list
    df["ngram_total_bigrams"] = total_bigrams_list
    df["ngram_total_trigrams"] = total_trigrams_list
    
    # Compute diversity ratios
    df["ngram_unigram_diversity"] = df["ngram_unique_unigrams"] / df["ngram_total_unigrams"].replace(0, 1)
    df["ngram_bigram_diversity"] = df["ngram_unique_bigrams"] / df["ngram_total_bigrams"].replace(0, 1)
    df["ngram_trigram_diversity"] = df["ngram_unique_trigrams"] / df["ngram_total_trigrams"].replace(0, 1)
    
    return df


# Example usage demonstrating the key difference
def compare_feature_types():
    """
    Demonstrates the difference between:
    1. Log-probability features (single feature with LM score)
    2. Discriminative n-gram features (one feature per n-gram)
    """
    
    extractor = DiscriminativeNgramExtractor(n=3)
    
    # Example sentence from the paper
    words = ["He", "has", "a", "point", "he", "wants", "to", "make"]
    pos = ["PRP", "VBZ", "DT", "NN", "PRP", "VBZ", "TO", "VB"]
    
    print("=" * 70)
    print("DISCRIMINATIVE N-GRAM FEATURES")
    print("=" * 70)
    print("\nExample sentence:", " ".join(words))
    print()
    
    # Extract discriminative n-gram features
    features = extractor.extract_features(words, pos_tags=pos)
    
    print(f"Total features extracted: {len(features)}")
    print("\nSample word features:")
    for feat, count in list(features.items())[:10]:
        print(f"  {feat}: {count}")
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES FROM LOG-PROBABILITY FEATURES")
    print("=" * 70)
    print("""
1. LOG-PROBABILITY FEATURES (baseline):
   - Single feature: "lm_logprob" = -45.23
   - One value representing overall sequence probability
   
2. DISCRIMINATIVE N-GRAM FEATURES (this implementation):
   - Separate feature for EACH n-gram: 
     * "word_1gram_He": 1
     * "word_2gram_He_has": 1  
     * "word_3gram_He_has_a": 1
     * ... (100s-1000s of features)
   - Counts allow learning specific weights for each n-gram
   
3. BENEFIT (from paper):
   "Discriminative training with n-gram features has the potential 
   to learn to negatively weight n-grams that appear in some of the 
   GEN(x_i) candidates, but which never appear in the naturally 
   occurring corpus used to train a standard, generative language model."
   
   Example: The model can learn that "video-viewing small" is bad
            even if the LM assigns it reasonable probability.
    """)
    
    return features


def demonstration_with_reranking():
    """
    Demonstrates how discriminative n-gram features are used in reranking.
    """
    print("\n" + "=" * 70)
    print("RERANKING EXAMPLE")
    print("=" * 70)
    
    extractor = DiscriminativeNgramExtractor(n=2)  # Using bigrams for clarity
    
    # Two candidate realizations for the same logical form
    # (from Table 6 in the paper)
    candidate1_words = ["Taipei", "'s", "growing", "number", "of", 
                       "video-viewing", "small", "parlors"]
    candidate2_words = ["Taipei", "'s", "growing", "number", "of", 
                       "small", "video-viewing", "parlors"]
    
    # Extract features for both
    feat1 = extractor.extract_features(candidate1_words)
    feat2 = extractor.extract_features(candidate2_words)
    
    print("\nCandidate 1 (incorrect word order):")
    print(" ".join(candidate1_words))
    print(f"Features: {len(feat1)}")
    
    print("\nCandidate 2 (correct word order):")
    print(" ".join(candidate2_words))
    print(f"Features: {len(feat2)}")
    
    # Show discriminating features
    print("\nKey discriminating bigrams:")
    print("  Candidate 1 has: 'word_2gram_video-viewing_small'")
    print("  Candidate 2 has: 'word_2gram_small_video-viewing'")
    print("\n  → Perceptron can learn negative weight for the first,")
    print("    positive weight for the second")


if __name__ == "__main__":
    # Run demonstrations
    features = compare_feature_types()
    demonstration_with_reranking()
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION NOTES")
    print("=" * 70)
    print("""
This implementation demonstrates discriminative n-gram features from:
White & Rajkumar (2009), Section 3, pages 5-6

Key points from the paper:
1. Features count n-gram occurrences rather than using log-probs
2. Applied to factored LM: words, NE classes, POS tags, supertags
3. Combined with log-prob features and syntactic features
4. Trained using averaged perceptron (Collins 2002)
5. Achieved 0.8506 BLEU on CCGbank Section 23

Training details (Table 2):
- Full model: 576,176 features (from 2.4M alphabet)
- 10 iterations, ~9 hours training time
- 96.40% training accuracy
    """)
