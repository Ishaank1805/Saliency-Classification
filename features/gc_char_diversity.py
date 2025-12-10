import numpy as np
import math
from collections import Counter
from .gc_base import BaseFeatureExtractor


class CharDiversityFeatureExtractor(BaseFeatureExtractor):
    """Extract character-level diversity measures"""
    
    def extract(self, text):
        """Extract all character diversity features"""
        chars = list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        if not chars:
            return {
                'cd_ttr': 0, 'cd_maas': 0, 'cd_msttr': 0, 'cd_mattr': 0,
                'cd_mtld': 0, 'cd_mtld_ma': 0, 'cd_yulesk': 0, 'cd_vocd': 0
            }
        
        features = {}
        
        # Type-Token Ratio
        unique_chars = set(chars)
        features['cd_ttr'] = len(unique_chars) / len(chars)
        
        # MAAS TTR
        N = len(chars)
        V = len(unique_chars)
        if N > 0 and V > 0:
            features['cd_maas'] = (math.log(N) - math.log(V)) / (math.log(N) ** 2)
        else:
            features['cd_maas'] = 0
        
        # MSTTR (Mean Segmental TTR)
        features['cd_msttr'] = self._calculate_msttr(chars)
        
        # MATTR (Moving Average TTR)
        features['cd_mattr'] = self._calculate_mattr(chars)
        
        # MTLD
        features['cd_mtld'] = self._calculate_mtld(chars)
        features['cd_mtld_ma'] = self._calculate_mtld_ma(chars)
        
        # Yule's K
        features['cd_yulesk'] = self._calculate_yules_k(chars)
        
        # VocD
        features['cd_vocd'] = self._calculate_vocd_simplified(chars)
        
        return features
    
    def _calculate_msttr(self, chars, segment_size=100):
        """Calculate Mean Segmental Type-Token Ratio"""
        segments = [chars[i:i+segment_size] for i in range(0, len(chars), segment_size)]
        segment_ttrs = []
        
        for segment in segments:
            if len(segment) >= 10:
                seg_unique = len(set(segment))
                seg_total = len(segment)
                segment_ttrs.append(seg_unique / seg_total)
        
        return np.mean(segment_ttrs) if segment_ttrs else len(set(chars)) / len(chars)
    
    def _calculate_mattr(self, chars, window_size=50):
        """Calculate Moving Average Type-Token Ratio"""
        window_size = min(window_size, len(chars))
        
        if window_size > 0 and len(chars) >= window_size:
            mattrs = []
            for i in range(len(chars) - window_size + 1):
                window = chars[i:i+window_size]
                window_unique = len(set(window))
                mattrs.append(window_unique / window_size)
            return np.mean(mattrs)
        else:
            return len(set(chars)) / len(chars)
    
    def _calculate_mtld(self, tokens, ttr_threshold=0.72):
        """Calculate Measure of Textual Lexical Diversity"""
        if not tokens:
            return 0
            
        def _compute_factor(tokens, ttr_threshold):
            factors = 0.0
            start = 0
            
            for i in range(1, len(tokens) + 1):
                segment = tokens[start:i]
                unique = len(set(segment))
                total = len(segment)
                ttr = unique / total if total > 0 else 0
                
                if ttr < ttr_threshold and i < len(tokens):
                    continue
                else:
                    factors += 1
                    start = i
                    
            if start < len(tokens):
                remaining_length = len(tokens) - start
                remaining_ttr = len(set(tokens[start:])) / remaining_length
                factors += remaining_ttr / ttr_threshold
                
            return factors
        
        forward_factor = _compute_factor(tokens, ttr_threshold)
        backward_factor = _compute_factor(tokens[::-1], ttr_threshold)
        
        if forward_factor > 0 and backward_factor > 0:
            mtld = len(tokens) / ((forward_factor + backward_factor) / 2)
        else:
            mtld = 0
            
        return mtld
    
    def _calculate_mtld_ma(self, tokens, window_size=50):
        """Calculate moving average MTLD"""
        if len(tokens) < window_size:
            return self._calculate_mtld(tokens)
            
        mtlds = []
        step = max(1, window_size // 2)
        
        for i in range(0, len(tokens) - window_size + 1, step):
            window = tokens[i:i+window_size]
            mtld_value = self._calculate_mtld(window)
            if mtld_value > 0:
                mtlds.append(mtld_value)
                
        return np.mean(mtlds) if mtlds else 0
    
    def _calculate_yules_k(self, chars):
        """Calculate Yule's K statistic"""
        char_freq = Counter(chars)
        M1 = len(chars)
        M2 = sum([freq * freq for freq in char_freq.values()])
        return 10000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0
    
    def _calculate_vocd_simplified(self, tokens):
        """Simplified VocD calculation"""
        if len(tokens) < 35:
            return 0
            
        sample_sizes = [35, 50, 70, 100]
        ttrs = []
        
        for size in sample_sizes:
            if size <= len(tokens):
                sample_ttrs = []
                for _ in range(10):
                    sample_indices = np.random.choice(len(tokens), size, replace=False)
                    sample = [tokens[i] for i in sample_indices]
                    ttr = len(set(sample)) / len(sample)
                    sample_ttrs.append(ttr)
                ttrs.append(np.mean(sample_ttrs))
        
        if len(ttrs) >= 2:
            vocd = np.mean(ttrs) * 100
        else:
            vocd = 0
            
        return vocd