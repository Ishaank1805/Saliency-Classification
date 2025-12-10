import numpy as np
from collections import defaultdict
from .gc_base import BaseFeatureExtractor


class SyntaxFeatureExtractor(BaseFeatureExtractor):
    """Extract syntactic and dependency features"""
    
    def extract(self, text):
        """Extract all syntactic features"""
        self.initialize_nlp()
        doc = self.nlp_stanza(text)
        
        features = {}
        
        # Dependency relations
        features.update(self._extract_dependency_features(doc))
        
        # Argument/Adjunct ratio
        features.update(self._calculate_argument_adjunct_ratio(doc))
        
        # Syntactic complexity
        features.update(self._calculate_syntactic_complexity(doc))
        
        # Dependency bigrams
        features.update(self._extract_dependency_bigrams(doc))
        
        return features
    
    def _extract_dependency_features(self, doc):
        """Extract dependency relation frequencies"""
        dep_counts = defaultdict(int)
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.deprel:
                    dep_counts[word.deprel] += 1
        
        total_deps = sum(dep_counts.values())
        dep_features = {}
        
        important_deps = ['nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 
                         'obl', 'vocative', 'expl', 'dislocated', 'advcl', 
                         'advmod', 'discourse', 'aux', 'cop', 'mark', 'nmod', 
                         'appos', 'nummod', 'acl', 'amod', 'det', 'clf', 
                         'case', 'conj', 'cc', 'fixed', 'flat', 'compound',
                         'list', 'parataxis', 'orphan', 'goeswith', 'reparandum',
                         'punct', 'root', 'dep']
        
        for dep in important_deps:
            dep_features[f'dep_rel_{dep}'] = dep_counts.get(dep, 0) / (total_deps + 1)
        
        return dep_features
    
    def _calculate_argument_adjunct_ratio(self, doc):
        """Calculate argument/adjunct ratio"""
        arguments = 0
        adjuncts = 0
        
        arg_rels = {'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp'}
        adj_rels = {'obl', 'advmod', 'advcl', 'nmod', 'amod'}
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.deprel in arg_rels:
                    arguments += 1
                elif word.deprel in adj_rels:
                    adjuncts += 1
        
        ratio = arguments / (adjuncts + 1)
        return {'arg_adj_ratio': ratio}
    
    def _calculate_syntactic_complexity(self, doc):
        """Calculate syntactic complexity measures"""
        features = {}
        
        # Tree depth
        max_depths = []
        for sent in doc.sentences:
            depths = self._calculate_tree_depths(sent)
            if depths:
                max_depths.append(max(depths))
        
        features['syn_comp_depth_mean'] = np.mean(max_depths) if max_depths else 0
        features['syn_comp_depth_std'] = np.std(max_depths) if max_depths else 0
        
        # Dependency distances
        dep_distances = []
        for sent in doc.sentences:
            distances = self._calculate_dependency_distances(sent)
            dep_distances.extend(distances)
        
        features['syn_comp_add_mean'] = np.mean(dep_distances) if dep_distances else 0
        features['syn_comp_add_std'] = np.std(dep_distances) if dep_distances else 0
        
        # Index of Syntactic Complexity
        isc_values = []
        for sent in doc.sentences:
            isc = self._calculate_isc(sent)
            isc_values.append(isc)
        
        features['syn_comp_isc_mean'] = np.mean(isc_values) if isc_values else 0
        features['syn_comp_isc_std'] = np.std(isc_values) if isc_values else 0
        
        return features
    
    def _calculate_tree_depths(self, sentence):
        """Calculate depth of each node in dependency tree"""
        depths = []
        
        def get_depth(word_id, current_depth=0):
            depths.append(current_depth)
            for word in sentence.words:
                if word.head == word_id:
                    get_depth(word.id, current_depth + 1)
        
        for word in sentence.words:
            if word.deprel == 'root':
                get_depth(word.id)
                break
                
        return depths
    
    def _calculate_dependency_distances(self, sentence):
        """Calculate dependency distances"""
        distances = []
        
        for word in sentence.words:
            if word.head > 0:
                distance = abs(word.id - word.head)
                distances.append(distance)
                
        return distances
    
    def _calculate_isc(self, sentence):
        """Calculate Index of Syntactic Complexity"""
        subordinating_conj = sum(1 for w in sentence.words if w.upos == 'SCONJ')
        wh_words = sum(1 for w in sentence.words if w.text.lower() in 
                      ['who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how'])
        verb_forms = sum(1 for w in sentence.words if w.upos == 'VERB')
        
        total_words = len(sentence.words)
        
        if total_words > 0:
            isc = (subordinating_conj + wh_words + verb_forms) / total_words
        else:
            isc = 0
            
        return isc
    
    def _extract_dependency_bigrams(self, doc):
        """Extract dependency bigrams with word order information"""
        bigram_counts = defaultdict(int)
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.head > 0:
                    head_word = sent.words[word.head - 1]
                    
                    dep_pos = word.upos
                    head_pos = head_word.upos
                    
                    if word.id < word.head:
                        bigram = f'{dep_pos}_before_{head_pos}'
                    else:
                        bigram = f'{dep_pos}_after_{head_pos}'
                    
                    bigram_counts[bigram] += 1
        
        total_bigrams = sum(bigram_counts.values())
        bigram_features = {}
        
        for bigram, count in bigram_counts.items():
            bigram_features[f'dep_big_{bigram}'] = count / (total_bigrams + 1)
        
        return bigram_features