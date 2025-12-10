from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import nltk
from tqdm.auto import tqdm

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

# Ensure punkt for sentence tokenization used later for surprisal consistency
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

EN_STOPWORDS = set(stopwords.words("english"))


def _tokenize_words(text: str) -> List[str]:
    return [w for w in nltk.word_tokenize(text) if any(c.isalpha() for c in w)]


def type_token_ratio(text: str, exclude_function_words: bool = False) -> float:
    tokens = _tokenize_words(text)
    if exclude_function_words:
        tokens = [t for t in tokens if t.lower() not in EN_STOPWORDS]
    if not tokens:
        return 0.0
    unique = set(t.lower() for t in tokens)
    return float(len(unique)) / float(len(tokens))


def add_ttr_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    df = df.copy()
    df["ttr"] = [
        type_token_ratio(x, exclude_function_words=False)
        for x in tqdm(df[text_col].tolist(), desc="TTR")
    ]
    # Drop ttr_no_func per pruning suggestion
    return df


def _sentence_tokenize(text: str) -> List[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [text]


def add_length_structure_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    df = df.copy()
    texts = df[text_col].tolist()
    sent_counts: List[int] = []
    token_counts: List[int] = []
    avg_sent_len: List[float] = []
    var_sent_len: List[float] = []
    exclaim_rate: List[float] = []
    question_rate: List[float] = []
    uppercase_ratio: List[float] = []
    dialogue_ratio: List[float] = []
    for t in tqdm(texts, desc="Len/Struct"):
        sents = _sentence_tokenize(t or "")
        sents = [s for s in sents if s.strip()]
        sent_counts.append(len(sents))
        words = _tokenize_words(t or "")
        token_counts.append(len(words))
        slens = [len(_tokenize_words(s)) for s in sents] or [0]
        avg_sent_len.append(float(np.mean(slens)))
        var_sent_len.append(float(np.var(slens)))
        text_len = max(1, len(t))
        exclaim_rate.append(t.count("!") / text_len)
        question_rate.append(t.count("?") / text_len)
        upper = sum(1 for c in t if c.isupper())
        letters = sum(1 for c in t if c.isalpha())
        uppercase_ratio.append((upper / letters) if letters else 0.0)
        # simple dialogue heuristic: lines starting with uppercase word + colon, or quoted text
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        dialogue_lines = [ln for ln in lines if (":" in ln.split(" ")[0] if ln.split(" ") else False) or ("\"" in ln or "'" in ln)]
        dialogue_ratio.append((len(dialogue_lines) / len(lines)) if lines else 0.0)
    df["sentence_count"] = sent_counts
    df["token_count"] = token_counts
    df["avg_sentence_len"] = avg_sent_len
    df["var_sentence_len"] = var_sent_len
    df["exclaim_rate"] = exclaim_rate
    df["question_rate"] = question_rate
    df["uppercase_ratio"] = uppercase_ratio
    df["dialogue_ratio"] = dialogue_ratio
    return df


def add_position_overlap_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # position
    if "scene_index" in df.columns:
        by_movie = df.groupby("movie_id")["scene_index"].transform("max").replace(0, 1)
        df["scene_index_norm"] = df["scene_index"] / by_movie
    else:
        df["scene_index_norm"] = 0.0
    # lexical overlap with neighbors (Jaccard)
    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return float(len(a & b)) / float(len(a | b))
    prev_overlap: List[float] = []
    next_overlap: List[float] = []
    for i in tqdm(range(len(df)), desc="Overlap"):
        cur = set(w.lower() for w in _tokenize_words(df.loc[i, "scene_text"]) if w)
        # find prev/next within same movie
        prev_set: set = set()
        next_set: set = set()
        if i > 0 and df.loc[i - 1, "movie_id"] == df.loc[i, "movie_id"]:
            prev_set = set(w.lower() for w in _tokenize_words(df.loc[i - 1, "scene_text"]))
        if i + 1 < len(df) and df.loc[i + 1, "movie_id"] == df.loc[i, "movie_id"]:
            next_set = set(w.lower() for w in _tokenize_words(df.loc[i + 1, "scene_text"]))
        prev_overlap.append(jaccard(cur, prev_set))
        next_overlap.append(jaccard(cur, next_set))
    df["overlap_prev"] = prev_overlap
    df["overlap_next"] = next_overlap
    return df


def add_similarity_change_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """
    Detect plot shifts through changes in similarity patterns.
    
    NEW FEATURES:
    - sim_change_magnitude: How different is this vs the trend?
    - vocab_novelty: New words appearing (reveals/plot twists)
    - dialogue_shift: Sudden change in dialogue amount
    
    Key insight: Salient scenes break from the pattern.
    """
    df = df.copy()
    
    # Initialize columns
    df["sim_change_magnitude"] = 0.0
    df["vocab_novelty"] = 0.0
    df["dialogue_shift"] = 0.0
    
    # Process each movie separately
    for movie_id, movie_group in tqdm(df.groupby("movie_id"), desc="Similarity changes"):
        movie_df = movie_group.sort_values("scene_index").copy()
        indices = movie_df.index.tolist()
        
        # Pre-tokenize all texts
        all_texts = []
        all_words = []
        all_dialogue_counts = []
        
        for idx in indices:
            text = movie_df.loc[idx, text_col]
            words = set(w.lower() for w in _tokenize_words(text))
            dialogue_count = text.count('"') + text.count("'")
            
            all_texts.append(text)
            all_words.append(words)
            all_dialogue_counts.append(dialogue_count)
        
        # Process each scene with context
        for i, idx in enumerate(indices):
            curr_words = all_words[i]
            curr_text = all_texts[i]
            curr_dialogue = all_dialogue_counts[i]
            
            # 1. SIMILARITY CHANGE MAGNITUDE
            # Compare current similarity with baseline trend
            if i >= 2:
                prev_words = all_words[i-1]
                prev_prev_words = all_words[i-2]
                
                # What was the "normal" similarity between prev scenes?
                baseline_sim = _jaccard_similarity(prev_words, prev_prev_words)
                
                # How similar is current to previous?
                current_sim = _jaccard_similarity(curr_words, prev_words)
                
                # Big change = plot shift / reveal moment
                change = abs(current_sim - baseline_sim)
                df.loc[idx, "sim_change_magnitude"] = float(change)
            
            # 2. VOCABULARY NOVELTY
            # New words = new concepts = reveals/twists
            if i > 0:
                prev_words = all_words[i-1]
                new_words = curr_words - prev_words
                novelty = len(new_words) / max(1, len(curr_words))
                df.loc[idx, "vocab_novelty"] = float(novelty)
            
            # 3. DIALOGUE SHIFT
            # Sudden change in dialogue amount
            if i > 0:
                prev_dialogue = all_dialogue_counts[i-1]
                text_length = max(1, len(curr_text.split()))
                shift = abs(curr_dialogue - prev_dialogue) / text_length
                df.loc[idx, "dialogue_shift"] = float(shift)
    
    return df


def add_structural_position_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """
    Add three-act structure and thematic callback features.
    
    NEW FEATURES:
    - pos_edge_proximity: U-shaped importance (beginning/end salient)
    - pos_act: Which act (1=setup, 2=conflict, 3=resolution)
    - pos_within_act: Position within current act
    - callback_to_opening: Thematic return to opening
    - callback_to_ending: Foreshadowing of ending
    
    Key insight: Position matters. Edges and callbacks are salient.
    """
    df = df.copy()
    
    # Initialize columns
    df["pos_edge_proximity"] = 0.0
    df["pos_act"] = 1.0
    df["pos_within_act"] = 0.0
    df["callback_to_opening"] = 0.0
    df["callback_to_ending"] = 0.0
    
    # Process each movie separately
    for movie_id, movie_group in tqdm(df.groupby("movie_id"), desc="Structural position"):
        movie_df = movie_group.sort_values("scene_index").copy()
        indices = movie_df.index.tolist()
        total_scenes = len(indices)
        
        if total_scenes == 0:
            continue
        
        # Get opening and ending text
        first_text = movie_df.loc[indices[0], text_col]
        last_text = movie_df.loc[indices[-1], text_col]
        first_words = set(w.lower() for w in _tokenize_words(first_text))
        last_words = set(w.lower() for w in _tokenize_words(last_text))
        
        # Process each scene
        for i, idx in enumerate(indices):
            # Normalized position (0 = start, 1 = end)
            pos_norm = i / max(1, total_scenes - 1)
            
            # 1. EDGE PROXIMITY (U-shaped curve)
            # Maximum at edges (0 and 1), minimum at middle (0.5)
            # Formula: 1 - 2*|pos - 0.5|
            edge_proximity = 1.0 - 2.0 * abs(pos_norm - 0.5)
            df.loc[idx, "pos_edge_proximity"] = float(edge_proximity)
            
            # 2. THREE-ACT STRUCTURE
            # Act 1: 0-25% (setup)
            # Act 2: 25-75% (confrontation)
            # Act 3: 75-100% (resolution)
            if pos_norm < 0.25:
                act = 1
                within_act = pos_norm / 0.25
            elif pos_norm < 0.75:
                act = 2
                within_act = (pos_norm - 0.25) / 0.5
            else:
                act = 3
                within_act = (pos_norm - 0.75) / 0.25
            
            df.loc[idx, "pos_act"] = float(act)
            df.loc[idx, "pos_within_act"] = float(within_act)
            
            # 3. THEMATIC CALLBACKS
            # Similarity with opening/ending = structural callbacks
            curr_text = movie_df.loc[idx, text_col]
            curr_words = set(w.lower() for w in _tokenize_words(curr_text))
            
            callback_opening = _jaccard_similarity(curr_words, first_words)
            callback_ending = _jaccard_similarity(curr_words, last_words)
            
            df.loc[idx, "callback_to_opening"] = float(callback_opening)
            df.loc[idx, "callback_to_ending"] = float(callback_ending)
    
    return df


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


