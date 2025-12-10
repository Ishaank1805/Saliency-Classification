from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import spacy
except Exception:
    spacy = None  # type: ignore


PRONOUNS = {
    "he", "she", "they", "him", "her", "them",
    "his", "hers", "theirs", "himself", "herself", "themselves",
}


def _load_spacy_model():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            return spacy.load("en_core_web_md")
        except Exception:
            return None


def add_entity_character_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    df = df.copy()
    nlp = _load_spacy_model()
    unique_person_count: List[float] = []
    top_char_rate: List[float] = []
    pronoun_ratio: List[float] = []
    name_repetition_rate: List[float] = []

    texts = df[text_col].tolist()
    if nlp is None:
        # Fallback: zeros when spaCy NER is unavailable
        for t in tqdm(texts, desc="Entity (fallback)"):
            words = t.split()
            wc = max(1, len(words))
            unique_person_count.append(0.0)
            top_char_rate.append(0.0)
            pronoun_ratio.append(float(sum(1 for w in words if w.lower() in PRONOUNS)) / wc)
            name_repetition_rate.append(0.0)
        df["unique_PERSON_count"] = unique_person_count
        df["top_character_mention_rate"] = top_char_rate
        df["pronoun_ratio"] = pronoun_ratio
        df["name_repetition_rate"] = name_repetition_rate
        return df

    for doc in tqdm(nlp.pipe(texts, batch_size=64, disable=[]), total=len(texts), desc="Entity/Character"):
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        total_mentions = len(persons)
        norm_names = [p.strip().lower() for p in persons if p.strip()]
        unique_names = set(norm_names)
        unique_person_count.append(float(len(unique_names)))
        if total_mentions > 0:
            # top character mention rate: max name freq / total mentions
            counts = {}
            for n in norm_names:
                counts[n] = counts.get(n, 0) + 1
            top = max(counts.values())
            top_char_rate.append(float(top) / float(total_mentions))
            name_repetition_rate.append(float(total_mentions - len(unique_names)) / float(total_mentions))
        else:
            top_char_rate.append(0.0)
            name_repetition_rate.append(0.0)

        # pronoun ratio: pronoun tokens / total tokens
        tokens = [t.text for t in doc if not t.is_space]
        wc = max(1, len(tokens))
        prons = sum(1 for t in tokens if t.lower() in PRONOUNS or (t.isalpha() and getattr(getattr(t, "pos_", None), "lower", lambda: "")() == "pron"))
        pronoun_ratio.append(float(prons) / float(wc))

    df["unique_PERSON_count"] = unique_person_count
    df["top_character_mention_rate"] = top_char_rate
    df["pronoun_ratio"] = pronoun_ratio
    df["name_repetition_rate"] = name_repetition_rate
    return df


def add_character_arc_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """
    Add character transition features across movie timeline.
    
    NEW FEATURES (high impact):
    - char_new_introductions: First appearance in movie (setup)
    - char_returns: Previously seen but not in last scene (payoff!)
    - char_callbacks: Characters from opening scene return
    - char_turnover: How much character lineup changed vs prev scene
    
    These capture: "Oh! That character from Act 1 is back!" moments.
    """
    df = df.copy()
    nlp = _load_spacy_model()
    
    # Initialize columns
    df["char_new_introductions"] = 0.0
    df["char_returns"] = 0.0
    df["char_callbacks"] = 0.0
    df["char_turnover"] = 0.0
    
    if nlp is None:
        return df
    
    # Process each movie separately to track character history
    for movie_id, movie_group in tqdm(df.groupby("movie_id"), desc="Character arcs"):
        # Sort by scene index to process chronologically
        movie_df = movie_group.sort_values("scene_index").copy()
        indices = movie_df.index.tolist()
        
        # Extract all characters for all scenes in this movie
        all_scene_chars = []
        for idx in indices:
            text = movie_df.loc[idx, text_col]
            doc = nlp(text[:1000])  # Limit length for speed
            chars = {ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"}
            all_scene_chars.append(chars)
        
        # Track all characters seen so far (cumulative)
        all_previous_chars = set()
        first_scene_chars = all_scene_chars[0] if all_scene_chars else set()
        
        # Process each scene
        for i, idx in enumerate(indices):
            current_chars = all_scene_chars[i]
            prev_chars = all_scene_chars[i-1] if i > 0 else set()
            
            # 1. NEW INTRODUCTIONS: Characters appearing for first time
            new_chars = current_chars - all_previous_chars
            df.loc[idx, "char_new_introductions"] = float(len(new_chars))
            
            # 2. RETURNS: Characters seen before but not in previous scene
            # This captures "payoff" moments - character from Act 1 returns!
            returning_chars = current_chars & (all_previous_chars - prev_chars)
            df.loc[idx, "char_returns"] = float(len(returning_chars))
            
            # 3. CALLBACKS: Characters from opening scene
            # Captures structural callbacks to beginning
            callback_chars = current_chars & first_scene_chars
            df.loc[idx, "char_callbacks"] = float(len(callback_chars))
            
            # 4. TURNOVER: How much did character lineup change?
            # High turnover = scene transition / location change
            if prev_chars or current_chars:
                left = len(prev_chars - current_chars)
                joined = len(current_chars - prev_chars)
                total = len(prev_chars | current_chars)
                turnover = (left + joined) / max(1, total)
            else:
                turnover = 0.0
            df.loc[idx, "char_turnover"] = float(turnover)
            
            # Update cumulative set for next iteration
            all_previous_chars |= current_chars
    
    return df


