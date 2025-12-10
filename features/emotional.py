from __future__ import annotations

import numpy as np
import pandas as pd
from transformers import pipeline
import torch
from tqdm.auto import tqdm
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class EmotionalTrajectoryExtractor:
    def __init__(self):
        # Use RoBERTa-based emotion classifier (6 emotions + neutral)
        # This is universal - no word lists, pure deep learning
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def compute_trajectory_features(self, text: str) -> dict:
        """Extract emotional trajectory using transformer model."""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return {
                'emo_shift_magnitude': 0.0,
                'emo_volatility': 0.0,
                'emo_final_vs_initial': 0.0,
                'emo_peak_position': 0.5,
                'emo_dominant_emotion': 0.0
            }
        
        # Get emotion distributions for each sentence
        emotion_sequences = []
        for sent in sentences:
            # Model handles 512 tokens max
            emotions = self.emotion_model(sent[:512])[0]
            # Convert to dict: {anger: 0.1, disgust: 0.05, fear: 0.2, joy: 0.4, ...}
            emotion_dict = {e['label']: e['score'] for e in emotions}
            emotion_sequences.append(emotion_dict)
        
        # Extract trajectory features
        features = {}
        
        # 1. Emotional shift magnitude (most important feature)
        # Compare first third vs last third of scene
        n = len(emotion_sequences)
        first_third = emotion_sequences[:max(1, n//3)]
        last_third = emotion_sequences[-max(1, n//3):]
        
        # Average emotions in each third
        def avg_emotions(emo_list):
            avg = {}
            for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
                values = [e.get(emotion, 0) for e in emo_list]
                avg[emotion] = np.mean(values)
            return avg
        
        first_avg = avg_emotions(first_third)
        last_avg = avg_emotions(last_third)
        
        # Compute shift as euclidean distance in emotion space
        shift = np.sqrt(sum((last_avg[e] - first_avg[e])**2 
                           for e in first_avg.keys()))
        features['emo_shift_magnitude'] = float(shift)
        
        # 2. Emotional volatility (variance across scene)
        all_values = []
        for emotion in ['anger', 'fear', 'joy', 'sadness']:
            values = [e.get(emotion, 0) for e in emotion_sequences]
            all_values.extend(values)
        features['emo_volatility'] = float(np.std(all_values))
        
        # 3. Final vs Initial emotional state
        # Positive if ending more positive than beginning
        initial_joy = emotion_sequences[0].get('joy', 0)
        final_joy = emotion_sequences[-1].get('joy', 0)
        initial_neg = (emotion_sequences[0].get('anger', 0) + 
                      emotion_sequences[0].get('fear', 0) + 
                      emotion_sequences[0].get('sadness', 0))
        final_neg = (emotion_sequences[-1].get('anger', 0) + 
                    emotion_sequences[-1].get('fear', 0) + 
                    emotion_sequences[-1].get('sadness', 0))
        
        features['emo_final_vs_initial'] = float(
            (final_joy - initial_joy) - (final_neg - initial_neg)
        )
        
        # 4. Peak emotion position (where is the emotional climax)
        max_intensities = []
        for emo_dict in emotion_sequences:
            # Get maximum emotion value for each sentence
            max_intensity = max(emo_dict.values())
            max_intensities.append(max_intensity)
        
        peak_idx = np.argmax(max_intensities)
        features['emo_peak_position'] = float(peak_idx / max(1, n - 1))
        
        # 5. Dominant emotion strength
        all_emotions = {'anger': [], 'fear': [], 'joy': [], 'sadness': []}
        for e_dict in emotion_sequences:
            for emo in all_emotions.keys():
                all_emotions[emo].append(e_dict.get(emo, 0))
        
        avg_per_emotion = {e: np.mean(v) for e, v in all_emotions.items()}
        features['emo_dominant_emotion'] = float(max(avg_per_emotion.values()))
        
        return features


def add_emotional_trajectory_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """Add emotional trajectory features to dataframe."""
    df = df.copy()
    extractor = EmotionalTrajectoryExtractor()
    
    # Process all texts
    all_features = []
    for text in tqdm(df[text_col].tolist(), desc="Emotional Trajectories"):
        features = extractor.compute_trajectory_features(text)
        all_features.append(features)
    
    # Add to dataframe
    features_df = pd.DataFrame(all_features)
    for col in features_df.columns:
        df[col] = features_df[col].values
    
    return df
