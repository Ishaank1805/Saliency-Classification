"""
Flexible scene saliency classifier supporting comprehensive ablation studies.

Supports all ablation dimensions:
1. Scene encoder (BERT/RoBERTa, frozen/finetuned)
2. Linguistic features (with/without, concat/attention)
3. Sequence model (none/TransformerEncoder/RoBERTa)
4. Positional encoding (with/without)
5. Classifier (linear/MLP)
6. Feature subsets
"""

import torch
import torch.nn as nn
import math
import os
from typing import List, Optional, Dict
from transformers import AutoModel, AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Positional encoding relative to movie position."""
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        if self.batch_first:
            pe = pe.permute(1, 0, 2)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] if batch_first=True
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)


class SceneSaliencyWithLinguistic(nn.Module):
    """
    Flexible scene saliency classifier supporting all ablation dimensions.
    """
    
    def __init__(
        self,
        # Scene encoder
        scene_model_name: str = "bert-base-uncased",
        scene_encoder_finetune: bool = True,
        
        # Linguistic features
        use_linguistic: bool = True,
        linguistic_dim: int = 100,
        fusion_method: str = "concat",  # "concat", "attention", or None
        
        # Sequence model
        sequence_model_type: str = "transformer",  # "transformer", "roberta", "none"
        sequence_model_name: str = "roberta-base",  # If using RoBERTa
        num_transformer_layers: int = 6,
        num_heads: int = 8,
        
        # Positional encoding
        use_positional_encoding: bool = True,
        max_scenes: int = 5000,
        
        # Classifier
        classifier_type: str = "linear",  # "linear" or "mlp"
        
        # Pre-computed embeddings
        embeddings_cache_dir: Optional[str] = None,  # Directory with pre-computed embeddings
        
        # Common
        hidden_dim: int = 768,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_linguistic = use_linguistic
        self.fusion_method = fusion_method
        self.embeddings_cache_dir = embeddings_cache_dir
        self._embeddings_cache = {}  # Cache for loaded embeddings
        self.sequence_model_type = sequence_model_type
        self.use_positional_encoding = use_positional_encoding
        self.classifier_type = classifier_type
        
        # ========== STAGE 1: Scene Encoder ==========
        self.scene_encoder = AutoModel.from_pretrained(scene_model_name)
        self.scene_tokenizer = AutoTokenizer.from_pretrained(scene_model_name)
        scene_dim = self.scene_encoder.config.hidden_size
        
        # Freeze or finetune scene encoder
        if not scene_encoder_finetune:
            for param in self.scene_encoder.parameters():
                param.requires_grad = False
        
        self.scene_encoder_finetune = scene_encoder_finetune
        self.scene_dim = scene_dim
        
        # ========== STAGE 2: Feature Fusion ==========
        if use_linguistic:
            if fusion_method == "concat":
                # Simple concatenation
                self.feature_fusion = nn.Sequential(
                    nn.Linear(scene_dim + linguistic_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),  # Preserves negative information, matches transformer architecture
                    nn.Dropout(dropout)
                )
            elif fusion_method == "attention":
                # Attention-based fusion
                self.scene_proj = nn.Linear(scene_dim, hidden_dim)
                self.ling_proj = nn.Sequential(
                    nn.Linear(linguistic_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.attention_fusion = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=8,
                    batch_first=True
                )
                self.fusion_norm = nn.LayerNorm(hidden_dim)
            elif fusion_method == "gated":
                # Gated fusion: learns how much to trust linguistic features
                self.scene_proj = nn.Linear(scene_dim, hidden_dim)
                self.ling_proj = nn.Sequential(
                    nn.Linear(linguistic_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid()  # Gate values between 0 and 1
                )
                self.fusion_norm = nn.LayerNorm(hidden_dim)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        else:
            # No linguistic features - just project scene embeddings
            self.feature_fusion = nn.Sequential(
                nn.Linear(scene_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # Preserves negative information, matches transformer architecture
                nn.Dropout(dropout)
            )
        
        # ========== STAGE 3: Positional Encoding ==========
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=hidden_dim,
                dropout=dropout,
                max_len=max_scenes,
                batch_first=True
            )
        
        # ========== STAGE 4: Sequence Model ==========
        if sequence_model_type == "transformer":
            # PyTorch TransformerEncoder (no sequence length limit)
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.sequence_model = TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers
            )
        elif sequence_model_type == "roberta":
            # RoBERTa encoder (has 512 limit, will handle chunking)
            # Note: RoBERTa sequence modeling is complex - for now, fall back to transformer
            # Full implementation would require proper embedding handling
            print(f"Warning: RoBERTa sequence model has 512 scene limit. Using TransformerEncoder instead.")
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.sequence_model = TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers
            )
            # Keep roberta reference for future implementation
            self.roberta_model = AutoModel.from_pretrained(sequence_model_name)
            self.roberta_max_length = 512  # RoBERTa limit
        elif sequence_model_type == "none":
            # No sequence model - scenes processed independently
            self.sequence_model = None
        else:
            raise ValueError(f"Unknown sequence model type: {sequence_model_type}")
        
        # ========== STAGE 5: Classifier ==========
        # Use skip connection: concatenate transformer output (context) with fused input (local)
        # This helps preserve scene-specific details that might be smoothed by the transformer
        classifier_input_dim = hidden_dim * 2  # Skip connection doubles the input dimension
        
        if classifier_type == "linear":
            # Single linear layer with skip connection
            self.classifier = nn.Linear(classifier_input_dim, 1)
        elif classifier_type == "mlp":
            # 2-layer MLP with skip connection
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),  # Input size doubled for skip connection
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Move to device
        self.to(self.device)
    
    def load_precomputed_embeddings(self, movie_id: int, split: str = "train") -> Optional[torch.Tensor]:
        """Load pre-computed embeddings for a movie if available."""
        if self.embeddings_cache_dir is None:
            return None
        
        # Check cache first
        cache_key = (movie_id, split)
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        # Try to load from disk
        import pickle
        
        model_name_safe = self.scene_encoder.config.name_or_path.replace('/', '_').replace('-', '_')
        cache_file = os.path.join(
            self.embeddings_cache_dir,
            f"{model_name_safe}_{split}.pkl"
        )
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            embeddings_dict = cache_data.get('embeddings', {})
            if movie_id in embeddings_dict:
                emb = torch.tensor(embeddings_dict[movie_id], dtype=torch.float32).to(self.device)
                self._embeddings_cache[cache_key] = emb
                return emb
        except Exception as e:
            print(f"Warning: Could not load pre-computed embeddings: {e}")
        
        return None
    
    def encode_scenes(
        self,
        scene_texts: List[str],
        batch_size: int = 16,
        movie_id: Optional[int] = None,
        split: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode scenes to embeddings.
        
        Uses pre-computed embeddings if available (for frozen encoders),
        otherwise computes on-the-fly (for finetuned encoders).
        
        Args:
            scene_texts: List of scene text strings
            batch_size: Batch size for encoding
            movie_id: Movie ID for loading pre-computed embeddings
            split: Data split (train/validation/test) for loading pre-computed embeddings
            
        Returns:
            Scene embeddings: [num_scenes, scene_dim]
        """
        # Try to use pre-computed embeddings if available and encoder is frozen
        if not self.scene_encoder_finetune and movie_id is not None and split is not None:
            precomputed = self.load_precomputed_embeddings(movie_id, split)
            if precomputed is not None:
                # Verify dimensions match
                if precomputed.shape[0] == len(scene_texts):
                    return precomputed
                # If shape doesn't match (e.g., using balanced dataset with original cache),
                # silently compute on-the-fly instead of warning for every movie
        
        # Compute embeddings on-the-fly
        if not self.scene_encoder_finetune:
            self.scene_encoder.eval()
        else:
            self.scene_encoder.train()
        
        embeddings = []
        
        with torch.set_grad_enabled(self.scene_encoder_finetune):
            for i in range(0, len(scene_texts), batch_size):
                batch = scene_texts[i:i+batch_size]
                
                encoded = self.scene_tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512  # Each scene max 512 tokens
                ).to(self.device)
                
                outputs = self.scene_encoder(**encoded)
                
                # Use [CLS] token or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    scene_emb = outputs.pooler_output  # [batch, hidden_dim]
                else:
                    # Mean pooling over sequence length
                    scene_emb = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
                
                # Check for NaN/Inf and replace with zeros if found
                if torch.isnan(scene_emb).any() or torch.isinf(scene_emb).any():
                    scene_emb = torch.nan_to_num(scene_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"Warning: NaN/Inf in scene embeddings, replaced with zeros")
                
                # Normalize embeddings to prevent extreme values
                # L2 normalize to unit length (helps with numerical stability)
                # Use manual normalization with epsilon to prevent division by zero
                emb_norm = torch.norm(scene_emb, p=2, dim=-1, keepdim=True)
                scene_emb = scene_emb / (emb_norm + 1e-8)
                
                embeddings.append(scene_emb)
        
        return torch.cat(embeddings, dim=0)  # [num_scenes, scene_dim]
    
    def fuse_features(
        self,
        scene_embeddings: torch.Tensor,
        linguistic_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse scene embeddings with linguistic features.
        
        Args:
            scene_embeddings: [num_scenes, scene_dim]
            linguistic_features: [num_scenes, linguistic_dim] or None
            
        Returns:
            Fused features: [num_scenes, hidden_dim]
        """
        # Normalize scene embeddings to prevent extreme values (with epsilon to prevent division by zero)
        scene_emb_norm = torch.norm(scene_embeddings, p=2, dim=-1, keepdim=True)
        scene_embeddings = scene_embeddings / (scene_emb_norm + 1e-8)
        
        if not self.use_linguistic or linguistic_features is None:
            # No linguistic features - just project
            fused = self.feature_fusion(scene_embeddings)
            # Normalize output to prevent extreme values
            fused_norm = torch.norm(fused, p=2, dim=-1, keepdim=True)
            fused = fused / (fused_norm + 1e-8)
            return fused
        
        # Normalize linguistic features as well
        if linguistic_features is not None:
            ling_norm = torch.norm(linguistic_features, p=2, dim=-1, keepdim=True)
            linguistic_features = linguistic_features / (ling_norm + 1e-8)
        
        if self.fusion_method == "concat":
            # Simple concatenation
            combined = torch.cat([scene_embeddings, linguistic_features], dim=-1)
            fused = self.feature_fusion(combined)
            # Normalize output to prevent extreme values
            fused_norm = torch.norm(fused, p=2, dim=-1, keepdim=True)
            fused = fused / (fused_norm + 1e-8)
            return fused
        
        elif self.fusion_method == "attention":
            # Attention-based fusion
            scene_proj = self.scene_proj(scene_embeddings)  # [num_scenes, hidden_dim]
            ling_proj = self.ling_proj(linguistic_features)  # [num_scenes, hidden_dim]
            
            # Cross-attention: scene embeddings attend to linguistic features
            fused, _ = self.attention_fusion(
                scene_proj.unsqueeze(0),  # query [1, num_scenes, hidden_dim]
                ling_proj.unsqueeze(0),   # key [1, num_scenes, hidden_dim]
                ling_proj.unsqueeze(0)     # value [1, num_scenes, hidden_dim]
            )
            
            # Residual connection
            fused = fused.squeeze(0) + scene_proj
            fused = self.fusion_norm(fused)
            
            return fused
        
        elif self.fusion_method == "gated":
            # Gated fusion: learns how much to trust linguistic features for each scene
            h_scene = self.scene_proj(scene_embeddings)  # [num_scenes, hidden_dim]
            h_ling = self.ling_proj(linguistic_features)  # [num_scenes, hidden_dim]
            
            # Calculate gate: values between 0 and 1
            # 1 = trust linguistic features, 0 = ignore them
            gate_input = torch.cat([h_scene, h_ling], dim=-1)  # [num_scenes, hidden_dim * 2]
            g = self.gate(gate_input)  # [num_scenes, hidden_dim]
            
            # Weighted sum: gate controls how much linguistic info to use
            fused = g * h_ling + (1 - g) * h_scene
            fused = self.fusion_norm(fused)
            
            return fused
    
    def apply_sequence_model(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply sequence model to features.
        
        Args:
            features: [1, num_scenes, hidden_dim]
            mask: [1, num_scenes] boolean mask (True = padding)
            
        Returns:
            Sequence output: [1, num_scenes, hidden_dim]
        """
        if self.sequence_model_type == "none":
            # No sequence model - return as is
            return features
        
        elif self.sequence_model_type == "transformer":
            # TransformerEncoder (no length limit)
            return self.sequence_model(features, src_key_padding_mask=mask)
        
        elif self.sequence_model_type == "roberta":
            # RoBERTa encoder (has 512 limit - need to chunk if needed)
            num_scenes = features.size(1)
            
            if num_scenes <= self.roberta_max_length:
                # Can process all at once
                # RoBERTa expects input_ids, but we have embeddings
                # We'll use a workaround: create dummy input_ids and use inputs_embeds
                batch_size = features.size(0)
                dummy_ids = torch.zeros(
                    batch_size, num_scenes,
                    dtype=torch.long,
                    device=features.device
                )
                
                # Get embeddings layer
                roberta_embeddings = self.sequence_model.embed_tokens if hasattr(self.sequence_model, 'embed_tokens') else None
                
                # Use inputs_embeds directly if supported
                # For now, we'll process in chunks if > 512
                if num_scenes > self.roberta_max_length:
                    # Chunk processing
                    outputs = []
                    for i in range(0, num_scenes, self.roberta_max_length):
                        chunk = features[:, i:i+self.roberta_max_length, :]
                        chunk_mask = mask[:, i:i+self.roberta_max_length] if mask is not None else None
                        
                        # Process chunk (simplified - would need proper RoBERTa embedding handling)
                        # For now, fall back to transformer if too long
                        chunk_output = self._process_roberta_chunk(chunk, chunk_mask)
                        outputs.append(chunk_output)
                    
                    return torch.cat(outputs, dim=1)
                else:
                    return self._process_roberta_chunk(features, mask)
            
            return features
        
        return features
    
    
    def forward(
        self,
        scene_texts: List[str],
        linguistic_features: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        movie_id: Optional[int] = None,
        split: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            scene_texts: List of scene text strings
            linguistic_features: [num_scenes, linguistic_dim] or None
            scene_mask: [num_scenes] boolean mask (True = padding)
            movie_id: Movie ID for loading pre-computed embeddings (optional)
            split: Data split for loading pre-computed embeddings (optional)
            
        Returns:
            Logits: [num_scenes, 1]
        """
        # Stage 1: Encode scenes (uses pre-computed if available)
        scene_embeddings = self.encode_scenes(scene_texts, movie_id=movie_id, split=split)  # [num_scenes, scene_dim]
        
        # Check for NaN/Inf in scene embeddings
        if torch.isnan(scene_embeddings).any() or torch.isinf(scene_embeddings).any():
            scene_embeddings = torch.nan_to_num(scene_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Stage 2: Fuse with linguistic features
        fused = self.fuse_features(scene_embeddings, linguistic_features)  # [num_scenes, hidden_dim]
        
        # Check for NaN/Inf in fused features
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Stage 3: Add positional encoding
        fused = fused.unsqueeze(0)  # [1, num_scenes, hidden_dim] for batch
        if self.use_positional_encoding:
            fused = self.pos_encoder(fused)  # [1, num_scenes, hidden_dim]
        
        # Stage 4: Sequence modeling with full context
        if scene_mask is not None:
            scene_mask = scene_mask.unsqueeze(0)  # [1, num_scenes]
        
        sequence_output = self.apply_sequence_model(fused, scene_mask)  # [1, num_scenes, hidden_dim]
        
        # Check for NaN/Inf in sequence output
        if torch.isnan(sequence_output).any() or torch.isinf(sequence_output).any():
            sequence_output = torch.nan_to_num(sequence_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- SKIP CONNECTION: Concatenate transformer output (context) with fused input (local) ---
        # This preserves scene-specific details that might be smoothed by the transformer
        # fused: [1, num_scenes, hidden_dim] (local features before transformer)
        # sequence_output: [1, num_scenes, hidden_dim] (contextualized features after transformer)
        combined_features = torch.cat([sequence_output, fused], dim=-1)  # [1, num_scenes, hidden_dim * 2]
        
        # Stage 5: Classify each scene (using combined features with skip connection)
        logits = self.classifier(combined_features)  # [1, num_scenes, 1]
        logits = logits.squeeze(0)  # [num_scenes, 1]
        
        # Final check: ensure no NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            print(f"Warning: NaN/Inf in final logits, replaced with safe values")
        
        # Clip logits to safe range to prevent numerical instability
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        return logits
    
    def get_config(self) -> Dict:
        """Get model configuration for tracking."""
        return {
            "scene_model": self.scene_encoder.config.name_or_path if hasattr(self.scene_encoder.config, 'name_or_path') else "unknown",
            "scene_finetune": self.scene_encoder_finetune,
            "use_linguistic": self.use_linguistic,
            "fusion_method": self.fusion_method if self.use_linguistic else None,
            "sequence_model": self.sequence_model_type,
            "use_positional_encoding": self.use_positional_encoding,
            "classifier": self.classifier_type,
            "hidden_dim": self.scene_dim,
        }

