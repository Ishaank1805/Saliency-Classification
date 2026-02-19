"""
Comprehensive experiment configurations for exhaustive ablation studies.

All 6 dimensions of ablation:
1. Scene encoder (BERT-base/large, RoBERTa-base/large, frozen/finetuned)
2. Linguistic features (with/without, concat/attention)
3. Sequence model (none/TransformerEncoder/RoBERTa)
4. Positional encoding (with/without)
5. Classifier (linear/MLP)
6. Feature subsets
"""

EXPERIMENT_CONFIGS = {
    # ========== BASELINE EXPERIMENTS (No Linguistic Features) ==========
    
    "baseline_saxena": {
        "description": "Saxena & Keller baseline (reproduction) - RoBERTa-large frozen",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "sequence_model_name": "roberta-base",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "baseline_bert_base_frozen": {
        "description": "BERT-base frozen, no linguistic, TransformerEncoder",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "baseline_bert_base_finetuned": {
        "description": "BERT-base finetuned, no linguistic, TransformerEncoder",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "baseline_bert_large_frozen": {
        "description": "BERT-large frozen, no linguistic, TransformerEncoder",
        "scene_model_name": "bert-large-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "baseline_roberta_base_frozen": {
        "description": "RoBERTa-base frozen, no linguistic, TransformerEncoder",
        "scene_model_name": "roberta-base",
        "scene_encoder_finetune": False,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "baseline_roberta_large_frozen": {
        "description": "RoBERTa-large frozen, no linguistic, TransformerEncoder",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "baseline_bert_independent": {
        "description": "BERT-base independent scenes (no sequence model)",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": False,
        "fusion_method": "concat",
        "sequence_model_type": "none",
        "use_positional_encoding": False,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    # ========== MAIN CONTRIBUTIONS: Linguistic Features ==========
    # BERT-base variants
    
    "bert_base_frozen_linguistic_concat": {
        "description": "BERT-base frozen + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "bert_base_finetuned_linguistic_concat": {
        "description": "BERT-base finetuned + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "bert_base_frozen_linguistic_attention": {
        "description": "BERT-base frozen + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "bert_base_finetuned_linguistic_attention": {
        "description": "BERT-base finetuned + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    # BERT-large variants
    
    "bert_large_frozen_linguistic_concat": {
        "description": "BERT-large frozen + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "bert-large-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 2,  # Increased from 6 to 10 for better performance
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "bert_large_frozen_linguistic_concat_mlp": {
        "description": "BERT-large frozen + Linguistic (concat) + TransformerEncoder + MLP classifier",
        "scene_model_name": "bert-large-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,  # Match linear version to control for capacity
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "mlp",  # MLP instead of linear
        "hidden_dim": 1024,
        "dropout": 0.05,  # Lower dropout in classifier to reduce interference
    },
    
    "roberta_large_frozen_linguistic_concat_enhanced": {
        "description": "RoBERTa-large frozen + Linguistic (concat) + Enhanced TransformerEncoder",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 12,  # More layers
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "mlp",  # MLP classifier
        "hidden_dim": 1024,
        "dropout": 0.15,
    },
    
    "bert_large_finetuned_linguistic_concat": {
        "description": "BERT-large finetuned + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "bert-large-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "bert_large_frozen_linguistic_attention": {
        "description": "BERT-large frozen + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "bert-large-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "bert_large_frozen_linguistic_gated": {
        "description": "BERT-large frozen + Linguistic (gated) + TransformerEncoder",
        "scene_model_name": "bert-large-uncased",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "gated",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    # RoBERTa-base variants
    
    "roberta_base_frozen_linguistic_concat": {
        "description": "RoBERTa-base frozen + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "roberta-base",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "roberta_base_finetuned_linguistic_concat": {
        "description": "RoBERTa-base finetuned + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "roberta-base",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "roberta_base_frozen_linguistic_attention": {
        "description": "RoBERTa-base frozen + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "roberta-base",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "roberta_base_finetuned_linguistic_attention": {
        "description": "RoBERTa-base finetuned + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "roberta-base",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    # RoBERTa-large variants (KEY FOR FAIR COMPARISON)
    
    "roberta_large_frozen_linguistic_concat": {
        "description": "RoBERTa-large frozen + Linguistic (concat) + TransformerEncoder - FAIR COMPARISON",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "roberta_large_finetuned_linguistic_concat": {
        "description": "RoBERTa-large finetuned + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "roberta_large_frozen_linguistic_attention": {
        "description": "RoBERTa-large frozen + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    # DeBERTa-v3-large variants
    "deberta_v3_large_frozen_linguistic_concat": {
        "description": "DeBERTa-v3-large frozen + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "microsoft/deberta-v3-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 2,  # Reduced from 10 to 2 to prevent overfitting (824 movies)
        "num_heads": 8,  # Reduced from 16 to 8
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,  # DeBERTa-v3-large has hidden_size=1024
        "dropout": 0.2,  # Increased from 0.1 to 0.2 for better regularization
    },
    
    "deberta_v3_large_frozen_linguistic_attention": {
        "description": "DeBERTa-v3-large frozen + Linguistic (attention) + TransformerEncoder",
        "scene_model_name": "microsoft/deberta-v3-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "attention",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "deberta_v3_large_finetuned_linguistic_concat": {
        "description": "DeBERTa-v3-large finetuned + Linguistic (concat) + TransformerEncoder",
        "scene_model_name": "microsoft/deberta-v3-large",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 2,  # Reduced from 6 to 2 to prevent overfitting (824 movies)
        "num_heads": 8,  # Reduced from 16 to 8
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.2,  # Increased from 0.1 to 0.2 for better regularization
    },
    
    # ========== SEQUENCE MODEL COMPARISON ==========
    
    "no_sequence_bert_base": {
        "description": "BERT-base, no sequence model (independent scenes)",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "none",
        "use_positional_encoding": False,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "transformer_sequence_bert_base": {
        "description": "BERT-base + TransformerEncoder sequence",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    # ========== POSITIONAL ENCODING ABLATION ==========
    
    "no_pos_encoding_roberta_large": {
        "description": "RoBERTa-large + Linguistic, no positional encoding",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": False,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "with_pos_encoding_roberta_large": {
        "description": "RoBERTa-large + Linguistic, with positional encoding",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    # ========== CLASSIFIER COMPARISON ==========
    
    "linear_classifier_roberta_large": {
        "description": "RoBERTa-large + Linguistic, linear classifier",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "mlp_classifier_roberta_large": {
        "description": "RoBERTa-large + Linguistic, MLP classifier",
        "scene_model_name": "roberta-large",
        "scene_encoder_finetune": False,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "mlp",
        "hidden_dim": 1024,
        "dropout": 0.1,
    },
    
    "linear_classifier_bert_base": {
        "description": "BERT-base + Linguistic, linear classifier",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "mlp_classifier_bert_base": {
        "description": "BERT-base + Linguistic, MLP classifier",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "mlp",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    # ========== TRANSFORMER LAYERS ABLATION ==========
    
    "transformer_3_layers": {
        "description": "BERT-base + Linguistic, 3 transformer layers",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 3,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "transformer_6_layers": {
        "description": "BERT-base + Linguistic, 6 transformer layers",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "transformer_10_layers": {
        "description": "BERT-base + Linguistic, 10 transformer layers",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 10,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    # ========== ATTENTION HEADS ABLATION ==========
    
    "transformer_4_heads": {
        "description": "BERT-base + Linguistic, 4 attention heads",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 4,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "transformer_8_heads": {
        "description": "BERT-base + Linguistic, 8 attention heads",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 8,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
    
    "transformer_16_heads": {
        "description": "BERT-base + Linguistic, 16 attention heads",
        "scene_model_name": "bert-base-uncased",
        "scene_encoder_finetune": True,
        "use_linguistic": True,
        "fusion_method": "concat",
        "sequence_model_type": "transformer",
        "num_transformer_layers": 6,
        "num_heads": 16,
        "use_positional_encoding": True,
        "classifier_type": "linear",
        "hidden_dim": 768,
        "dropout": 0.1,
    },
}


# Experiment groups for organized running
EXPERIMENT_GROUPS = {
    # Core experiments for fair comparison
    "core_fair_comparison": [
        "baseline_saxena",  # RoBERTa-large, no linguistic
        "roberta_large_frozen_linguistic_concat",  # RoBERTa-large, with linguistic
        "roberta_large_finetuned_linguistic_concat",  # RoBERTa-large finetuned, with linguistic
    ],
    
    # Model size comparison (with linguistic features)
    "model_size_comparison": [
        "bert_base_finetuned_linguistic_concat",
        "bert_large_finetuned_linguistic_concat",
        "roberta_base_finetuned_linguistic_concat",
        "roberta_large_frozen_linguistic_concat",
    ],
    
    # Finetuning impact
    "finetuning_impact": [
        "bert_base_frozen_linguistic_concat",
        "bert_base_finetuned_linguistic_concat",
        "roberta_base_frozen_linguistic_concat",
        "roberta_base_finetuned_linguistic_concat",
        "roberta_large_frozen_linguistic_concat",
        "roberta_large_finetuned_linguistic_concat",
    ],
    
    # Fusion method comparison
    "fusion_methods": [
        "bert_base_finetuned_linguistic_concat",
        "bert_base_finetuned_linguistic_attention",
        "roberta_base_finetuned_linguistic_concat",
        "roberta_base_finetuned_linguistic_attention",
        "roberta_large_frozen_linguistic_concat",
        "roberta_large_frozen_linguistic_attention",
    ],
    
    # Sequence model impact
    "sequence_model": [
        "no_sequence_bert_base",
        "transformer_sequence_bert_base",
        "baseline_bert_independent",
    ],
    
    # Positional encoding
    "positional_encoding": [
        "no_pos_encoding_roberta_large",
        "with_pos_encoding_roberta_large",
    ],
    
    # Classifier comparison
    "classifier": [
        "linear_classifier_roberta_large",
        "mlp_classifier_roberta_large",
        "linear_classifier_bert_base",
        "mlp_classifier_bert_base",
    ],
    
    # Transformer architecture
    "transformer_layers": [
        "transformer_3_layers",
        "transformer_6_layers",
        "transformer_10_layers",
    ],
    
    "attention_heads": [
        "transformer_4_heads",
        "transformer_8_heads",
        "transformer_16_heads",
    ],
    
    # All baselines (no linguistic)
    "baselines": [
        "baseline_saxena",
        "baseline_bert_base_frozen",
        "baseline_bert_base_finetuned",
        "baseline_bert_large_frozen",
        "baseline_roberta_base_frozen",
        "baseline_roberta_large_frozen",
        "baseline_bert_independent",
    ],
    
    # All BERT-base experiments
    "bert_base_all": [
        "baseline_bert_base_frozen",
        "baseline_bert_base_finetuned",
        "bert_base_frozen_linguistic_concat",
        "bert_base_finetuned_linguistic_concat",
        "bert_base_frozen_linguistic_attention",
        "bert_base_finetuned_linguistic_attention",
    ],
    
    # All RoBERTa-large experiments (most important for fair comparison)
    "roberta_large_all": [
        "baseline_saxena",
        "baseline_roberta_large_frozen",
        "roberta_large_frozen_linguistic_concat",
        "roberta_large_finetuned_linguistic_concat",
        "roberta_large_frozen_linguistic_attention",
        "no_pos_encoding_roberta_large",
        "with_pos_encoding_roberta_large",
        "linear_classifier_roberta_large",
        "mlp_classifier_roberta_large",
    ],
    
    # All experiments
    "all": list(EXPERIMENT_CONFIGS.keys()),
}


def get_experiment_config(experiment_name: str) -> dict:
    """Get configuration for a specific experiment."""
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    return EXPERIMENT_CONFIGS[experiment_name].copy()


def get_experiment_group(group_name: str) -> list:
    """Get list of experiments in a group."""
    if group_name not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown group: {group_name}")
    return EXPERIMENT_GROUPS[group_name]


def list_all_experiments() -> list:
    """List all available experiments."""
    return list(EXPERIMENT_CONFIGS.keys())


def list_all_groups() -> list:
    """List all available experiment groups."""
    return list(EXPERIMENT_GROUPS.keys())
