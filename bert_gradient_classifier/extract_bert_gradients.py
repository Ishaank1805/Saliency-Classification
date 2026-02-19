"""
Extract BERT gradients and activations for binary classification.

This module:
1. Trains a binary classifier on top of BERT
2. Extracts gradients from last layer and multiple layers
3. Extracts BERT activations (hidden states)
4. Returns combined feature representations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd


class BERTGradientExtractor:
    """Extract gradients and activations from BERT for binary classification."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        device: Optional[str] = None,
        extract_layers: List[int] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
            extract_layers: Which layers to extract gradients from (None = all layers)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extract_layers = extract_layers or [-1]  # Default: last layer only
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.to(self.device)
        self.base_model.eval()
        
        # Classification head (will be trained)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 1)
        self.classifier.to(self.device)
        
    def train_classifier(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict[str, float]:
        """
        Train binary classifier on top of BERT.
        
        Returns:
            Training metrics dictionary
        """
        # Prepare data
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            labels_tensor
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            list(self.base_model.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.base_model.train()
        self.classifier.train()
        
        metrics = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output  # [CLS] token representation
                logits = self.classifier(pooled_output)
                
                # Loss
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(accuracy)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        self.base_model.eval()
        self.classifier.eval()
        
        return {
            "final_loss": metrics["loss"][-1],
            "final_accuracy": metrics["accuracy"][-1],
        }
    
    def extract_gradients(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        batch_size: int = 16,
        compute_both_labels: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract gradients from BERT for binary saliency classification.
        
        Computes gradients for BOTH labels (1=salient, 0=non-salient) separately,
        and also computes the difference as a discriminative feature.
        
        Extracts from:
        1. Input embeddings layer (first layer)
        2. Last hidden layer (final transformer layer)
        
        Args:
            texts: List of input texts
            labels: Binary labels (1=salient, 0=non-salient). Not used if compute_both_labels=True.
            batch_size: Batch size for processing
            compute_both_labels: If True, compute gradients for both label=1 and label=0 separately
            
        Returns:
            Dictionary with gradient features:
            - 'emb_grad_label1': Gradients w.r.t. input embeddings for label=1 [batch, 768]
            - 'emb_grad_label0': Gradients w.r.t. input embeddings for label=0 [batch, 768]
            - 'emb_grad_diff': Difference (label1 - label0) for embeddings [batch, 768]
            - 'last_grad_label1': Gradients w.r.t. last layer for label=1 [batch, 768]
            - 'last_grad_label0': Gradients w.r.t. last layer for label=0 [batch, 768]
            - 'last_grad_diff': Difference (label1 - label0) for last layer [batch, 768]
        """
        self.base_model.train()  # Enable gradient computation
        self.classifier.train()
        
        # Storage for both labels and difference
        all_emb_grad_label1 = []
        all_emb_grad_label0 = []
        all_emb_grad_diff = []
        all_last_grad_label1 = []
        all_last_grad_label0 = []
        all_last_grad_diff = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting gradients (both labels)"):
            batch_texts = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]  # Keep as long for model forward
            
            # ===== EXTRACT FROM INPUT EMBEDDINGS LAYER =====
            # Get embeddings and enable gradients
            # Make embeddings a leaf so .grad is reliably populated
            embeddings = self.base_model.embeddings(input_ids).detach()
            embeddings.requires_grad_(True)
            
            # Forward pass through full model with inputs_embeds (handles attention mask correctly)
            encoder_outputs = self.base_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get pooled output ([CLS] token)
            pooled = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            logits = self.classifier(pooled)
            
            criterion = nn.BCEWithLogitsLoss()
            
            if compute_both_labels:
                # Compute gradients for label=1 (salient)
                labels_1 = torch.ones_like(logits).to(self.device)
                loss_1 = criterion(logits, labels_1)
                
                self.base_model.zero_grad()
                self.classifier.zero_grad()
                # Zero embeddings gradient if it exists
                if embeddings.grad is not None:
                    embeddings.grad.zero_()
                # Need retain_graph since we do a second backward on the same forward pass
                loss_1.backward(retain_graph=True)
                emb_grad_label1 = embeddings.grad.clone() if embeddings.grad is not None else torch.zeros_like(embeddings)  # [batch, seq_len, hidden_size]
                
                # Compute gradients for label=0 (non-salient)
                labels_0 = torch.zeros_like(logits).to(self.device)
                loss_0 = criterion(logits, labels_0)
                
                self.base_model.zero_grad()
                self.classifier.zero_grad()
                # Zero embeddings gradient before second backward pass
                if embeddings.grad is not None:
                    embeddings.grad.zero_()
                loss_0.backward()
                emb_grad_label0 = embeddings.grad.clone() if embeddings.grad is not None else torch.zeros_like(embeddings)  # [batch, seq_len, hidden_size]
                
                # Difference (discriminative signal)
                emb_grad_diff = emb_grad_label1 - emb_grad_label0
                
                # Aggregate: mean pool over sequence length
                emb_grad_label1_mean = emb_grad_label1.mean(dim=1).cpu().numpy()  # [batch, 768]
                emb_grad_label0_mean = emb_grad_label0.mean(dim=1).cpu().numpy()  # [batch, 768]
                emb_grad_diff_mean = emb_grad_diff.mean(dim=1).cpu().numpy()      # [batch, 768]
                
                all_emb_grad_label1.append(emb_grad_label1_mean)
                all_emb_grad_label0.append(emb_grad_label0_mean)
                all_emb_grad_diff.append(emb_grad_diff_mean)
                
                # Clear GPU memory periodically
                del emb_grad_label1, emb_grad_label0, emb_grad_diff
            else:
                # Original approach: use actual labels
                batch_labels = labels[i:i+batch_size] if labels else None
                if batch_labels is None:
                    with torch.no_grad():
                        batch_labels = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                else:
                    batch_labels = np.array(batch_labels)
                
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(self.device)
                loss = criterion(logits, batch_labels_tensor)
                
                self.base_model.zero_grad()
                self.classifier.zero_grad()
                loss.backward()
                
                emb_grad = embeddings.grad
                emb_grad_mean = emb_grad.mean(dim=1).cpu().numpy()
                all_emb_grad_label1.append(emb_grad_mean)  # Store as label1 for compatibility
                del emb_grad
            
            # ===== EXTRACT FROM LAST HIDDEN LAYER =====
            # Get last layer hidden states BEFORE deleting encoder_outputs
            # Detach to avoid reusing the full graph from embeddings; make leaf so .grad works
            last_hidden = encoder_outputs.hidden_states[-1].detach()  # [batch, seq_len, 768]
            last_hidden.requires_grad_(True)
            
            # Now we can clear intermediate tensors (but keep last_hidden)
            del embeddings, encoder_outputs, pooled, logits, encodings, input_ids, attention_mask
            
            # Clear GPU cache periodically (every 50 batches)
            if (i // batch_size) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Forward pass from last layer
            pooled_last = last_hidden[:, 0, :]  # [CLS] token from last layer
            logits_last = self.classifier(pooled_last)
            
            if compute_both_labels:
                # Compute gradients for label=1 (salient) from last layer
                loss_1_last = criterion(logits_last, labels_1)
                
                self.classifier.zero_grad()
                if last_hidden.grad is not None:
                    last_hidden.grad.zero_()
                # Need retain_graph since we do a second backward on the same logits_last graph
                loss_1_last.backward(retain_graph=True)
                last_grad_label1 = last_hidden.grad[:, 0, :].clone().cpu().numpy()  # [CLS] token [batch, 768]
                
                # Compute gradients for label=0 (non-salient) from last layer
                loss_0_last = criterion(logits_last, labels_0)
                
                self.classifier.zero_grad()
                if last_hidden.grad is not None:
                    last_hidden.grad.zero_()
                loss_0_last.backward()
                last_grad_label0 = last_hidden.grad[:, 0, :].clone().cpu().numpy()  # [CLS] token [batch, 768]
                
                # Difference (discriminative signal)
                last_grad_diff = last_grad_label1 - last_grad_label0
                
                all_last_grad_label1.append(last_grad_label1)
                all_last_grad_label0.append(last_grad_label0)
                all_last_grad_diff.append(last_grad_diff)
            else:
                # Original approach: use actual labels
                loss_last = criterion(logits_last, batch_labels_tensor)
                
                self.classifier.zero_grad()
                if last_hidden.grad is not None:
                    last_hidden.grad.zero_()
                loss_last.backward()
                
                last_grad = last_hidden.grad[:, 0, :].cpu().numpy()
                all_last_grad_label1.append(last_grad)  # Store as label1 for compatibility
        
        self.base_model.eval()
        self.classifier.eval()
        
        # Combine all batches
        result = {}
        if all_emb_grad_label1:
            result["emb_grad_label1"] = np.vstack(all_emb_grad_label1)  # [batch, 768]
        if all_emb_grad_label0:
            result["emb_grad_label0"] = np.vstack(all_emb_grad_label0)  # [batch, 768]
        if all_emb_grad_diff:
            result["emb_grad_diff"] = np.vstack(all_emb_grad_diff)      # [batch, 768]
        if all_last_grad_label1:
            result["last_grad_label1"] = np.vstack(all_last_grad_label1)  # [batch, 768]
        if all_last_grad_label0:
            result["last_grad_label0"] = np.vstack(all_last_grad_label0)  # [batch, 768]
        if all_last_grad_diff:
            result["last_grad_diff"] = np.vstack(all_last_grad_diff)      # [batch, 768]
        
        return result
    
    def extract_activations(
        self,
        texts: List[str],
        batch_size: int = 16,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract BERT activations (hidden states).
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            layers: Which layers to extract (None = all layers)
            
        Returns:
            Dictionary with activation features:
            - 'pooled_output': [CLS] token representation
            - 'last_hidden_state': Last layer hidden states (mean pooled)
            - 'all_layers': All layer representations (mean pooled)
        """
        self.base_model.eval()
        
        all_pooled = []
        all_last_hidden = []
        all_layers = []
        
        layers = layers or list(range(self.base_model.config.num_hidden_layers))
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
                batch_texts = texts[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.base_model(**encodings, output_hidden_states=True)
                
                # Pooled output ([CLS] token)
                pooled = outputs.pooler_output.cpu().numpy()
                all_pooled.append(pooled)
                
                # Last hidden state (mean pooled)
                last_hidden = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_last_hidden.append(last_hidden)
                
                # All layers (mean pooled)
                if outputs.hidden_states:
                    layer_reps = []
                    for layer_idx in layers:
                        if layer_idx < len(outputs.hidden_states):
                            layer_hidden = outputs.hidden_states[layer_idx].mean(dim=1).cpu().numpy()
                            layer_reps.append(layer_hidden)
                    if layer_reps:
                        all_layers.append(np.hstack(layer_reps))
                
                # Clear intermediate tensors
                del outputs, encodings
                
                # Clear GPU cache periodically (every 100 batches)
                if (i // batch_size) % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        result = {}
        if all_pooled:
            result["pooled_output"] = np.vstack(all_pooled)
        if all_last_hidden:
            result["last_hidden_state"] = np.vstack(all_last_hidden)
        if all_layers:
            result["all_layers"] = np.vstack(all_layers)
        
        return result
    
    def extract_all_features(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        extract_grads: bool = True,
        extract_acts: bool = True,
        compute_both_labels: bool = True,
        batch_size: int = 16,
    ) -> Dict[str, np.ndarray]:
        """
        Extract both gradients and activations.
        
        Args:
            texts: List of input texts
            labels: Optional labels (not used if compute_both_labels=True)
            extract_grads: Whether to extract gradients
            extract_acts: Whether to extract activations
            compute_both_labels: If True, compute gradients for both label=1 and label=0
            batch_size: Batch size for processing
        
        Returns:
            Combined dictionary of all features:
            - Gradient features: emb_grad_label1, emb_grad_label0, emb_grad_diff,
                                last_grad_label1, last_grad_label0, last_grad_diff
            - Activation features: pooled_output, last_hidden_state, all_layers
        """
        all_features = {}
        
        if extract_grads:
            grad_features = self.extract_gradients(
                texts, labels, 
                compute_both_labels=compute_both_labels,
                batch_size=batch_size
            )
            all_features.update(grad_features)
        
        if extract_acts:
            act_features = self.extract_activations(texts, batch_size=batch_size)
            all_features.update(act_features)
        
        return all_features


def extract_bert_features_for_dataframe(
    df: pd.DataFrame,
    text_col: str = "scene_text",
    label_col: str = "label",
    model_name: str = "bert-base-uncased",
    train_classifier: bool = True,
    extract_gradients: bool = True,
    extract_activations: bool = True,
    max_length: int = 256,
    batch_size: int = 16,
    chunk_size: int = 10000,  # Process in chunks to avoid memory issues
) -> pd.DataFrame:
    """
    Extract BERT gradients and activations for a DataFrame.
    Processes in chunks to avoid memory exhaustion.
    
    Args:
        df: DataFrame with texts and labels
        text_col: Column name for text
        label_col: Column name for labels
        model_name: BERT model name
        train_classifier: Whether to train classifier first
        extract_gradients: Whether to extract gradients
        extract_activations: Whether to extract activations
        max_length: Max sequence length
        batch_size: Batch size for feature extraction
        chunk_size: Number of samples to process at once (to avoid OOM)
        
    Returns:
        DataFrame with added BERT features
    """
    import gc
    import torch
    
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist() if label_col in df.columns else None
    total_samples = len(texts)
    
    extractor = BERTGradientExtractor(
        model_name=model_name,
        max_length=max_length,
    )
    
    # Train classifier if needed (use all data for training)
    if train_classifier and labels is not None:
        print("Training BERT classifier...")
        extractor.train_classifier(texts, labels)
        # Clear memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Process in chunks
    print(f"Extracting BERT features in chunks of {chunk_size} samples...")
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    all_feature_dicts = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx}-{end_idx})...")
        
        chunk_texts = texts[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx] if labels else None
        
        # Extract features for this chunk
        chunk_features = extractor.extract_all_features(
            chunk_texts,
            chunk_labels,
            extract_grads=extract_gradients,
            extract_acts=extract_activations,
            compute_both_labels=True,  # Compute gradients for both labels
            batch_size=batch_size,
        )
        
        # Store chunk features
        all_feature_dicts.append(chunk_features)
        
        # Clear GPU cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  ✓ Completed chunk {chunk_idx + 1}/{num_chunks}")
    
    # Concatenate all chunks
    print("Concatenating features from all chunks...")
    final_features = {}
    
    for feat_name in all_feature_dicts[0].keys():
        chunk_arrays = [chunk_features[feat_name] for chunk_features in all_feature_dicts]
        final_features[feat_name] = np.vstack(chunk_arrays)
    
    # Clear chunk data
    del all_feature_dicts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Add to DataFrame
    result_df = df.copy()
    
    for feat_name, feat_array in final_features.items():
        if feat_array.ndim == 2:
            # Multi-dimensional features - add as columns
            # Each column gets a unique name based on feature type and dimension
            for i in range(feat_array.shape[1]):
                # Use shorter prefix for gradient features
                if "grad" in feat_name:
                    result_df[f"bert_{feat_name}_{i}"] = feat_array[:, i]
                else:
                    result_df[f"bert_{feat_name}_{i}"] = feat_array[:, i]
        else:
            # Single-dimensional features
            result_df[f"bert_{feat_name}"] = feat_array
    
    print(f"✓ Added {len([c for c in result_df.columns if c.startswith('bert_')])} BERT feature columns")
    
    return result_df

