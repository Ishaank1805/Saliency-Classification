"""
Classification Models: SVM and Llama-based classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import os


class SVMClassifier:
    """SVM classifier with hyperparameter tuning."""
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: Optional[str] = "balanced",
        tune_hyperparameters: bool = True,
    ):
        """
        Args:
            kernel: SVM kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
            class_weight: Class weight strategy
            tune_hyperparameters: Whether to tune hyperparameters
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.tune_hyperparameters = tune_hyperparameters
        self.model = None
        self.best_params = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> Dict[str, float]:
        """
        Train SVM classifier.
        
        Returns:
            Training metrics
        """
        if self.tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["rbf", "linear", "poly"],
            }
            
            # Base estimator
            base_svm = SVC(class_weight=self.class_weight, probability=True)
            
            # Grid search
            print("Tuning SVM hyperparameters...")
            grid_search = GridSearchCV(
                base_svm,
                param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV F1: {grid_search.best_score_:.4f}")
        else:
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                class_weight=self.class_weight,
                probability=True,
            )
            self.model.fit(X, y)
        
        # Training metrics
        train_pred = self.model.predict(X)
        train_f1 = f1_score(y, train_pred)
        
        return {
            "train_f1": train_f1,
            "best_params": self.best_params,
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate on test set."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)


class LlamaClassifier:
    """Llama-based classifier using fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 256,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to use
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Prepare dataset for training."""
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        
        dataset = Dataset.from_dict({"text": texts, "label": labels})
        dataset = dataset.map(tokenize, batched=True)
        return dataset
    
    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Fine-tune Llama model.
        
        Returns:
            Training metrics
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        self.model.to(self.device)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(texts, labels)
        
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./llama_checkpoints",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Training Llama model...")
        train_result = self.trainer.train()
        
        return {
            "train_loss": train_result.training_loss,
        }
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict class labels."""
        dataset = self.prepare_dataset(texts, [0] * len(texts))  # Dummy labels
        predictions = self.trainer.predict(dataset)
        
        # Get predicted labels
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        return pred_labels
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        dataset = self.prepare_dataset(texts, [0] * len(texts))  # Dummy labels
        predictions = self.trainer.predict(dataset)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
        return probs
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate on test set."""
        y_pred = self.predict(texts)
        y_proba = self.predict_proba(texts)[:, 1]
        
        return {
            "precision": precision_score(labels, y_pred, zero_division=0),
            "recall": recall_score(labels, y_pred, zero_division=0),
            "f1": f1_score(labels, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, y_pred).tolist(),
        }
    
    def save(self, path: str):
        """Save model to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)

