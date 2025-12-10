from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize


class SurprisalComputer:
    def __init__(self, model_name: str = "gpt2", device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # Ensure a pad token exists to avoid shape issues
        if self.tokenizer.pad_token is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def text_nll(self, text: str, max_length: int = 512) -> float:
        # Guard against empty/whitespace text
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = "."
        enc = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        # Fallback if tokenizer produced empty sequence
        if input_ids.numel() == 0 or input_ids.shape[-1] == 0:
            enc = self.tokenizer(".", return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        if input_ids.shape[-1] == 0:
            return 0.0
        outputs = self.model(input_ids, attention_mask=attn, labels=input_ids)
        # outputs.loss is mean cross-entropy per token (nats if model uses natural log)
        return float(outputs.loss.detach().cpu().item())

    def scene_surprisal_features(self, text: str) -> dict:
        # Sentence-level NLL as proxy for surprisal; compute mean and std
        sentences = sent_tokenize(text or "") or [text or "."]
        values: List[float] = [self.text_nll(s) for s in sentences if len(s.strip()) > 0]
        if not values:
            values = [self.text_nll(text or ".")]
        arr = np.array(values, dtype=np.float32)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = float(std / mean) if mean != 0 else 0.0
        p75 = float(np.percentile(arr, 75))
        maxv = float(np.max(arr))
        # simple slope across sentence order
        if len(arr) >= 2:
            x = np.arange(len(arr), dtype=np.float32)
            x_mean = float(np.mean(x)); y_mean = mean
            num = float(np.sum((x - x_mean) * (arr - y_mean)))
            den = float(np.sum((x - x_mean) ** 2)) or 1.0
            slope = num / den
        else:
            slope = 0.0
        return {
            "surprisal_mean": mean,
            "surprisal_std": std,
            "surprisal_cv": cv,
            "surprisal_p75": p75,
            "surprisal_max": maxv,
            "surprisal_slope": float(slope),
        }


