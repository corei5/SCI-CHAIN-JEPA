"""
model.py -- base LM for NTP + JEPA stage prediction.
Supports random init (pretraining) or pretrained load (fine-tuning), [PRED]
tokens, cosine/MSE JEPA losses, gradient checkpointing, optional LoRA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig,
                          BitsAndBytesConfig)
from config import CFG


class SciChainJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if CFG.from_scratch:
            arch = AutoConfig.from_pretrained(CFG.model_name)
            self.lm = AutoModelForCausalLM.from_config(arch, torch_dtype=CFG.dtype)
            print(f"[model] PRETRAINING (random init from {CFG.model_name} arch).")
        else:
            quant = None
            if CFG.load_in_4bit:
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=CFG.dtype, bnb_4bit_use_double_quant=True)
            self.lm = AutoModelForCausalLM.from_pretrained(
                CFG.model_name, torch_dtype=CFG.dtype, quantization_config=quant)
            print(f"[model] FINE-TUNING (loaded pretrained {CFG.model_name}).")

        if CFG.use_pred_tokens:
            pred_tokens = [f"[PRED{i}]" for i in range(CFG.num_pred_tokens)]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": pred_tokens})
            self.lm.resize_token_embeddings(len(self.tokenizer))

        if CFG.gradient_checkpointing:
            self.lm.gradient_checkpointing_enable()
            self.lm.config.use_cache = False

        if CFG.use_lora and not CFG.from_scratch:
            from peft import (LoraConfig, get_peft_model,
                              prepare_model_for_kbit_training)
            if CFG.load_in_4bit:
                self.lm = prepare_model_for_kbit_training(self.lm)
            lora = LoraConfig(
                r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])
            self.lm = get_peft_model(self.lm, lora)
            self.lm.print_trainable_parameters()

        self.hidden_size = self.lm.config.hidden_size

    def _last_token_emb(self, hidden, attn_mask):
        lengths = attn_mask.sum(dim=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return F.normalize(hidden.gather(1, idx).squeeze(1), dim=-1)

    def encode(self, texts):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=CFG.max_stage_tokens
                             ).to(self.lm.device)
        out = self.lm(**enc, output_hidden_states=True)
        return self._last_token_emb(out.hidden_states[-1], enc["attention_mask"])

    def predict_stage(self, context_text):
        if CFG.use_pred_tokens:
            pred_str = "".join(f"[PRED{i}]" for i in range(CFG.num_pred_tokens))
            prompts = [ctx + " " + pred_str for ctx in context_text]
        else:
            prompts = list(context_text)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=CFG.max_stage_tokens * 5
                             ).to(self.lm.device)
        out = self.lm(**enc, output_hidden_states=True)
        return self._last_token_emb(out.hidden_states[-1], enc["attention_mask"])

    def ntp_loss(self, full_texts):
        enc = self.tokenizer(full_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=CFG.max_stage_tokens * 6
                             ).to(self.lm.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return self.lm(**enc, labels=labels).loss
