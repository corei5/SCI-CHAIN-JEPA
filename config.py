"""
config.py
=========
Central configuration. Edit ONLY this file (or override via experiment scripts).

This project empirically validates: "Pretraining stronger generative models via
JEPA." Studies included:
  - Main: JEPA on/off during PRETRAINING (from random weights).
  - Ablation: design-choice on/off.
  - lambda sweep: how the JEPA weight affects quality.
  - Loss-dropout: speed vs quality tradeoff ("Faster LLM-JEPAs via loss dropout").
"""

from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class Config:
    # ----------------------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-0.5B"   # open, no HF login required

    # ----------------------------------------------------------------------
    # PRETRAINING vs FINE-TUNING
    #   from_scratch=True  -> random init, model learns from zero (PRETRAINING).
    #   from_scratch=False -> load pretrained weights (fine-tuning study).
    # ----------------------------------------------------------------------
    from_scratch: bool = True

    # ----------------------------------------------------------------------
    # THE CRITICAL SWITCH (JEPA on/off) + the headline metric
    # ----------------------------------------------------------------------
    jepa_enabled: bool = True
    eval_perplexity: bool = True
    perplexity_eval_examples: int = 500

    # ----------------------------------------------------------------------
    # DATA (LAION PubMed)
    # ----------------------------------------------------------------------
    hf_dataset: str = "ai4sci-tib/LAION_pubmed-fulltext"
    stages: List[str] = field(default_factory=lambda: [
        "title", "hypothesis", "method", "experiment", "result", "conclusion"])
    stage_columns: Dict[str, str] = field(default_factory=lambda: {
        "title":      "summary_title",
        "hypothesis": "research_question_hypothesis",
        "method":     "methodological_details",
        "experiment": "procedures_architectures",
        "result":     "key_results",
        "conclusion": "interpretation_implications",
    })
    year_column: str = "oa_year"
    citation_column: str = "oa_cited_by_count"
    field_column: str = "field_subfield"
    id_column: str = "paper_id"

    train_cutoff_year: int = 2024
    test_year: int = 2025
    random_split_test_frac: float = 0.1

    chain_edges: List[tuple] = field(default_factory=lambda: [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 4)])
    trivial_edge_cosine_threshold: float = 0.95
    n_distractors: int = 4
    max_stage_chars: int = 1000

    # ----------------------------------------------------------------------
    # IMBALANCE (F2 labels)
    # ----------------------------------------------------------------------
    max_papers_per_field: int = 0
    use_log_citations: bool = True
    f2_label_strategy: str = "field_quartile"
    f2_top_quantile: float = 0.75

    # ----------------------------------------------------------------------
    # DESIGN CHOICES (toggled by the ablation study)
    # ----------------------------------------------------------------------
    use_pred_tokens: bool = True          # [PRED] tokens vs plain last token
    num_pred_tokens: int = 4
    use_long_range_edge: bool = True       # include method->result edge (2,4)
    use_trivial_edge_filter: bool = True   # skip already-easy edges
    jepa_loss_type: str = "cosine"         # "cosine" | "mse"

    # ----------------------------------------------------------------------
    # LOSS DROPOUT ("Faster LLM-JEPAs via loss dropout")
    #   Fraction of JEPA edges randomly skipped each step. Higher = faster but
    #   potentially weaker. The dropout study sweeps this value.
    # ----------------------------------------------------------------------
    loss_dropout: float = 0.125

    # ----------------------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------------------
    streaming: bool = True
    max_train_examples: int = 5_000        # SMALL default; raise for real runs
    max_test_examples: int = 1_000
    shard_dir: str = "./data_shards"
    num_workers: int = 2
    shuffle_buffer: int = 2_000

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = False

    grad_accum_steps: int = 4
    use_amp: bool = True
    gradient_checkpointing: bool = True

    max_stage_tokens: int = 64
    lambda_jepa: float = 1.0
    lr: float = 6e-4
    batch_size: int = 4
    num_epochs: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    seed: int = 0

    save_every_steps: int = 1000
    log_every_steps: int = 20
    resume_from: str = ""
    max_train_steps: int = 500             # SMALL default; raise for real runs

    # ----------------------------------------------------------------------
    # SYSTEM
    # ----------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    output_dir: str = "./outputs"


CFG = Config()


# ----------------------------------------------------------------------------
# SCALE TIERS -- call set_tier("P1") etc. from a script to scale up.
# ----------------------------------------------------------------------------
def set_tier(name: str):
    if name == "smoke":   # ~2 min, catches errors
        CFG.max_train_examples = 2000; CFG.max_test_examples = 200
        CFG.max_train_steps = 30; CFG.batch_size = 2; CFG.warmup_steps = 5
        CFG.perplexity_eval_examples = 50
    elif name == "P1":    # pretraining proof-of-concept, 1-2 GPUs
        CFG.max_train_examples = 200_000; CFG.max_test_examples = 3000
        CFG.max_train_steps = 50_000; CFG.batch_size = 16
        CFG.grad_accum_steps = 16; CFG.warmup_steps = 2000
        CFG.num_epochs = 3; CFG.perplexity_eval_examples = 1000
    elif name == "P2":    # credible pretraining, multi-GPU multi-day
        CFG.max_train_examples = 1_000_000; CFG.max_test_examples = 3000
        CFG.max_train_steps = 200_000; CFG.batch_size = 16
        CFG.grad_accum_steps = 32; CFG.warmup_steps = 4000
        CFG.num_epochs = 5; CFG.perplexity_eval_examples = 1000
    else:
        raise ValueError(f"unknown tier {name}")
    print(f"[config] tier set to {name}")
