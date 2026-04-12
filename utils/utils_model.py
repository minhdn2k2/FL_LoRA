from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification


StateDict = OrderedDict[str, torch.Tensor]


def parse_target_modules(target_modules: str) -> Optional[list[str]]:
    """Parse a comma-separated target_modules string into a list."""
    if not target_modules:
        return None
    items = [item.strip() for item in target_modules.split(",") if item.strip()]
    return items if items else None


def build_base_sequence_classifier(args, num_labels: int):
    model_kwargs = {"num_labels": num_labels}
    if args.model_cache:
        model_kwargs["cache_dir"] = args.model_cache
    return AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)


def build_lora_sequence_classifier(args, num_labels: int):
    """
    Build a LoRA-augmented sequence-classification model for GLUE tasks.

    For FFA-LoRA, the LoRA-A factor is frozen immediately after PEFT wrapping.
    """
    model_kwargs = {"num_labels": num_labels}
    if args.model_cache:
        model_kwargs["cache_dir"] = args.model_cache

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)

    peft_kwargs = dict(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    target_modules = parse_target_modules(args.target_modules)
    if target_modules is not None:
        peft_kwargs["target_modules"] = target_modules

    model = get_peft_model(model, LoraConfig(**peft_kwargs))
    return model

# ----------------------------- Model Function -----------------------------

def configure_ffa_model(model):
    """
    FFA setup:
    1. Freeze LoRA-A
    2. Zero-initialize LoRA-B
    """
    for name, p in model.named_parameters():
        if "lora_A" in name:
            p.requires_grad = False
        elif "lora_B" in name:
            p.data.zero_()

    return model

# --------------------------------------------------------------------------

# ----------------------------- Model Function -----------------------------


def freeze_lora_A(model) -> None:
    """
    Freeze all LoRA-A parameters in the model.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False


def freeze_lora_B(model) -> None:
    """
    Freeze all LoRA-B parameters in the model.
    """
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.requires_grad = False            


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> StateDict:
    """
    Create a detached CPU copy of a state dictionary.

    Each tensor is detached from the current computation graph, moved to
    CPU memory, and cloned so that the returned state is fully independent
    from the source tensors.

    Args:
        state_dict:
            A mapping from parameter names to tensors.

    Returns:
        StateDict:
            A new ordered state dictionary whose tensors are detached CPU
            clones of the input tensors.
    """
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


def load_partial_state_dict(model, partial_state: Dict[str, torch.Tensor]) -> None:
    """
    Load a partial state dictionary into a model.

    The loading is performed with ``strict=False`` so that only the provided
    subset of parameters is updated. This is useful for selectively loading
    shared or local LoRA parameters without requiring a full model checkpoint.

    Args:
        model:
            The target PyTorch model.
        partial_state:
            A partial state dictionary containing only the parameters to load.
    """
    if not partial_state:
        return
    model.load_state_dict(partial_state, strict=False)


def extract_trainable_state_dict(model) -> StateDict:
    """
    Extract all trainable parameters from the model as a detached CPU state dict.

    Args:
        model:
            The source PyTorch model.

    Returns:
        StateDict:
            An ordered state dictionary containing only the trainable
            parameters.
    """
    selected_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    full_state = model.state_dict()
    return OrderedDict(
        (name, full_state[name].detach().cpu().clone())
        for name in full_state.keys()
        if name in selected_names
    )


def attach_lora_sequence_classifier(base_model, args):
    peft_kwargs = dict(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    target_modules = parse_target_modules(args.target_modules)
    if target_modules is not None:
        peft_kwargs["target_modules"] = target_modules

    return get_peft_model(base_model, LoraConfig(**peft_kwargs))


def configure_rolora_round(model, round_idx: int):
    """
    Odd communication round (1-based): freeze A, update B.
    Even communication round (1-based): freeze B, update A.

    Since round_idx is 0-based in code:
    - round_idx % 2 == 0  -> round 1, 3, 5, ... -> update B
    - round_idx % 2 == 1  -> round 2, 4, 6, ... -> update A
    """
    update_B = (round_idx % 2 == 0)

    for name, p in model.named_parameters():
        if "lora_A" in name:
            p.requires_grad = not update_B
        elif "lora_B" in name:
            p.requires_grad = update_B
        elif ("classifier" in name) or ("score" in name):
            p.requires_grad = True




