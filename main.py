from __future__ import annotations

import argparse
import random
import warnings

import numpy as np
import torch

from utils.utils_dataset import GLUEFederatedData
from utils.utils_model import build_lora_sequence_classifier

from servers.serverffa import ServerFFA
from servers.serverfedit import ServerFedIT
from servers.serverlorafair import ServerLoRAFAIR
from servers.serverfedex import ServerFedEx
from servers.serverrolora import ServerRoLoRA

warnings.simplefilter("ignore")

TASK_CHOICES = [
    "cola",
    "mnli",
    "mnli-m",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]


def normalize_task_name(task_name: str) -> tuple[str, bool]:
    """
    Normalize the CLI task name into the internal GLUE task name plus the
    MNLI validation-split flag.

    Returns:
        (canonical_task_name, matched)
    """
    task_name = task_name.lower()
    if task_name in {"mnli", "mnli-m"}:
        return "mnli", True
    if task_name == "mnli-mm":
        return "mnli", False
    return task_name, True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal FFA-LoRA for GLUE")

    # Reproducibility / device
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Use automatic mixed precision during local training.")

    # Algorithm
    parser.add_argument("--algorithm", type=str, default="ffa", choices=["ffa", "fedit", "lorafair", 
                                                                         "fedex", "rolora"])

    # Dataset / pretrained model
    parser.add_argument("--task_name", type=str, default="mnli-m", choices=TASK_CHOICES, help="GLUE task name.")
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--model_cache", type=str, default="data/cache")
    parser.add_argument("--max_length", type=int, default=128)

    # Federated learning
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--global_rounds", type=int, default=1000)
    parser.add_argument("--selected_ratio", type=float, default=1.0)
    parser.add_argument("--split_rule", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--split_coef", type=float, default=0.5)

    # Local optimization
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="",
        help="Optional comma-separated target_modules for PEFT. Empty means PEFT model defaults.",
    )

    # LoRA-FAIR specific parameters
    parser.add_argument("--fair_refine_steps", type=int, default=500)
    parser.add_argument("--fair_refine_lr", type=float, default=0.01)
    parser.add_argument("--fair_lambda", type=float, default=1.0)

    # Logging / saving
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="outputs")

    args = parser.parse_args()
    args.task_raw_name = args.task_name
    args.task_name, args.matched = normalize_task_name(args.task_name)

    set_seed(args.seed)

    args.data_obj = GLUEFederatedData(args)
    args.num_labels = args.data_obj.num_labels
    args.global_model = build_lora_sequence_classifier(args, num_labels=args.num_labels)

    if args.algorithm == "ffa":
        server = ServerFFA(args)
    elif args.algorithm == "fedit":
        server = ServerFedIT(args)
    elif args.algorithm == "lorafair":
        server = ServerLoRAFAIR(args)
    elif args.algorithm == "fedex":
        server = ServerFedEx(args)
    elif args.algorithm == "rolora":
        server = ServerRoLoRA(args)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    server.train()