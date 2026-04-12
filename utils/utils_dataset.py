from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


task_to_keys: Dict[str, Tuple[str, str | None]] = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

primary_metric_map: Dict[str, str] = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
}


def summarize_federated_split(
    datasets: List[Dataset],
    label_list: List[str] | None = None,
    split_name: str = "train",
) -> None:
    """Print per-client sample counts and label distributions."""
    print(f"\n[{split_name} distribution by client]")
    for cid, ds in enumerate(datasets):
        labels = np.array(ds["label"])
        cnt = Counter(labels.tolist())

        if label_list is not None:
            pretty_cnt = {}
            for k, v in sorted(cnt.items()):
                if isinstance(k, (int, np.integer)) and 0 <= int(k) < len(label_list):
                    pretty_cnt[label_list[int(k)]] = v
                else:
                    pretty_cnt[k] = v
        else:
            pretty_cnt = dict(sorted(cnt.items()))

        print(f"Client {cid:02d}: n={len(labels)} | label_count={pretty_cnt}")


def summarize_shared_split(
    dataset: Dataset,
    label_list: List[str] | None = None,
    split_name: str = "val(shared)",
) -> None:
    """Print sample count and label distribution for a shared validation set."""
    labels = np.array(dataset["label"])
    cnt = Counter(labels.tolist())

    if label_list is not None:
        pretty_cnt = {}
        for k, v in sorted(cnt.items()):
            if isinstance(k, (int, np.integer)) and 0 <= int(k) < len(label_list):
                pretty_cnt[label_list[int(k)]] = v
            else:
                pretty_cnt[k] = v
    else:
        pretty_cnt = dict(sorted(cnt.items()))

    print(f"\n[{split_name}] n={len(labels)} | label_count={pretty_cnt}")


class GLUEFederatedData:
    """
    Minimal GLUE loader for FFA-LoRA.

    Design choices:
    - The GLUE train split is partitioned across clients.
    - The validation split is kept at the server for global evaluation.
    - MNLI uses validation_matched or validation_mismatched based on args.matched.
    """

    def __init__(self, args):
        self.task_name: str = args.task_name.lower()
        if self.task_name not in task_to_keys:
            raise ValueError(f"Unsupported GLUE task: {self.task_name}")

        self.model_name: str = args.model_name
        self.data_root: str = args.data_root
        self.model_cache: str = args.model_cache
        self.max_length: int = args.max_length
        self.n_clients: int = args.n_clients
        self.partition: str = args.split_rule.lower()
        self.dirichlet_alpha: float = args.split_coef
        self.seed: int = args.seed
        self.matched: bool = args.matched

        self.tokenizer = None
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.client_train_datasets: List[Dataset]
        self.label_list: List[str] | None = None
        self.num_labels: int
        self.is_regression: bool
        self.metric = evaluate.load("glue", self.task_name)
        self.metric_name: str = primary_metric_map[self.task_name]

        self._load_and_tokenize()
        self.client_train_datasets = self._build_client_train_splits()

        summarize_federated_split(self.client_train_datasets, self.label_list, split_name="train")
        summarize_shared_split(self.val_dataset, self.label_list, split_name="val(shared)")

    def _load_and_tokenize(self) -> None:
        raw = load_dataset("glue", self.task_name, cache_dir=self.data_root)
        self.is_regression = self.task_name == "stsb"

        if not self.is_regression:
            self.label_list = raw["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.label_list = None
            self.num_labels = 1

        tokenizer_kwargs = {}
        if self.model_cache:
            tokenizer_kwargs["cache_dir"] = self.model_cache
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)

        sentence1_key, sentence2_key = task_to_keys[self.task_name]

        def preprocess(batch: Dict[str, Sequence[str]]) -> Dict[str, Sequence[int]]:
            texts = (batch[sentence1_key],)
            if sentence2_key is not None:
                texts = (batch[sentence1_key], batch[sentence2_key])
            return self.tokenizer(
                *texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        encoded = raw.map(preprocess, batched=True, load_from_cache_file=True)
        columns = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in encoded["train"].column_names:
            columns.append("token_type_ids")
        encoded.set_format(type="torch", columns=columns)

        self.train_dataset = encoded["train"]
        if self.task_name == "mnli":
            self.val_dataset = encoded["validation_matched" if self.matched else "validation_mismatched"]
        else:
            self.val_dataset = encoded["validation"]

    def compute_metric(self, predictions, references) -> Dict[str, float]:
        return self.metric.compute(predictions=predictions, references=references)

    def _build_client_train_splits(self) -> List[Dataset]:
        train_labels = np.asarray(self.train_dataset["label"])
        train_indices = partition_indices(
            labels=train_labels,
            n_clients=self.n_clients,
            partition=self.partition,
            alpha=self.dirichlet_alpha,
            seed=self.seed,
            regression_fallback=self.is_regression,
        )
        return [self.train_dataset.select(idx.tolist()) for idx in train_indices]


def partition_indices(
    labels: np.ndarray,
    n_clients: int,
    partition: str,
    alpha: float,
    seed: int,
    regression_fallback: bool = False,
) -> List[np.ndarray]:
    """Partition sample indices across clients using IID or Dirichlet splitting."""
    rng = np.random.default_rng(seed)
    num_samples = len(labels)

    if num_samples == 0:
        return [np.array([], dtype=np.int64) for _ in range(n_clients)]

    if partition == "iid" or regression_fallback:
        shuffled = rng.permutation(num_samples)
        return [np.asarray(split, dtype=np.int64) for split in np.array_split(shuffled, n_clients)]

    if partition != "dirichlet":
        raise ValueError(f"Unsupported partition mode: {partition}")

    labels = labels.astype(np.int64)
    unique_labels = sorted(np.unique(labels).tolist())
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        rng.shuffle(label_idx)

        proportions = rng.dirichlet(
            alpha=np.full(shape=(n_clients,), fill_value=alpha, dtype=np.float64)
        )
        counts = np.floor(proportions * len(label_idx)).astype(int)

        diff = len(label_idx) - int(counts.sum())
        if diff > 0:
            for cid in np.argsort(proportions)[-diff:]:
                counts[cid] += 1
        elif diff < 0:
            for cid in np.argsort(proportions)[: -diff]:
                if counts[cid] > 0:
                    counts[cid] -= 1

        cursor = 0
        for cid, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            client_indices[cid].extend(label_idx[cursor: cursor + count].tolist())
            cursor += count

    for cid in range(n_clients):
        if len(client_indices[cid]) > 0:
            continue
        donor = int(np.argmax([len(v) for v in client_indices]))
        if len(client_indices[donor]) <= 1:
            continue
        moved_index = client_indices[donor].pop()
        client_indices[cid].append(moved_index)

    return [np.asarray(sorted(idx), dtype=np.int64) for idx in client_indices]
