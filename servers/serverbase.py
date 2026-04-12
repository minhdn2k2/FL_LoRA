from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from utils.utils_model import clone_state_dict, load_partial_state_dict


class BaseServer:
    def __init__(self, args):
        self.seed = args.seed
        self.device = args.device
        self.global_model = copy.deepcopy(args.global_model).to("cpu")
        self.data_obj = args.data_obj

        self.algorithm = args.algorithm
        self.n_clients = args.n_clients
        self.selected_ratio = args.selected_ratio
        self.global_rounds = args.global_rounds
        self.eval_every = args.eval_every
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers
        self.pin_memory = bool(torch.cuda.is_available() and "cuda" in args.device)

        self.lora_alpha = args.lora_alpha
        self.lora_r = args.lora_r

        self.metric_name = self.data_obj.metric_name
        self.is_regression = self.data_obj.is_regression

        self.clients = []
        self.selected_clients = []
        self.uploaded_states = []
        self.uploaded_weights = []
        self.uploaded_num_samples = []

        self.metric_hist: List[float] = []
        self.best_metric = -np.inf

        self.base_name = (
            f"{args.algorithm}_{args.task_raw_name}_{args.seed}_n_clients{args.n_clients}"
            f"LoRA_r{args.lora_r}_Dir{args.split_coef}_lr{args.lr}")
        self.save_dir = args.save_dir

        os.makedirs(os.path.join(self.save_dir, "plot"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "best_acc"), exist_ok=True)
        self.metric_hist_path = os.path.join(self.save_dir, f"plot/metric_hist_{self.base_name}.npy")
        self.best_metric_path = os.path.join(self.save_dir, f"best_acc/best_eval_acc_{self.base_name}.txt")


    def setup_clients(self, args, clientObj):
        self.clients = []
        for client_id in range(self.n_clients):
            client = clientObj(
                    args=args,
                    client_id=client_id,
                    train_dataset=self.data_obj.client_train_datasets[client_id])
            self.clients.append(client)


    def select_clients(self, round_idx: int) -> List[int]:
        rng = np.random.default_rng(self.seed + round_idx)
        num_selected = max(1, int(round(self.n_clients * self.selected_ratio)))
        selected = rng.choice(self.n_clients, size=num_selected, replace=False)
        return sorted(selected.tolist())


    def send_model(self, client_id: int) -> None:
        self.clients[client_id].model = copy.deepcopy(self.global_model)


    def receive_models(self) -> None:
        self.uploaded_states = []
        self.uploaded_weights = []
        self.uploaded_num_samples = []

        total_samples = 0

        for client_id in self.selected_clients:
            client = self.clients[client_id]

            if client.upload_state is None:
                raise ValueError(f"Client {client_id} has no upload_state. Did you call train()?") 
 
            self.uploaded_states.append(clone_state_dict(client.upload_state))
            self.uploaded_num_samples.append(client.num_samples)

            total_samples += client.num_samples

        if total_samples <= 0:
            raise ValueError("Total number of uploaded samples must be > 0.")
        
        self.uploaded_weights = [num / float(total_samples) for num in self.uploaded_num_samples]


    def aggregate_parameters(self) -> Dict[str, torch.Tensor]:
        if len(self.uploaded_states) == 0:
            raise ValueError("No uploaded states to aggregate. Call receive_models() first.")

        agg_state = {}

        for name in self.uploaded_states[0].keys():
            agg_state[name] = torch.zeros_like(self.uploaded_states[0][name])

        for weight, client_state in zip(self.uploaded_weights, self.uploaded_states):
            for name, tensor in client_state.items():
                agg_state[name] += weight * tensor

        return agg_state


    def update_global_model(self, agg_state: Dict[str, torch.Tensor]) -> None:
        if len(agg_state) == 0:
            print("Warning: agg_state is empty. Global model is not updated.")
            return

        load_partial_state_dict(self.global_model, agg_state)


    def empty_cache(self):
        for client_id in self.selected_clients:
            if self.clients[client_id].model is not None:
                self.clients[client_id].model = self.clients[client_id].model.to("cpu")
                self.clients[client_id].model = None

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()


    def evaluate(self) -> tuple[float, float]:
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()

        val_loader = DataLoader(
            self.data_obj.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("label")

                outputs = self.global_model(**batch, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                bs = labels.size(0)
                total_loss += float(loss.item()) * bs
                total_samples += bs

                if self.is_regression:
                    preds = logits.squeeze(-1)
                else:
                    preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        metric_dict = self.data_obj.compute_metric(
            predictions=all_preds,
            references=all_labels,
        )
        metric_value = float(metric_dict[self.metric_name])
        eval_loss = total_loss / float(total_samples)

        self.metric_hist.append(metric_value)
        np.save(self.metric_hist_path, np.asarray(self.metric_hist, dtype=np.float32))

        if metric_value > self.best_metric:
            self.best_metric = metric_value
            with open(self.best_metric_path, "w") as f:
                f.write(f"{self.best_metric:.4f}\n")

        self.global_model = self.global_model.to("cpu")
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.empty_cache()

        return metric_value, eval_loss

