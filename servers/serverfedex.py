from collections import OrderedDict

import torch

from clients.clientfedex import ClientFedEx
from servers.serverbase import BaseServer
from utils.utils_model import load_partial_state_dict


class ServerFedEx(BaseServer):
    def __init__(self, args):
        super().__init__(args)

        self.setup_clients(args, ClientFedEx)
        print("Finished creating FedEx-LoRA server and clients.")

    def _lora_A_to_B_key(self, a_key: str) -> str:
        return a_key.replace("lora_A", "lora_B")

    def _lora_A_to_base_key(self, a_key: str) -> str:
        return a_key.replace("lora_A.default.weight", "base_layer.weight")

    def aggregate_parameters(self):
        if len(self.uploaded_states) == 0:
            raise ValueError("No uploaded states to aggregate. Call receive_models() first.")

        # 1) standard weighted average of all uploaded trainable params
        agg_state = OrderedDict()
        for name in self.uploaded_states[0].keys():
            agg_state[name] = torch.zeros_like(self.uploaded_states[0][name])

        for weight, client_state in zip(self.uploaded_weights, self.uploaded_states):
            for name, tensor in client_state.items():
                agg_state[name] += weight * tensor

        # 2) exact residual correction for each LoRA layer
        updated_partial_state = OrderedDict(agg_state)
        current_state = self.global_model.state_dict()

        scaling = self.lora_alpha / float(self.lora_r)

        lora_A_keys = [k for k in agg_state.keys() if "lora_A" in k]
        for a_key in lora_A_keys:
            b_key = self._lora_A_to_B_key(a_key)
            base_key = self._lora_A_to_base_key(a_key)

            # ideal average low-rank product
            M = None
            for weight, client_state in zip(self.uploaded_weights, self.uploaded_states):
                A_k = client_state[a_key]
                B_k = client_state[b_key]
                prod = B_k @ A_k
                M = weight * prod if M is None else M + weight * prod

            A_avg = agg_state[a_key]
            B_avg = agg_state[b_key]
            residue = M - (B_avg @ A_avg)

            updated_partial_state[base_key] = (
                current_state[base_key].detach().cpu().clone() + scaling * residue
            )

        return updated_partial_state

    def train(self):
        print(
            f"Start training | algorithm={self.algorithm} | task={self.data_obj.task_name} | "
            f"model={self.data_obj.model_name}")

        for round_idx in range(self.global_rounds):
            print(f"\n---------------- Round {round_idx + 1}/{self.global_rounds} ----------------")
            self.selected_clients = self.select_clients(round_idx)

            for client_id in self.selected_clients:
                self.send_model(client_id)
                self.clients[client_id].train()

            self.receive_models()
            agg_partial_state = self.aggregate_parameters()
            self.update_global_model(agg_partial_state)
            self.empty_cache()

            if ((round_idx + 1) % self.eval_every) == 0:
                acc, val_loss = self.evaluate()
                print(
                    f"[Round {round_idx + 1:03d}] "
                    f"{self.metric_name}={acc:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"best_{self.metric_name}={self.best_metric:.4f}"
                )