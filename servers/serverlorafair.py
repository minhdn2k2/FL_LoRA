from collections import OrderedDict

import torch
import torch.nn.functional as F

from clients.clientlorafair import ClientLoRAFAIR
from servers.serverbase import BaseServer


class ServerLoRAFAIR(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.setup_clients(args, ClientLoRAFAIR)

        # server-side refinement hyperparameters
        self.fair_refine_steps = args.fair_refine_steps
        self.fair_refine_lr = args.fair_refine_lr
        self.fair_lambda = args.fair_lambda

        print("Finished creating server and clients for LoRA-FAIR.")

    def _get_lora_pairs(self, state_dict):
        pairs = []
        for key in state_dict.keys():
            if "lora_A" in key:
                b_key = key.replace("lora_A", "lora_B")
                if b_key in state_dict:
                    pairs.append((key, b_key))
        return pairs
    
    def _flatten_for_product(self, A, B):
        # A: [r, in], B: [out, r] for Linear LoRA
        A2 = A.view(A.size(0), -1)
        B2 = B.view(B.size(0), -1)
        return A2, B2
    
    def _refine_lora_B(self, avg_A, avg_B, ideal_product):
        """
        Solve the LoRA-FAIR refinement:
            min 1 - cosine( ideal_product, (avg_B + delta_B) @ avg_A )
                + lambda * ||delta_B||^2
        """
        delta_B = torch.zeros_like(avg_B, requires_grad=True)
        optimizer = torch.optim.SGD([delta_B], lr=self.fair_refine_lr)

        avg_A_2d, avg_B_2d = self._flatten_for_product(avg_A, avg_B)
        target = ideal_product.detach()

        for _ in range(self.fair_refine_steps):
            optimizer.zero_grad()

            corrected_B = avg_B_2d + delta_B.view(avg_B_2d.shape)
            recon = corrected_B @ avg_A_2d

            cos = F.cosine_similarity(
                recon.reshape(1, -1),
                target.reshape(1, -1),
                dim=1,
            ).mean()

            loss = (1.0 - cos) + self.fair_lambda * torch.norm(delta_B, p=2) ** 2
            loss.backward()
            optimizer.step()

        return (avg_B_2d + delta_B.detach().view(avg_B_2d.shape)).view_as(avg_B)

    def aggregate_lora_fair(self):
        """
        LoRA-FAIR:
        1) weighted-average all trainable params
        2) for each LoRA pair (A, B), refine averaged B with a server-side correction
           so that corrected_B @ averaged_A better matches the ideal weighted average
           of local products B_k @ A_k
        """
        if len(self.uploaded_states) == 0:
            raise ValueError("No uploaded states to aggregate. Call receive_models() first.")

        # Step 1: FedAvg all uploaded trainable params
        agg_state = OrderedDict()
        for name in self.uploaded_states[0].keys():
            agg_state[name] = torch.zeros_like(self.uploaded_states[0][name])

        for weight, client_state in zip(self.uploaded_weights, self.uploaded_states):
            for name, tensor in client_state.items():
                agg_state[name] += weight * tensor

        # Step 2: LoRA-FAIR correction on B
        print("Server starts refinement for LoRA_B .....")
        lora_pairs = self._get_lora_pairs(agg_state)

        for a_key, b_key in lora_pairs:
            avg_A = agg_state[a_key]
            avg_B = agg_state[b_key]

            # ideal global update = weighted average of local products B_k @ A_k
            ideal_product = None
            for weight, client_state in zip(self.uploaded_weights, self.uploaded_states):
                A_k = client_state[a_key]
                B_k = client_state[b_key]
                A_k_2d, B_k_2d = self._flatten_for_product(A_k, B_k)
                product_k = B_k_2d @ A_k_2d

                if ideal_product is None:
                    ideal_product = weight * product_k
                else:
                    ideal_product += weight * product_k

            corrected_B = self._refine_lora_B(avg_A, avg_B, ideal_product)
            agg_state[b_key] = corrected_B

        return agg_state

    def train(self):
        print(
            f"Start training | algorithm={self.algorithm} | task={self.data_obj.task_name} | "
            f"model={self.data_obj.model_name}"
        )

        for round_idx in range(self.global_rounds):
            print(f"\n---------------- Round {round_idx + 1}/{self.global_rounds} ----------------")
            self.selected_clients = self.select_clients(round_idx)

            for client_id in self.selected_clients:
                self.send_model(client_id)
                self.clients[client_id].train()

            self.receive_models()
            agg_state = self.aggregate_lora_fair()
            self.update_global_model(agg_state)

            self.empty_cache()

            if ((round_idx + 1) % self.eval_every) == 0:
                acc, val_loss = self.evaluate()
                print(
                    f"[Round {round_idx + 1:03d}] "
                    f"{self.metric_name}={acc:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"best_{self.metric_name}={self.best_metric:.4f}"
                )