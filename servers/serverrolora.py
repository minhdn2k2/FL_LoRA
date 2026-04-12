import copy

from clients.clientrolora import ClientRoLoRA
from servers.serverbase import BaseServer
from utils.utils_model import configure_rolora_round


class ServerRoLoRA(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.setup_clients(args, ClientRoLoRA)
        print("Finished creating RoLoRA server and clients.")

    def send_model(self, client_id: int, round_idx: int) -> None:
        local_model = copy.deepcopy(self.global_model)
        configure_rolora_round(local_model, round_idx)
        self.clients[client_id].model = local_model

    def train(self):
        print(
            f"Start training | algorithm={self.algorithm} | task={self.data_obj.task_name} | "
            f"model={self.data_obj.model_name}"
        )

        for round_idx in range(self.global_rounds):
            phase = "update B / freeze A" if (round_idx % 2 == 0) else "update A / freeze B"
            print(
                f"\n---------------- Round {round_idx + 1}/{self.global_rounds} "
                f"({phase}) ----------------"
            )
            self.selected_clients = self.select_clients(round_idx)

            for client_id in self.selected_clients:
                self.send_model(client_id, round_idx)
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