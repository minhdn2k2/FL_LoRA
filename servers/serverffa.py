from clients.clientffa import ClientFFA
from servers.serverbase import BaseServer

from utils.utils_model import configure_ffa_model


class ServerFFA(BaseServer):
    def __init__(self, args):
        super().__init__(args)

        # Config for FFA-LoRA
        configure_ffa_model(self.global_model)

        self.setup_clients(args, ClientFFA)
        print("Finished creating server and clients for FFA-LoRA.")

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
            agg_state = self.aggregate_parameters()
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