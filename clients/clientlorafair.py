import itertools
import torch
from torch.amp import autocast, GradScaler

from clients.clientbase import Client
from utils.utils_model import extract_trainable_state_dict


class ClientLoRAFAIR(Client):
    def __init__(self, args, client_id, train_dataset):
        super().__init__(args=args, client_id=client_id, train_dataset=train_dataset)

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()

        train_loader = self.load_train_data()
        train_iter = itertools.cycle(train_loader)

        optimizer = torch.optim.SGD(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scaler = GradScaler("cuda", enabled=self.use_amp)

        total_loss = 0.0
        for _ in range(self.local_steps):
            batch = next(train_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label")

            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()

            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in self.model.parameters() if param.requires_grad],
                    max_norm=self.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())

        self.avg_local_loss = total_loss / float(self.local_steps)
        self.upload_state = extract_trainable_state_dict(self.model)

        self.model = self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"  client={self.client_id:03d} | samples={self.num_samples:5d} | "
            f"avg_local_loss={self.avg_local_loss:.4f}"
        )