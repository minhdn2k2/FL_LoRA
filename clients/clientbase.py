import copy
import itertools

import torch
from torch.utils.data import DataLoader

class Client:
    def __init__(self, args, client_id: int, train_dataset):
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.local_steps = args.local_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.num_workers = args.num_workers

        self.train_dataset = train_dataset
        self.num_samples = len(self.train_dataset)

        self.model = None
        self.upload_state = None

        self.amp_device_type = "cuda" if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
        self.use_amp = bool(args.fp16 and self.amp_device_type == "cuda")
        self.pin_memory = bool(torch.cuda.is_available() and "cuda" in args.device)

    def load_train_data(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )