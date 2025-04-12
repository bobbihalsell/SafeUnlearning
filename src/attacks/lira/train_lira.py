import os
from pathlib import Path

import numpy as np
import torch
from utils import get_loaders_from_indices, get_retain_forget_val_indices

from src.unlearning.main import UnlearnApp


class UnlearnAppForLiRA(UnlearnApp):
    def __init__(self, config):
        super().__init__(config)

    def run(self, dataloaders):
        print("running...")
        # Step 1: Initialize the model
        original_model = self.load_model_from_disk(self.model_ckpt_path)

        # Step 2: Unlearning
        unlearner = self.initialize_unlearner()
        print("unlearner initialized")
        unlearned_model, losses = unlearner.unlearn(
            original_model,
            data_dict=dataloaders,
            verbose=self.verbose,
            **self.unlearn_params,
        )
        print("model unlearned")

        # Step 3: Save the unlearned model
        directory = f"{self.output_dir}/unlearn/{self.unlearning_algorithm}"
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

        filepath = os.path.join(
            directory, f"{self.model_name}_{self.seed}_{self.model_type}.pt"
        )
        torch.save(unlearned_model.state_dict(), filepath)


def main(config, unlearner):
    model_name = config.model.name
    dataset_name = config.dataset.name
    root = config.root
    seed = config.seed
    num_splits = config.dataset.num_splits
    num_forgets = config.dataset.num_forgets

    name_save_path = root + "unlearn"
    if not Path(name_save_path).exists():
        Path(name_save_path).mkdir(parents=True, exist_ok=True)

    app = UnlearnAppForLiRA(config)

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            retain, forget, val = get_retain_forget_val_indices(
                lira_path=root / "splits",
                split_ndx=split_ndx,
                forget_ndx=forget_ndx,
            )

            retain_loader, forget_loader, val_loader = get_loaders_from_indices(
                root=root,
                indices=[retain, forget, val],
                config=config,
            )
            app.run([retain_loader, forget_loader, val_loader])
