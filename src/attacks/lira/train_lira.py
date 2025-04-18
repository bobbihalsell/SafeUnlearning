import os
from pathlib import Path

import numpy as np
import torch
from attacks.lira.utils import get_loaders_from_indices, get_retain_forget_val_indices

from unlearning.main import UnlearnApp


class UnlearnAppForLiRA(UnlearnApp):
    def __init__(self, config, unlearner_name):
        super().__init__(config)
        self.unlearner_name = unlearner_name

    def run(self, dataloaders):
        print("running...")
        # Step 1: Initialize the model
        original_model = self.load_model()

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
        return unlearned_model


def run(config, unlearner, root, save_path):
    num_splits = config.attack.cfg.num_splits
    num_forgets = config.attack.cfg.num_forgets

    name_save_path = root / save_path / "models"
    if not Path(name_save_path).exists():
        Path(name_save_path).mkdir(parents=True, exist_ok=True)

    app = UnlearnAppForLiRA(config, unlearner)

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            retain, forget, val = get_retain_forget_val_indices(
                lira_path=root / "splits",
                split_ndx=split_ndx,
                forget_ndx=forget_ndx,
            )
            if save_path == "original":
                retain = np.concatenate([retain, forget])

            loaders = get_loaders_from_indices(
                config=config,
                indices=[retain, forget, val],
            )

            if save_path not in ["original", "naive"]:
                app.model_ckpt_path = (
                    root
                    / "original"
                    / "models"
                    / f"{config.model.name}_{split_ndx}_{forget_ndx}.pth"
                )
            unlearned_model = app.run(loaders)
            torch.save(
                {"model_state_dict": unlearned_model.state_dict()},
                root
                / save_path
                / "models"
                / f"{config.model.name}_{split_ndx}_{forget_ndx}.pth",
            )
