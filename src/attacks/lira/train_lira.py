import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from attacks.lira.utils import (get_loaders_from_indices, 
                                get_retain_forget_val_indices)

from unlearning.main import UnlearnApp


class UnlearnAppForLiRA(UnlearnApp):
    def __init__(self, config, unlearner_name):
        super().__init__(config)
        self.unlearner_name = unlearner_name
    
    def run(self, dataloaders, unlearning=True):
        print("running...")
        # Step 1: Initialize the model
        original_model = self.load_model()

        # Step 2: Unlearning
        unlearner = self.initialize_unlearner()
        if unlearning:
            print("f{self.unlearner_name} initialized")
        else:
            print("finetuner initialized")
        unlearned_model, losses = unlearner.unlearn(
            original_model,
            data_dict=dataloaders,
            verbose=self.verbose,
            **self.unlearn_params,
        )
        if unlearning:
            print("model unlearned")
        else:
            print("model finetuned")
        return unlearned_model


def run(
    config: DictConfig,
    model_name: str,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
    output_dir: Path,
) -> None:
    models_path = output_dir / unlearner / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    if os.listdir(models_path):
        print(f"{unlearner} models are already generated. Skipping.")
        return

    if unlearner in ["original", "naive"]:
        unlearner_name = "finetune"
    else:
        unlearner_name = unlearner
    app = UnlearnAppForLiRA(config, unlearner_name)

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            retain, forget, val = get_retain_forget_val_indices(
                lira_path=output_dir / "splits",
                split_ndx=split_ndx,
                forget_ndx=forget_ndx,
            )
            if unlearner == "original":
                retain = np.concatenate([retain, forget])

            loaders = get_loaders_from_indices(
                config=config,
                indices=[retain, forget, val],
            )

            if unlearner not in ["original", "naive"]:
                app.model_ckpt_path = (
                    output_dir
                    / "original"
                    / "models"
                    / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
                )
            unlearned_model = app.run(loaders)
            torch.save(
                {"model_state_dict": unlearned_model.state_dict()},
                output_dir
                / unlearner
                / "models"
                / f"{config.model.name}_{split_ndx}_{forget_ndx}.pth",
            )
