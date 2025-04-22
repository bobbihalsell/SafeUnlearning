import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from lira_utils import (get_loaders_from_indices, 
                        get_retain_forget_val_indices)
from unlearning.main import UnlearnApp
from omegaconf import OmegaConf, DictConfig


class UnlearnAppForLiRA(UnlearnApp):
    def __init__(self, config, unlearner_name):
        # Convert config to dict if it's an OmegaConf object
        if not isinstance(config, dict):
            config = OmegaConf.to_container(config, resolve=True)
        
        # Update configuration
        config["model"]["pretrained"] = False
        config["unlearner"]["name"] = unlearner_name
        # Convert back to OmegaConf and initialize parent
        super().__init__(OmegaConf.create(config))
        self.unlearner_name = unlearner_name
      
    def run(self, dataloaders, unlearning=True):
        print("running...")
        # Step 1: Initialize the model
        original_model = self.load_model()

        # Step 2: Unlearning
        unlearner = self.initialize_unlearner()
        if unlearning:
            print(f"{self.unlearner_name} initialized")
        else:
            print("finetuner initialized")
        unlearned_model, _ = unlearner.unlearn(
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
    

def train_models(
    config: DictConfig,
    model_name: str,
    unlearner: str,
    num_splits: int,
    num_forgets: int,
    output_dir: Path,
) -> None:
    models_path = output_dir / unlearner / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    # No longer immediately return if directory has files
    # We'll check each file individually instead

    if unlearner in ["original", "naive"]:
        unlearner_name = "finetune"
    else:
        unlearner_name = unlearner
    print(f"Unlearner: {unlearner_name}")
    app = UnlearnAppForLiRA(config, unlearner_name)
    print('dataset       ', hasattr(app, "dataset_name"))
    total_models = num_splits * num_forgets
    model_num = 0
    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            model_num += 1
            
            # Check if this specific model already exists
            target_model_path = (
                output_dir
                / unlearner
                / "models"
                / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
            )
            
            if os.path.exists(target_model_path):
                print(f"Model {model_name}_{split_ndx}_{forget_ndx}.pth "
                      "already exists. Skipping.")
                continue
            
            # Update progress bar with current model information
            print(f"Split {split_ndx+1}/{num_splits}, Forget {forget_ndx+1}/"
                  f"{num_forgets}, Model {model_num}/{total_models}")
            
            try:
                retain, forget, val = get_retain_forget_val_indices(
                    lira_path=output_dir / "splits",
                    split_ndx=split_ndx,
                    forget_ndx=forget_ndx,
                )
                if unlearner == "original":
                    retain = np.concatenate([retain, forget])

                loaders = get_loaders_from_indices(
                    app.dataset_name,
                    app.dataset_save_dir,
                    [retain, forget, val],
                    app.batch_sizes,
                    app.num_workers,
                )
                unlearning = False
                if unlearner not in ["original", "naive"]:
                    unlearning = True
                    original_model_path = (
                        output_dir
                        / "original"
                        / "models"
                        / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
                    )
                    
                    # Check if original model exists before trying to load it
                    if not os.path.exists(original_model_path):
                        print(f"Warning: Original model {original_model_path} not found. Skipping.")
                        continue
                        
                    app.model_ckpt_path = original_model_path
                    
                unlearned_model = app.run(loaders, unlearning=unlearning)
                torch.save(
                    {"model_state_dict": unlearned_model.state_dict()},
                    target_model_path,
                )
                
            except Exception as e:
                print(f"Error processing model {model_name}_{split_ndx}_{forget_ndx}: {str(e)}")
                # Continue with next model rather than crashing completely
            


# def run(
#     config: DictConfig,
#     model_name: str,
#     unlearner: str,
#     num_splits: int,
#     num_forgets: int,
#     output_dir: Path,
# ) -> None:
#     models_path = output_dir / unlearner / "models"
#     models_path.mkdir(parents=True, exist_ok=True)

#     if os.listdir(models_path):
#         print(f"{unlearner} models are already generated. Skipping.")
#         return

#     if unlearner in ["original", "naive"]:
#         unlearner_name = "finetune"
#     else:
#         unlearner_name = unlearner
#     print(f"Unlearner: {unlearner_name}")
#     app = UnlearnAppForLiRA(config, unlearner_name)

#     total_models = num_splits * num_forgets
#     model_num = 0
#     with tqdm(total=total_models, desc="Unlearning Models") as pbar:
#         for split_ndx in range(num_splits):
#             for forget_ndx in range(num_forgets):
#                 model_num += 1
                
#                 # Update progress bar with current model information
#                 pbar.set_description(f"Split {split_ndx+1}/{num_splits}, Forget {forget_ndx+1}/{num_forgets}, Model {model_num}/{total_models}")
                
#                 retain, forget, val = get_retain_forget_val_indices(
#                     lira_path=output_dir / "splits",
#                     split_ndx=split_ndx,
#                     forget_ndx=forget_ndx,
#                 )
#                 if unlearner == "original":
#                     retain = np.concatenate([retain, forget])

#                 loaders = get_loaders_from_indices(
#                     app.dataset_name,
#                     app.dataset_save_dir,
#                     [retain, forget, val],
#                     app.batch_sizes,
#                     app.num_workers,
#                 )
#                 unlearning = False
#                 if unlearner not in ["original", "naive"]:
#                     unlearning = True
#                     app.model_ckpt_path = (
#                         output_dir
#                         / "original"
#                         / "models"
#                         / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
#                     )
#                 unlearned_model = app.run(loaders, unlearning=unlearning)
#                 torch.save(
#                     {"model_state_dict": unlearned_model.state_dict()},
#                     output_dir
#                     / unlearner
#                     / "models"
#                     / f"{model_name}_{split_ndx}_{forget_ndx}.pth",
#                 )
                
#                 # Update progress bar
#                 pbar.update(1)

#     # num_models = num_splits * num_forgets
#     # model_num = 0
#     # for split_ndx in range(num_splits):
#     #     for forget_ndx in range(num_forgets):
#     #         model_num += 1
#     #         print(f"Unlearning model {model_num}/{num_models}")
#     #         retain, forget, val = get_retain_forget_val_indices(
#     #             lira_path=output_dir / "splits",
#     #             split_ndx=split_ndx,
#     #             forget_ndx=forget_ndx,
#     #         )
#     #         if unlearner == "original":
#     #             retain = np.concatenate([retain, forget])

#     #         loaders = get_loaders_from_indices(
#     #                     app.dataset_name,
#     #                     app.dataset_save_dir,
#     #                     [retain, forget, val],
#     #                     app.batch_sizes,
#     #                     app.num_workers,
#     #                     )
#     #         unlearning = False
#     #         if unlearner not in ["original", "naive"]:
#     #             unlearning = True
#     #             app.model_ckpt_path = (
#     #                 output_dir
#     #                 / "original"
#     #                 / "models"
#     #                 / f"{model_name}_{split_ndx}_{forget_ndx}.pth"
#     #             )
#     #         unlearned_model = app.run(loaders, unlearning=unlearning)
#     #         torch.save(
#     #             {"model_state_dict": unlearned_model.state_dict()},
#     #             output_dir
#     #             / unlearner
#     #             / "models"
#     #             / f"{model_name}_{split_ndx}_{forget_ndx}.pth",
#     #         )
