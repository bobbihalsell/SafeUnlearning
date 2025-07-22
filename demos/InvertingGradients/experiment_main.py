import os
import subprocess
import sys
from dataclasses import dataclass

import hydra

LR = 0.01   
MOMENTUM = 0  # for unlearning
SEED = 42
MODEL_NAME = "resnet18"
MODEL = "torchvision"
DATASET = "imagenet"

DATASET_SAVE_PATH = "demos/GGL/imagenet_example_split"
ORIGINAL_MODEL_SAVE_PATH = "demos/InvertingGradients/"


@dataclass
class Config:
    method: str
    epochs: int
    beta: float
    verbose: bool = False
    seed: int = 42
    init: str = "randn"
    min_epochs: int = 0
    gamma: float = 0.0001
    alpha: float = 0.0001
    retain_strength: float = 0.01
    attack: str = "invertgrad"  # "invertgrad" or "ggl"
    unlearned_label: int = 88  # The label to be unlearned, default is 88 for ImageNet


# The @hydra.main decorator initializes the Hydra configuration system.
@hydra.main(version_base=None, config_path="./", config_name="exp")
def run_exp(cfg: Config):
    print("============ Run Experiment ============")
    print(cfg)

    # Set environment variables
    if not cfg.verbose:
        os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_START_METHOD"] = "thread"

    retain_size = get_retain_size(f"{DATASET_SAVE_PATH}/retain")
    forget_labels = get_forget_labels(f"{DATASET_SAVE_PATH}/forget")
    forget_size = 1

    # Set unique experiment name and model directory based on the method
    exp_params = get_exp_params(cfg, forget_size, retain_size)
    experiment_name = f"{cfg.method}_{exp_params}"
    model_dir = f"model/{experiment_name}"
    model_save_path = "./demos/InvertingGradients/resnet18_42_original.pt"
    original_weights = f"{model_dir}/unlearn/{cfg.method}/{MODEL_NAME}_{SEED}_original.pt"
    unlearned_weights = f"{model_dir}/unlearn/{cfg.method}/{MODEL_NAME}_{SEED}_unlearned.pt"

    base_args = [
        "python",
        "src/unlearning/main.py",
        f"dataset={DATASET}",
        f"model={MODEL}",
        f"seed={SEED}",
        f"experiment_name=InvertGradExp",
        f"verbose={cfg.verbose}",
        f"output_dir={model_dir}",
        f"dataset.save_path={DATASET_SAVE_PATH}",
        f"model.model_name={MODEL_NAME}",
        # f"model.original_model_ckpt_path={model_save_path}",
        "unlearner.cfg.lr_decay_factor=null",
    ]

    method_specific_args = {
        "neggrad": [
            "unlearner=neggrad",
            f"unlearner.cfg.epochs={cfg.epochs}",
            f"unlearner.cfg.lr={LR}",
        ],
        "neggradplus": [
            "unlearner=neggradplus",
            f"unlearner.cfg.epochs={cfg.epochs}",
            f"unlearner.cfg.beta={cfg.beta}",
            f"unlearner.cfg.lr={LR}",
        ],
        "scrub": [
            "unlearner=scrub",
            f"unlearner.cfg.max_epochs={cfg.epochs}",
            f"unlearner.cfg.min_epochs={cfg.min_epochs}",
            f"unlearner.cfg.lr={LR}",
            f"unlearner.cfg.weight_decay=0",
            f"unlearner.cfg.momentum={MOMENTUM}",
            f"unlearner.cfg.alpha={cfg.alpha}",
            f"unlearner.cfg.gamma={cfg.gamma}",
        ],
        "sgru": [
            "unlearner=sgru",
            f"unlearner.cfg.retain_strength={cfg.retain_strength}",
            f"unlearner.cfg.epochs={cfg.epochs}",
            f"unlearner.cfg.lr=0.001",
            f"unlearner.cfg.weight_decay=0",
            f"unlearner.cfg.momentum={MOMENTUM}",
        ],
    }
    try:
        subprocess.run(base_args + method_specific_args[cfg.method], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during main.py execution: {e}")
        sys.exit(1)

    # Step 3: run reconstruction
    if forget_size == 1:  # if only one sample is forgotten, we can use pixelmean
        num_runs = 1
        scoring_choice = "loss"
        iterations = 2000
    else:
        num_runs = 1
        scoring_choice = "loss"
        iterations = 7500

    reconstruction_experiment_name = f"{experiment_name}_{cfg.seed}"
    try:
        if cfg.attack == "invertgrad":
            subprocess.run(
                [
                    "python",
                    "src/attacks/main_reconstructor.py",
                    f"dataset={DATASET}",
                    "model=torchvision",
                    "attack=invertgrad",
                    "model.model_name=resnet18",
                    f"dataset.save_path={DATASET_SAVE_PATH}",
                    f"verbose={cfg.verbose}",
                    f"seed={cfg.seed}",
                    f"model.original_model_ckpt_path={original_weights}",
                    f"model.unlearned_model_ckpt_path={unlearned_weights}",
                    f"attack.unlearned_labels=[88]",
                    f"attack.cfg.num_runs={num_runs}",
                    f"attack.cfg.scoring_choice={scoring_choice}",
                    f"attack.cfg.recon_iterations={iterations}",
                    f"attack.cfg.init={cfg.init}",
                    f"experiment_name={reconstruction_experiment_name}",
                    "+wandb=default",
                    f"wandb.project_name=InvertGradExpImagenet",
                ],
                check=True,
            )
        else:
                        subprocess.run(
                [
                    "python",
                    "src/attacks/main_reconstructor.py",
                    f"dataset={DATASET}",
                    "model=torchvision",
                    "attack=ggl",
                    "model.model_name=resnet18",
                    f"dataset.save_path={DATASET_SAVE_PATH}",
                    f"verbose={cfg.verbose}",
                    f"seed={cfg.seed}",
                    f"model.original_model_ckpt_path={original_weights}",
                    f"model.unlearned_model_ckpt_path={unlearned_weights}",
                    f"attack.unlearned_labels={cfg.unlearned_label}",
                    f"attack.lr=0.001",
                    f"attack.cfg.budget=1000",
                    f"attack.cfg.initial_lr=1",
                    f"attack.cfg.batch_size=3",
                    f"experiment_name={reconstruction_experiment_name}",
                    "+wandb=default",
                    f"wandb.project_name=InvertGradExpImagenet",
                ],
                check=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Error during main_reconstructor.py execution: {e}")
        sys.exit(1)


def get_exp_params(cfg, forget_size, retain_size):
    if cfg.method == "neggrad":
        return f"{forget_size}s_{cfg.epochs}e"
    elif cfg.method == "neggradplus":
        return f"{forget_size}s_{cfg.epochs}e_{cfg.beta}b_{retain_size}r"
    elif cfg.method == "scrub":
        return f"{forget_size}s_{cfg.epochs}e_{cfg.min_epochs}m_{retain_size}r_{cfg.alpha}a_{cfg.gamma}g"
    elif cfg.method == "sgru":
        return f"{forget_size}s_{cfg.epochs}e_{retain_size}r_{cfg.retain_strength}rs"
    else:
        raise ValueError(f"Unknown method: {cfg.method}")


if __name__ == "__main__":
    from exp_utils import get_forget_labels, get_retain_size
    run_exp()
