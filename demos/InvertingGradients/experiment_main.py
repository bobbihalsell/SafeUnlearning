from dataclasses import dataclass
import hydra
import subprocess
import os
import sys
from utils import get_forget_labels, get_retain_size

LR = 0.01
MOMENTUM = 0
SEED = 42
MODEL_NAME = "resnet18"
MODEL = "torchvision"
DATASET = "cifar10"
DATASET_SAVE_PATH = "./data"
NUM_CLASSES = 10
TRAINING_EPOCHS = 15
TRAINING_LR = 0.01
ORIGINAL_MODEL_SAVE_PATH = "./model"


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


@hydra.main(version_base=None,
            config_path="./",
            config_name="exp")
def run_exp(cfg: Config):

    print('============ Run Experiment ============')
    print(cfg)

    # Set environment variables
    if not cfg.verbose:
        os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_START_METHOD"] = "thread"

    retain_size = get_retain_size(f"{DATASET_SAVE_PATH}/retain")
    forget_labels = get_forget_labels(f"{DATASET_SAVE_PATH}/forget")
    forget_size = len(forget_labels)

    # Set unique experiment name and model directory based on the method
    if cfg.method == "neggrad" or cfg.method == "sgru":
        exp_params = f"{forget_size}s_{cfg.epochs}e"

    elif cfg.method == "neggradplus":
        exp_params = f"{forget_size}s_{cfg.epochs}e_{cfg.beta}b_{retain_size}r"

    elif cfg.method == "scrub":
        exp_params = f"{forget_size}s_{cfg.epochs}e_{cfg.min_epochs}m_{retain_size}r_{cfg.alpha}a_{cfg.gamma}g"

    experiment_name = f"{cfg.method}_{exp_params}"
    model_dir = f"model/{experiment_name}"

    # Trained model is saved here:
    # model_save_path = f"{ORIGINAL_MODEL_SAVE_PATH}/{MODEL_NAME}_{SEED}_original.pt"
    model_save_path = "vol/bitbucket/vb524/v3_safee/safe-unlearning/model/resnet18_42_original.pt"
    # After unlearning, the model will be saved in the following paths
    original_weights = f"{model_dir}/unlearn/{cfg.method}/{MODEL_NAME}_{SEED}_original.pt"
    unlearned_weights = f"{model_dir}/unlearn/{cfg.method}/{MODEL_NAME}_{SEED}_unlearned.pt"

    # Step 1: run the unlearning
    try:
        if cfg.method == "neggrad":
            subprocess.run(
                ["python", "src/unlearning/main.py",
                 f"dataset={DATASET}",
                 f"model={MODEL}",
                 f"seed={SEED}",
                 "unlearner=neggrad",
                 "experiment_name=InvertGradExp",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.original_model_ckpt_path={model_save_path}",
                 f"unlearner.cfg.epochs={cfg.epochs}",
                 f"unlearner.cfg.lr={LR}",
                 "unlearner.cfg.lr_decay_factor=null"],
                check=True
            )
        elif cfg.method == "neggradplus":
            subprocess.run(
                ["python", "src/unlearning/main.py",
                 f"dataset={DATASET}",
                 f"model={MODEL}",
                 f"seed={SEED}",
                 "unlearner=neggradplus",
                 "experiment_name=InvertGradExp",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.original_model_ckpt_path={model_save_path}",
                 f"unlearner.cfg.epochs={cfg.epochs}",
                 f"unlearner.cfg.beta={cfg.beta}",
                 f"unlearner.cfg.lr={LR}",
                 "unlearner.cfg.lr_decay_factor=null",
                 ],
                check=True
            )
        elif cfg.method == "scrub":
            subprocess.run(
                ["python", "src/unlearning/main.py",
                 f"dataset={DATASET}",
                 f"model={MODEL}",
                 f"seed={SEED}",
                 "unlearner=scrub",
                 "experiment_name=InvertGradExp",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.original_model_ckpt_path={model_save_path}",
                 f"unlearner.cfg.max_epochs={cfg.epochs}",
                 f"unlearner.cfg.min_epochs={cfg.min_epochs}",
                 f"unlearner.cfg.lr={LR}",
                 "unlearner.cfg.weight_decay=0",
                 f"unlearner.cfg.momentum={MOMENTUM}",
                 "unlearner.cfg.lr_decay_factor=null",
                 f"unlearner.cfg.alpha={cfg.alpha}",
                 f"unlearner.cfg.gamma={cfg.gamma}",
                 ],
                check=True
            )
        elif cfg.method == "sgru":
            subprocess.run(
                ["python", "src/unlearning/main.py",
                 f"dataset={DATASET}",
                 f"model={MODEL}",
                 f"seed={SEED}",
                 "unlearner=sgru",
                 "experiment_name=InvertGradExp",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.original_model_ckpt_path=/vol/bitbucket/vb524/v3_safee/safe-unlearning/model/resnet18_42_original.pt",
                 f"unlearner.cfg.epochs={cfg.epochs}",
                 f"unlearner.cfg.lr={LR}",
                 "unlearner.cfg.weight_decay=0",
                 f"unlearner.cfg.momentum={MOMENTUM}",
                 "unlearner.cfg.lr_decay_factor=null"
                 ],
                check=True
            )
    except subprocess.CalledProcessError as e:
        print(f"Error during main.py execution: {e}")
        sys.exit(1)

    # Step 3: run reconstruction
    if forget_size == 1:
        num_runs = 1
        scoring_choice = "pixelmean"
        iterations = 5000
    else:
        num_runs = 1
        scoring_choice = "loss"
        iterations = 7_500
    try:
        subprocess.run(
            ["python", "src/attacks/main_reconstructor.py",
             "dataset=cifar10",
             "model=torchvision",
             "attack=invertgrad",
             "model.model_name=resnet18",
             f"dataset.save_path={DATASET_SAVE_PATH}",
             f"verbose={cfg.verbose}",
             f"seed={cfg.seed}",
             f"model.original_model_ckpt_path={original_weights}",
             f"model.unlearned_model_ckpt_path={unlearned_weights}",
             f"attack.unlearned_labels={forget_labels}",
             f"attack.cfg.num_runs={num_runs}",
             f"attack.cfg.scoring_choice={scoring_choice}",
             f"attack.cfg.recon_iterations={iterations}",
             f"attack.cfg.init={cfg.init}",
             f"experiment_name={experiment_name}"
            #  "+wandb_cfg=default",
            #  f"wandb_cfg.extra_config.unlearning_method={cfg.method}",
            #  f"wandb_cfg.extra_config.epochs={cfg.epochs}",
            #  f"+wandb_cfg.extra_config.seed={cfg.seed}",
            #  f"+wandb_cfg.extra_config.beta={cfg.beta}",
            #  f"+wandb_cfg.extra_config.retain={retain_size}",
            #  f"+wandb_cfg.extra_config.min_epochs={cfg.min_epochs}",
            #  f"+wandb_cfg.extra_config.gamma={cfg.gamma}",
            #  f"+wandb_cfg.extra_config.alpha={cfg.alpha}"
             ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during main_reconstructor.py execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_exp()
