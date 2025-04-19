from dataclasses import dataclass
import hydra
import subprocess
import os
import sys
from settings import LR, MOMENTUM, DATASET_SAVE_PATH, MODEL_NAME, MODEL, DATASET, MODEL_SAVE_PATH


@dataclass
class Config:
    method: str
    samples: int
    retain: int
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

    # Set unique experiment name and model directory based on the method
    if cfg.method == "neggrad":
        exp_params = f"{cfg.samples}s_{cfg.epochs}e"
        experiment_name = f"{cfg.method}_{exp_params}"
        model_dir = f"model/{experiment_name}"

    elif cfg.method == "neggradplus":
        exp_params = f"{cfg.samples}s_{cfg.epochs}e_{cfg.beta}b_{cfg.retain}r"
        experiment_name = f"{cfg.method}_{exp_params}"
        model_dir = f"model/{experiment_name}"

    elif cfg.method == "scrub":
        exp_params = f"{cfg.samples}s_{cfg.epochs}e_{cfg.min_epochs}m_{cfg.retain}r_{cfg.alpha}a_{cfg.gamma}g"
        experiment_name = f"{cfg.method}_{exp_params}"
        model_dir = f"model/{experiment_name}"

    # Trained model is saved here:
    model_save_path = f"{MODEL_SAVE_PATH}/{MODEL_NAME}_42_original.pt"

    # After unlearning, the model will be saved in the following paths
    original_weights = f"{model_dir}/unlearn/{cfg.method}/resnet18_42_original.pt"
    unlearned_weights = f"{model_dir}/unlearn/{cfg.method}/resnet18_42_unlearned.pt"

    # Step 1: move forget and retain samples from pool to data folder
    try:
        subprocess.run(
            ["python", "invert_grad_pipeline/utils.py",
             "--n_forget", str(cfg.samples),
             "--n_retain", str(cfg.retain),
             ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during utils.py execution: {e}")
        sys.exit(1)

    # Step 2: run the unlearning
    try:
        if cfg.method == "neggrad":
            subprocess.run(
                ["python", "src/unlearning/main.py",
                 f"dataset={DATASET}",
                 f"model={MODEL}",
                 "unlearner=neggrad",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.model_ckpt_path={model_save_path}",
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
                 "unlearner=neggradplus",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.model_ckpt_path={model_save_path}",
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
                 "unlearner=scrub",
                 f"verbose={cfg.verbose}",
                 f"output_dir={model_dir}",
                 f"dataset.save_path={DATASET_SAVE_PATH}",
                 f"model.model_name={MODEL_NAME}",
                 f"model.model_ckpt_path={model_save_path}",
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
    except subprocess.CalledProcessError as e:
        print(f"Error during main.py execution: {e}")
        sys.exit(1)

    # Step 3: run reconstruction
    if cfg.samples == 1:
        num_runs = 2
        scoring_choice = "pixelmean"
        iterations = 5000
    else:
        num_runs = 1
        scoring_choice = "loss"
        iterations = 7_500
    try:
        subprocess.run(
            ["python", "src/attacks/main_reconstructor.py",
             "data=cifar10",
             "reconstructor=invertgrad",
             f"verbose={cfg.verbose}",
             f"seed={cfg.seed}",
             f"original_weights={original_weights}",
             f"unlearned_weights={unlearned_weights}",
             "data.data_root=./data/forget",
             "data.model_name=resnet18",
             f"data.labels=invert_grad_pipeline/labels/labels_{cfg.samples}.txt",
             f"reconstructor.cfg.num_runs={num_runs}",
             f"reconstructor.cfg.scoring_choice={scoring_choice}",
             f"reconstructor.cfg.recon_iterations={iterations}",
             f"reconstructor.cfg.init={cfg.init}",
             f"experiment_name={experiment_name}",
             "+wandb_cfg=default",
             f"wandb_cfg.extra_config.unlearning_method={cfg.method}",
             f"wandb_cfg.extra_config.epochs={cfg.epochs}",
             f"+wandb_cfg.extra_config.seed={cfg.seed}",
             f"+wandb_cfg.extra_config.beta={cfg.beta}",
             f"+wandb_cfg.extra_config.retain={cfg.retain}",
             f"+wandb_cfg.extra_config.min_epochs={cfg.min_epochs}",
             f"+wandb_cfg.extra_config.gamma={cfg.gamma}",
             f"+wandb_cfg.extra_config.alpha={cfg.alpha}"
             ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during main_reconstructor.py execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_exp()
