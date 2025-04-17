import hydra
import os
from dataclasses import dataclass
import torch


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

@hydra.main(version_base=None,
            config_path="./",
            config_name="exp")
def run_exp(cfg: Config):

    print('============ Run Experiment ============')
    print(cfg)


    if not cfg.verbose:
        os.environ["WANDB_SILENT"] = "true"

    os.environ["WANDB_START_METHOD"] = "thread"


    if cfg.method == "neggrad":
        experiment_name = f"{cfg.method}_{cfg.samples}s_{cfg.epochs}e"
        model_dir = f"model/resnet18_{cfg.samples}s_{cfg.epochs}e/"
        original_weights = f"{model_dir}/unlearn/neggrad/resnet18_42_original.pt"
        unlearned_weights = f"{model_dir}/unlearn/neggrad/resnet18_42_unlearned.pt"

    elif cfg.method == "neggradplus":
        experiment_name = f"{cfg.method}_{cfg.samples}s_{cfg.epochs}e_{cfg.beta}b_{cfg.retain}r"
        model_dir = f"model/resnet18_{cfg.samples}s_{cfg.epochs}e_{cfg.beta}b_{cfg.retain}r/"
        original_weights = f"{model_dir}/unlearn/neggradplus/resnet18_42_original.pt"
        unlearned_weights = f"{model_dir}/unlearn/neggradplus/resnet18_42_unlearned.pt"


    # Unlearning
    os.system(f"python reconstruction_pipeline/utils.py --n_forget={cfg.samples} --n_retain={cfg.retain} "
              f"--label_output_path reconstruction_pipeline/labels/labels_{cfg.samples}.txt")
    
    if cfg.method == "neggrad":
        os.system(f"python src/unlearning/main.py dataset=cifar10_exp model=resnet18_exp unlearner=neggrad_exp verbose={cfg.verbose} "
                f"model.output_dir={model_dir} unlearner.cfg.epochs={cfg.epochs}")
    
    elif cfg.method == "neggradplus":
        os.system(f"python src/unlearning/main.py dataset=cifar10_exp model=resnet18_exp unlearner=neggradplus_exp verbose={cfg.verbose} "
        f"model.output_dir={model_dir} unlearner.cfg.epochs={cfg.epochs} unlearner.cfg.beta={cfg.beta}")
    
    checkpoint = torch.load(unlearned_weights, map_location='cpu')
    
    losses = checkpoint['forget_losses']
    print(f"Unlearning losses: {losses}")

    losses = losses[-1]

    if cfg.samples == 1:
        num_runs = 3
        scoring_choice = "pixelmean"
        iterations = 5000
    else:
        num_runs = 1
        scoring_choice = "loss"
        iterations = 7_500
    #  Reconstruction
    os.system(f"python src/attacks/main_reconstructor.py data=cifar10_exp reconstructor=inversegrad verbose={cfg.verbose} seed={cfg.seed} "
              f"reconstructor.cfg.num_runs={num_runs} reconstructor.cfg.scoring_choice={scoring_choice} reconstructor.cfg.recon_iterations={iterations} "
              f"reconstructor.cfg.init={cfg.init} "

              f"experiment_name={experiment_name} wandb.extra_config.unlearning_method=neggrad wandb.extra_config.epochs={cfg.epochs} "
              f"+wandb.extra_config.seed={cfg.seed} "
              f"+wandb.extra_config.unlearning_losses={losses} "
              f"wandb.extra_config.beta={cfg.beta} +wandb.extra_config.retain={cfg.retain} "
              f"original_weights={original_weights} unlearned_weights={unlearned_weights} "
              f"data.labels=reconstruction_pipeline/labels/labels_{cfg.samples}.txt ")    



if __name__ == "__main__":
    run_exp()