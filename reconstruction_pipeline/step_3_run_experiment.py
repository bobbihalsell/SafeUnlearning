import hydra
import os
from dataclasses import dataclass


@dataclass
class Config:
    method: str
    samples: int
    epochs: int
    beta: float
    verbose: bool = False
    seed: int = 42

@hydra.main(version_base=None,
            config_path="./",
            config_name="exp")
def run_exp(cfg: Config):

    print('============ Run Experiment ============')
    print(cfg)


    if cfg.verbose == False:
        os.environ["WANDB_SILENT"] = "true"

    os.environ["WANDB_START_METHOD"] = "thread"
    experiment_name = f"{cfg.method}_{cfg.samples}s_{cfg.epochs}e"
    model_dir = f"model/resnet18_{cfg.samples}s_{cfg.epochs}e/"
    original_weights = f"{model_dir}/unlearn/neggrad/resnet18_42_original.pt"
    unlearned_weights = f"{model_dir}/unlearn/neggrad/resnet18_42_unlearned.pt"

    # Unlearning
    os.system(f"python reconstruction_pipeline/utils.py --n {cfg.samples} "
              f"--label_output_path reconstruction_pipeline/labels/labels_{cfg.samples}.txt")
    
    # NOTE: we keep unlearning verbose true to track unlearning losses
    os.system(f"python src/unlearning/main.py dataset=cifar10_exp model=resnet18_exp unlearner=neggrad_exp  "
              f"model.output_dir={model_dir} unlearner.cfg.epochs={cfg.epochs}")
    
    #  Reconstruction
    os.system(f"python src/attacks/main_reconstructor.py data=cifar10_exp reconstructor=inversegrad verbose={cfg.verbose} seed={cfg.seed} "
              f"experiment_name={experiment_name} wandb.extra_config.unlearning_method=neggrad wandb.extra_config.epochs={cfg.epochs} "
              f"+wandb.extra_config.seed={cfg.seed} "
              f"original_weights={original_weights} unlearned_weights={unlearned_weights} "
              f"data.labels=reconstruction_pipeline/labels/labels_{cfg.samples}.txt ")    



if __name__ == "__main__":
    run_exp()