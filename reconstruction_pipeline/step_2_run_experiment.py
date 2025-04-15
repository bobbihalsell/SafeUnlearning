import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dataclasses import dataclass


def create_base_yaml():
    base_yamls = {
        "src/unlearning/config/model/resnet18_exp.yaml": """
name: resnet18
num_classes: 10
model_ckpt_path: ./model/resnet18_42_original.pt
output_dir: ./artifacts
""",

        "src/unlearning/config/dataset/cifar10_exp.yaml": """
name: cifar10
save_path: ./data
cfg:
  batch_sizes:
    retain: 32
    forget: 32
    val: 32
  num_workers: 4
""",

        "src/unlearning/config/unlearner/neggrad_exp.yaml": """
name: neggrad
evaluate: true
cfg:
  lr: 0.01
  weight_decay: 0
  use_l2_penalty: false
  epochs: ??? 
""",

        "src/unlearning/config/unlearner/neggradplus_exp.yaml": """
name: neggradplus 
evaluate: true
cfg:
  lr: 0.01
  weight_decay: 0
  use_l2_penalty: false
  epochs: ?? # Ignored for scrub
  beta: ?? 
""",

        "src/unlearning/config/unlearner/scrub_exp.yaml": """
name: scrub 
evaluate: true
cfg:
  lr: 0.01
  weight_decay: 0
  use_l2_penalty: false
  min_epochs: 2  # For scrub  retain aligning steps
  max_epochs: 5  # For scrub  forget away steps
  alpha: 0.5  
  gamma: 0.5 
""",
        "src/attacks/config/data/cifar10_exp.yaml": """
model_name: resnet18
dataset_name: cifar10
labels: ???  # Set at runtime
num_classes: 10
data_root: ./data
"""
    }

    for path, content in base_yamls.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content.strip() + "\n")


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

    experiment_name = f"{cfg.method}_{cfg.samples}s_{cfg.epochs}e"
    model_dir = f"model/resnet18_{cfg.samples}s_{cfg.epochs}e/"
    original_weights = f"{model_dir}/unlearn/neggrad/resnet18_42_original.pt"
    unlearned_weights = f"{model_dir}/unlearn/neggrad/resnet18_42_unlearned.pt"

    # Unlearning
    os.system(f"python reconstruction_pipeline/utils.py --n {cfg.samples} "
              f"--label_output_path reconstruction_pipeline/labels/labels_{cfg.samples}.txt")
    
    
    os.system(f"python src/unlearning/main.py dataset=cifar10_exp model=resnet18_exp unlearner=neggrad_exp verbose={cfg.verbose} "
              f"model.output_dir={model_dir} unlearner.cfg.epochs={cfg.epochs}")
    
    #  Reconstruction
    os.system(f"python src/attacks/main_reconstructor.py data=cifar10_exp reconstructor=inversegrad verbose={cfg.verbose} seed={cfg.seed} "
              f"experiment_name={experiment_name} wandb.extra_config.unlearning_method=neggrad wandb.extra_config.epochs={cfg.epochs} "
              f"original_weights={original_weights} unlearned_weights={unlearned_weights} "
              f"data.labels=reconstruction_pipeline/labels/labels_{cfg.samples}.txt ")    




if __name__ == "__main__":
    create_base_yaml()
    run_exp()