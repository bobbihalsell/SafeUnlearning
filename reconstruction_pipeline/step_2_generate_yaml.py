import os 
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


if __name__ == "__main__":
    create_base_yaml()
    