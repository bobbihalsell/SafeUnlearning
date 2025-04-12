import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.attacks.lira.compute_lira import run as lira_score
from src.attacks.lira.config_validation import LiRAValidator
from src.attacks.lira.generate_predictions import run as generate_predictions
from src.attacks.lira.generate_splits import run as generate_splits
from src.attacks.lira.train_lira import main as tl_main
from src.unlearning.utils import set_seed, setup_device


class LiRAApp:
    def __init__(self, config: DictConfig):
        self.config = OmegaConf.to_container(config, resolve=True)

        # Perform input validation
        LiRAValidator(self.config)

        self.device = setup_device()
        print(f"Using device: {self.device}")
        self.seed = self.config["seed"]
        set_seed(self.seed)

        self.root = Path("artifacts/attacks/lira" / self.config.exp_name)
        os.makedirs(self.root, exist_ok=True)

    def train_base_models(self):
        # call finetune unlearner
        tl_main()

    def train_test_models(self):
        # call finetune unlearner
        tl_main()

    def unlearn_models(self):
        # call config.unlearner unlearner
        tl_main()

    def get_predictions(self):
        generate_predictions(self.config)

    def get_scores(self):
        lira_score(self.config)

    def run(self):
        generate_splits(self.root, self.config)
        self.train_base_models()
        self.train_test_models()
        self.unlearn_models()
