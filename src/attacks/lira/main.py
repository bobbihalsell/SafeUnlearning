import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from attacks.lira.compute_lira import run as lira_score
from attacks.lira.config_validation import LiRAValidator
from attacks.lira.generate_predictions import run as generate_predictions
from attacks.lira.generate_splits import run as generate_splits
from attacks.lira.train_lira import main as tl_main
from unlearning.utils import set_seed, setup_device


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
        tl_main(self.config, "finetune", self.root, "original")
        generate_predictions(self.config, self.root, "original")

    def train_test_models(self):
        tl_main(self.config, "finetune", self.root, "naive")
        generate_predictions(self.config, self.root, "naive")

    def unlearn_models(self):
        tl_main(self.config, self.config.unlearner.name, self.root, self.config.unlearner.name)
        generate_predictions(self.config, self.root, self.config.unlearner.name)

    def get_scores(self):
        lira_score(self.config)

    def run(self):
        generate_splits(self.root, self.config)
        self.train_base_models()
        self.train_test_models()
        self.unlearn_models()
        self.get_scores()
