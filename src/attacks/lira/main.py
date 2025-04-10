import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig

from src.unlearning.utils import save_model, set_seed, setup_device
from src.unlearning.main import UnlearnApp

from src.attacks.lira.config_validation import LiRAValidator
from src.attacks.lira.generate_splits import run as generate_splits
from src.attacks.lira.train_lira import main as tl_main
from src.attacks.lira.generate_predictions import run as generate_predictions
from src.attacks.lira.compute_lira import run as lira_score


class LiRAApp:
    def __init__(self, config: DictConfig):
        self.config = OmegaConf.to_container(config, resolve=True)

        # Perform input validation
        LiRAValidator(self.config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = self.config['seed']
        set_seed(self.seed)

        # Output directory
        self.output_dir = self.config.get('root', 'artifacts/attacks/lira/')
        os.makedirs(self.output_dir, exist_ok=True)

    def configure_data(self):
        generate_splits(self.config)

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
        self.configure_data()
        self.train_base_models()
        self.train_test_models()
        self.unlearn_models()
