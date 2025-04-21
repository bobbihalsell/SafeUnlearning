import os
from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from attacks.lira.compute_lira import run as lira_score
from attacks.lira.config_validation import LiRAValidator
from attacks.lira.generate_predictions import run as generate_predictions
from attacks.lira.generate_splits import run as generate_splits
from attacks.lira.train_lira import run as train_models
from unlearning.utils import set_seed, setup_device


class LiRAApp(LiRAValidator):
    def __init__(self, config: DictConfig):
        # Perform input validation first
        self.config = config
        super().__init__(OmegaConf.to_container(config, resolve=True))

        self.device = setup_device()
        print(f"Using device: {self.device}")
        self.seed = self.config["seed"]
        set_seed(self.seed)

        self.output_dir = Path(self.config["output_dir"])
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_model(self):
        pass

    def generate_unlearning_configs(self, type):
        if type == 'original':
            config = deepcopy(self.config)
            config["unlearner"] = config["trainer"]
        if type == 'naive':
            config = deepcopy(self.config)
            config["unlearner"] = config["trainer"]
            config["dataset"]["forget_ratio"] = 0.0

    def train_original_models(self):
        original_config = deepcopy(self.config)
        original_config["unlearner"] = original_config["original"]

        train_models(
            original_config,
            self.model_name,
            "original",
            self.num_splits,
            self.num_forgets,
            self.output_dir,
        )
        generate_predictions(
            self.dataset_name,
            self.dataset_cfg,
            self.model_name,
            self.num_classes,
            "original",
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
            self.device,
        )

    def unlearn_models(self):
        train_models(
            self.config,
            self.model_name,
            self.unlearner_name,
            self.num_splits,
            self.num_forgets,
            self.output_dir,
        )
        generate_predictions(
            self.dataset_name,
            self.dataset_cfg,
            self.model_name,
            self.num_classes,
            self.unlearner_name,
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
            self.device,
        )

    def run(self):
        generate_splits(
            self.dataset_name,
            self.forget_ratio,
            self.val_ratio,
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
            self.seed
        )
        self.train_original_models()
        self.unlearn_models()
        lira_score(
            self.dataset_name,
            self.model_name,
            self.unlearner_name,
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
        )


@hydra.main(version_base=None,
            config_path="config",
            config_name="config")
def main(cfg: DictConfig):
    # Print the config for the user first
    print('============ Run Configuration ============')
    print(OmegaConf.to_yaml(cfg))
    print('============================================')
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            'Missing the following required arguments in the configuration: '
            f'{missing_keys}. \n'
            'Hint: python file.py key=value sets the appropriate value.')
    app = LiRAApp(cfg)
    app.run()


if __name__ == '__main__':
    main()
