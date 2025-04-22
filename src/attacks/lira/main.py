import os
from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from compute_lira import lira_score
from config_validation import LiRAValidator
from generate_predictions import  generate_predictions
from generate_splits import generate_splits
from train_lira import train_models
from utils import set_seed, setup_device


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
            
    def train_original_models(self):
        original_config = deepcopy(self.config)
        original_config["unlearner"] = original_config["trainer"]

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
            self.load_method,
            self.model_name,
            self.num_classes,
            self.init_path,
            self.model_kwargs,
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
            self.load_method,
            self.model_name,
            self.num_classes,
            self.init_path,
            self.model_kwargs,
            self.unlearner_name,
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
            self.device,
        )

    def run(self):
        print('Generating splits...')
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
        print('Training original models...')
        self.train_original_models()
        print('Unlearning models...')
        self.unlearn_models()
        print('Generating predictions and LIRA scores...')
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
