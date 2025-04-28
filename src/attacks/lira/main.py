import os
from copy import deepcopy
from pathlib import Path

import hydra
from compute_lira import lira_score
from config_validation import LiRAValidator
from generate_predictions import generate_predictions
from generate_splits import generate_splits
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from train_lira import train_models

from utils import set_seed, setup_device


class LiRAApp(LiRAValidator):
    """Main application class for running LiRA (Likelihood Ratio Attack).

    This class handles the complete pipeline for LiRA including data splitting,
    shadow models training, unlearning, and score computation.
    This attack was originally introduced in:
    https://ieeexplore.ieee.org/document/9833649
    U-LiRA adapts this approach for the machine unlearning setting.The unlearning variant is based on:
    https://arxiv.org/abs/2302.09880

    Args:
        config (DictConfig): Configuration object containing all experiment parameters.

    Attributes:
        config (DictConfig): Stored configuration object.
        device (str): Device to run computations on ('cuda' or 'cpu').
        seed (int): Random seed for reproducibility.
        output_dir (Path): Directory for storing experiment outputs.
    """

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
        """Trains the original shadow models and stores their predictions."""
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
        """Performs shadow model unlearning and stores their predictions."""
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
        """Executes the complete LiRA experiment pipeline.

        This method orchestrates the entire experiment process including:
        1. Generating data splits
        2. Training original models
        3. Unlearning models
        4. Computing LiRA scores as described in https://arxiv.org/abs/2410.01276

        """

        print("Generating splits...")
        generate_splits(
            self.dataset_name,
            self.forget_ratio,
            self.val_ratio,
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
            self.seed,
        )
        print("Training original models...")
        self.train_original_models()
        print("Unlearning models...")
        self.unlearn_models()
        print("Generating LiRA scores...")
        lira_score(
            self.dataset_name,
            self.model_name,
            self.unlearner_name,
            self.num_splits,
            self.num_forgets,
            self.dataset_save_dir,
            self.output_dir,
        )


@hydra.main(version_base=None, config_path="../../../configs", config_name="attacks")
def main(cfg: DictConfig):
    # Print the config for the user first
    print("============ Run Configuration ============")
    print(OmegaConf.to_yaml(cfg))
    print("============================================")
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            "Missing the following required arguments in the configuration: "
            f"{missing_keys}. \n"
            "Hint: python file.py key=value sets the appropriate value."
        )
    app = LiRAApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
