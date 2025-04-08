import os
import yaml
import argparse
from pathlib import Path

from src.unlearning.utils import save_model, set_seed, setup_device
from src.unlearning.main import UnlearnApp


class LiRAApp(UnlearnApp):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Specify path to lira .yaml file.")
        parser.add_argument("--config_path",
                            type=str,
                            required=True,
                            help="Path to .yaml file")
        args = parser.parse_args()

        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                print('.yaml file not found.')

        # Perform input validation first

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        # Output directory
        self.output_dir = config.get('output_dir', 'artifacts/lira/')
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        pass
