import yaml
from utils import set_seed, setup_device

class ReconstructionApp:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Specify path to .yaml file with configurations to run unlearning process.")
        parser.add_argument("--config_path", type=str, required=True, help="The path to the .yaml file for unlearning configurations.")
        args = parser.parse_args()

        # Load in the instructions for the unlearning run from a filepath
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                print('.yaml file not found.')

        # Get data from the config
        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config.get('seed', DEFAULT_SEED)
        set_seed(self.seed)