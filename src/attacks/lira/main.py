import yaml
import argparse

class LiRAApp:
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
