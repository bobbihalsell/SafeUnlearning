import yaml
import argparse
from datasets.load_datasets import (load_dataset,
                                    _get_stratified_subset)


class DatasetInitializer:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Specify path to dataset prep .yaml file.")
        parser.add_argument("--config_path",
                            type=str,
                            required=True,
                            help="Path to .yaml file")
        args = parser.parse_args()

        # Load in the instructions for the unlearning run from a filepath
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError('.yaml file not found.')

        print(config['dataset'])
        print(config['forget'])
        dataset_cfg = config['dataset']
        self.dataset_name = dataset_cfg['name']
        self.save_dir = dataset_cfg['save_dir']
        self.proportion = dataset_cfg['proportion']
        self.val_ratio = dataset_cfg['val_ratio']
        self.dataset_cfg = dataset_cfg['cfg']
        self.url = dataset_cfg['url']

        forget_cfg = config['forget']
        self.forget_method = forget_cfg['method']
        self.forget_params = forget_cfg['parameters']

    def get_train_test_data(self):
        """ Download and save benchmark datasets with name support.

        This will download respective datasets in the save directory.
        """
        train_dataset, test_dataset = load_dataset(
            dataset_name=self.dataset_name,
            proportion=self.proportion,
            dataset_save_path=self.save_dir
        )

        return train_dataset, test_dataset

if __name__ == '__main__':
    DatasetInitializer()
    # python src/datasets/main.py --config_path src/datasets/experiments/prep_simple.yaml