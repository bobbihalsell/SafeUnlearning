import argparse
import yaml
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unlearning.utils import save_model, set_seed
from unlearning.neggrad import NegGrad, NegGradPlus
from unlearning.preprocessing import (
    remove_samples_by_indices,
    remove_classes
)
from datasets.load_datasets import load_cifar10_datasets
DEFAULT_SEED = 42


class UnlearnApp:
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
        model_config = config['model']
        self.model_name = model_config['name']
        self.pretrained = model_config['pretrained']
        self.num_classes = model_config['num_classes']

        self.seed = config.get('seed', DEFAULT_SEED)
        set_seed(self.seed)

        self.unlearner_name = config['unlearner']['name']
        self.unlearn_params = config['unlearner']['cfg']

        self.dataset_name = config['dataset']['name']
        self.val_ratio = config['dataset'].get('val_ratio', 0)

        self.forget_method = config['forget_method']['name']
        self.forget_params = config['forget_method']['parameters']
        self.batch_size = config['batch_size']

    def retain_forget_split(self, dataset: torch.utils.data.Dataset):
        """ Perform dataset splits based on user configuration."""
        if self.forget_method == 'index':
            retain_set, forget_set = remove_samples_by_indices(
                dataset=dataset,
                forget_set_indices=self.forget_params,
                return_forget=True
            )
        elif self.forget_method == 'label':
            retain_set, forget_set = remove_classes(
                dataset=dataset,
                forget_labels=self.forget_params,
                return_forget=True
            )
        else:
            raise ValueError(f'{self.forget_method} not supported.')

        return retain_set, forget_set

    def convert_to_dataloader(self,
                              dataset: torch.utils.data.Dataset,
                              batch_size: int,
                              shuffle: bool
                              ):
        """ Convert a Dataset into a DataLoader."""
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=4)  # TODO adaptively set num_workers based on user hardware

        return dataloader

    def initialize_model(self):
        """ Initialize the model based on model name from user configuration."""
        # TODO replace this algorithm with model_init.py from Jebastin
        original_model = timm.create_model(model_name=self.model_name,
                                           pretrained=self.pretrained,
                                           num_classes=self.num_classes)
        save_model(original_model,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='original')

        return original_model

    def initialize_unlearner(self):
        """ Initialize the correct unlearner from user specification."""
        if self.unlearner_name == 'neggrad':
            unlearner = NegGrad(
                loss_fn=nn.CrossEntropyLoss()
                )
        # TODO fix NegGradPlus to work with our new approach.
        elif self.unlearner_name == 'neggradplus':
            unlearner = NegGradPlus(
                loss_fn=nn.CrossEntropyLoss()
                )
        else:
            raise ValueError(f'unlearner_name {self.unlearner_name}'
                             ' not supported.')

        return unlearner

    def initialize_dataset(self):
        """ Initialize the entire dataset based on dataset name."""
        if self.dataset_name == 'cifar10':
            train_dataset, test_dataset = load_cifar10_datasets()
        else:
            raise ValueError(f'{self.dataset_name} not supported.')

        return train_dataset, test_dataset

    def run(self):
        """ Run the pipeline."""
        original_model = self.initialize_model()

        train_dataset, test_dataset = self.initialize_dataset()

        retain_set, forget_set = self.retain_forget_split(
            dataset=train_dataset
            )
        retain_dataloader = self.convert_to_dataloader(
            retain_set,
            batch_size=self.batch_size,
            shuffle=True
            )
        forget_dataloader = self.convert_to_dataloader(
            forget_set,
            batch_size=self.batch_size,
            shuffle=True
            )
        unlearner = self.initialize_unlearner()
        # Unlearn based on the dictionary of params
        unlearned_model = unlearner.unlearn(
            model=original_model,
            retain_dataloader=retain_dataloader,
            forget_dataloader=forget_dataloader,
            **self.unlearn_params)

        save_model(unlearned_model,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='unlearned')

        return None


if __name__ == '__main__':
    app = UnlearnApp()
    app.run()
    # Run python src/unlearning/main.py --config_path src/experiments/unlearn_test.yaml

    # TODO val_set support
    # TODO decouple dataset loading from unlearn app
    # TODO ImageNet dataset support
