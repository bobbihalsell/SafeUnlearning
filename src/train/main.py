import argparse
import yaml
import torchvision
import torch
import timm
from datasets import load_datasets as src_datasets
from datasets import preprocessing


class TrainApp:
    """ Perform pretraining, or model loading and saving, for a model."""
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Path to .yaml file with configurations for training."
            )
        parser.add_argument("--config_path",
                            type=str,
                            required=True,
                            help="Path to .yaml file for training configs.")
        args = parser.parse_args()

        # Load in the instructions for the unlearning run from a filepath
        with open(args.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except FileNotFoundError:
                print('.yaml file not found.')

        self.model_name = config['name']
        self.pretrained = config['pretrained']
        self.num_classes = config['num_classes']
        self.save_path = config['save_path']
        assert self.save_path is not None

    def initialize_model(self):
        """Initialize the model based on model name from user configuration."""
        if hasattr(torchvision.models, self.model_name):
            model = torchvision.models.get_model(
                self.model_name,
                weights="DEFAULT" if self.pretrained else None,
            )

            # Adjust the last layer based on model type
            if hasattr(model, "fc"):  # ResNet-style
                model.fc = torch.nn.Linear(
                    model.fc.in_features,
                    self.num_classes
                )
            elif hasattr(model, "classifier"):
                # MobileNet, EfficientNet, VGG, DenseNet
                if isinstance(model.classifier, torch.nn.Sequential):
                    # Handle cases where classifier is Sequential
                    last_layer_idx = len(model.classifier) - 1
                    model.classifier[last_layer_idx] = torch.nn.Linear(
                        model.classifier[last_layer_idx].in_features,
                        self.num_classes
                    )
                else:
                    model.classifier = torch.nn.Linear(
                        model.classifier.in_features,
                        self.num_classes
                    )
            else:
                raise AttributeError("Unknown classification layer "
                                     f'for {self.model_name}')

        else:
            print(f'Could not find {self.model_name} in torchvision.'
                  ' Looking in timm.')
            try:
                model = timm.create_model(
                    self.model_name,
                    pretrained=self.pretrained,
                    num_classes=self.num_classes,
                )
            except Exception:
                raise AttributeError(f"{self.model_name} not found.")

        return model

    def initialize_datasets(self):
        """ Initialize the datasets based on dataset name."""
        if self.dataset_name == 'imagenet' and self.url is not None:
            # Download and save ImageNet
            src_datasets.download_imagenet_dataset_from_web(
                url=self.url,
                dataset_save_path=self.save_path)

        train_dataset, test_dataset = src_datasets.load_dataset(
            dataset_name=self.dataset_name,
            proportion=self.proportion,
            dataset_save_path=self.save_path)
        # Perform train/val split
        train_subset, val_dataset = preprocessing.train_val_split(
            train_dataset=train_dataset,
            val_ratio=self.val_ratio
        )
        self._validate_split(train_subset, val_dataset, test_dataset)

        return train_dataset, test_dataset

    def _validate_split(self, train_set, val_set, test_set):
        """ Debugger to check no dataset leakage"""
        train_set_indices = set(train_set.indices)
        val_set_indices = set(val_set.indices)
        test_set_indices = set(test_set.indices)

        assert len(train_set_indices & val_set_indices) == 0
        assert len(val_set_indices & test_set_indices) == 0
        assert len(train_set_indices & test_set_indices) == 0

    def pretrain(self, model):
        """ Perform pretraining of a model on a dataset."""
        pass

    def run(self):
        print(self.model_name)
        print(self.pretrained)
        print(self.num_classes)
        print(self.save_path)


if __name__ == '__main__':
    trainer = TrainApp()
    trainer.run()
