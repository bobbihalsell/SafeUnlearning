import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train.image_loading import RobustImageFolder
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import MissingMandatoryValue
from datasets.cifar10 import get_cifar10_test_transform
from datasets.cifar100 import get_cifar100_test_transform
from datasets.imagenet import get_imagenet_test_transform
from unlearning.config_validation import InputValidator
from unlearning.finetune import FinetuneUnlearner
from unlearning.scrub import SCRUB
from unlearning.kunlearn import KUnlearn
from unlearning.neggrad import NegGrad, NegGradPlus
from unlearning.utils import save_model, set_seed, setup_device, ConfigError
from unlearning.importmodel import ImportModel
import time


class UnlearnApp(InputValidator):
    def __init__(self, config: DictConfig):
        # Perform input validation first
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        self.unlearn_params['loss_fn'] = nn.CrossEntropyLoss()
        # Output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        """Initialize the model based on model name from user configuration."""
        importer = ImportModel(
            load_method=self.load_method,
            model_name=self.model_name,
            num_classes=self.num_classes,
            init_path=self.init_path,
            model_ckpt_path=self.model_ckpt_path,
            model_kwargs=self.model_kwargs,
            )
        model = importer.load_model()

        return model

    def initialize_unlearner(self):
        """ Initialize the correct unlearner from user specification."""
        if self.unlearner_name == 'finetune':
            unlearner = FinetuneUnlearner(
                self.device,
                self.evaluate
            )
        elif self.unlearner_name == 'neggrad':
            unlearner = NegGrad(
                self.device,
                self.evaluate
            )
        elif self.unlearner_name == 'neggradplus':
            unlearner = NegGradPlus(
                self.device,
                self.evaluate
            )
        elif self.unlearner_name == 'scrub':
            unlearner = SCRUB(
                self.device,
                self.evaluate
            )
        elif self.unlearner_name == 'euk':
            unlearner = KUnlearn(
                k=self.unlearn_params['k'],
                method=self.unlearner_name,
                device=self.device,
                evaluate=self.evaluate
                )
        elif self.unlearner_name == 'cfk':
            unlearner = KUnlearn(
                k=self.unlearn_params['k'],
                method=self.unlearner_name,
                device=self.device,
                evaluate=self.evaluate
                )
        else:
            raise ValueError(f'unlearner_name {self.unlearner_name}'
                             ' not supported.')
        self.unlearner = unlearner
        return unlearner

    def get_transform(self):
        if self.dataset_name == 'cifar10':
            return get_cifar10_test_transform()
        elif self.dataset_name == 'cifar100':
            return get_cifar100_test_transform()
        elif self.dataset_name == 'imagenet':
            return get_imagenet_test_transform()
        else:
            raise ConfigError(f'dataset_name {self.dataset_name} '
                              ' not supported.')

    def initialize_dataloaders(self):
        """ Initialize dataloaders from the ImageNet dataset folder."""
        transform = self.get_transform()

        # Load datasets for each split, exclude train and test data
        if not os.path.exists(self.dataset_save_dir):
            raise ValueError('Data directory not found.')
        splits = [d for d in os.listdir(self.dataset_save_dir) if
                  os.path.isdir(os.path.join(self.dataset_save_dir, d)) and
                  d not in ['train', 'test'] and
                  not d.startswith('.')
                  ]
        if 'forget' not in splits:
            raise ValueError('Forget data required in dataset directory.')

        dataloaders = {}

        for split in splits:
            if split not in self.batch_sizes:
                raise ConfigError(
                    f"Missing batch size configuration for split '{split}' "
                    "Configure this under dataset.cfg.batch_sizes.split_name."
                )
            batch_size = self.batch_sizes[split]
            split_dir = os.path.join(self.dataset_save_dir, split)

            dataset = RobustImageFolder(root=split_dir,
                                        transform=transform)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split in ['retain', 'forget']),
                num_workers=self.num_workers,
                pin_memory=True
            )

        sample_input, _ = dataset[0]
        input_shape = sample_input.shape
        # Add an empty retain dataloader if not present in the dataset
        if 'retain' not in splits:
            empty_x = torch.empty((0, *input_shape))
            empty_labels = torch.empty((0,), dtype=torch.long)
            empty_dataset = TensorDataset(empty_x, empty_labels)

            dataloaders['retain'] = DataLoader(
                empty_dataset
            )

        return dataloaders

    def run(self):
        # Step 1: Load retain/val/forget datalaoders
        dataloaders = self.initialize_dataloaders()
        print('Loaders loaded')
        # Step 2: Initialize the pretrained model
        original_model = self.load_model()
        print('Original model loaded')

        save_model(original_model,
                   output_dir=self.output_dir,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='original')

        # Step 3: Perform unlearning
        unlearner = self.initialize_unlearner()
        print('Unlearning algorithm initialized.')
        start_time = time.time()
        unlearned_model, losses = unlearner.unlearn(original_model,
                                                    data_dict=dataloaders,
                                                    verbose=self.verbose,
                                                    **self.unlearn_params)
        # Log the time taken for the whole unlearning job
        job_run_time = (time.time() - start_time)/60
        print(f'Model unlearning complete. Time: {job_run_time:.1f} min')

        # Step 4: Save the unlearned model
        save_model(unlearned_model,
                   output_dir=self.output_dir,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='unlearned',
                   payload=losses)


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
    app = UnlearnApp(cfg)
    app.run()


if __name__ == '__main__':
    main()
