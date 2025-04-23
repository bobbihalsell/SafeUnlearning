import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import MissingMandatoryValue
from unlearning.config_validation import UnlearningValidator
from unlearning.finetune import FinetuneUnlearner
from unlearning.scrub import SCRUB
from unlearning.neggrad import NegGrad, NegGradPlus
from unlearning.SGRU import SGRU
from unlearning.kunlearn import KUnlearn
from utils import set_seed, setup_device
from unlearning.unlearn_utils import save_model
from importmodel import ImportModel
from utils import initialize_dataloaders as init_dataloaders
import time
import wandb


class UnlearnApp(UnlearningValidator):
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
        # Set to true in main() if user provided wandb_config
        # self.wandb_enabled = False
        print(f' num_classes: {self.num_classes}')

    def load_model(self):
        """Initialize the model based on model name from user configuration."""
        importer = ImportModel(
            load_method=self.load_method,
            model_name=self.model_name,
            num_classes=self.num_classes,
            init_path=self.init_path,
            model_ckpt_path=self.model_ckpt_path,
            model_kwargs=self.model_kwargs,
            from_pretrained=self.pretrained,
            )
        model = importer.model
        return model

    def initialize_unlearner(self):
        """ Initialize the correct unlearner from user specification."""
        if self.unlearner_name == 'finetune':
            unlearner = FinetuneUnlearner(
                self.device,
                self.evaluate,
                self.wandb_enabled
            )
        elif self.unlearner_name == 'neggrad':
            unlearner = NegGrad(
                self.device,
                self.evaluate,
                self.wandb_enabled
            )
        elif self.unlearner_name == 'neggradplus':
            unlearner = NegGradPlus(
                self.device,
                self.evaluate,
                self.wandb_enabled
            )
        elif self.unlearner_name == 'scrub':
            unlearner = SCRUB(
                self.device,
                self.evaluate,
                self.wandb_enabled
            )
        elif self.unlearner_name == 'sgru':
            unlearner = SGRU(
                self.device,
                self.evaluate,
                self.wandb_enabled
            )
        elif self.unlearner_name == 'euk':
            unlearner = KUnlearn(
                self.k,
                self.device,
                self.unlearner_name,
                self.reinit_method,
                self.evaluate,
                self.wandb_enabled
            )
        elif self.unlearner_name == 'cfk':
            unlearner = KUnlearn(
                self.k,
                self.device,
                self.unlearner_name,
                None,
                self.evaluate,
                self.wandb_enabled
            )
        else:
            raise ValueError(f'unlearner_name {self.unlearner_name}'
                             ' not supported.')
        self.unlearner = unlearner
        return unlearner
    
    def initialize_dataloaders(self):
        """ Initialize dataloaders from the dataset folder."""
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

        dataloaders = init_dataloaders(splits=splits,
                                       batch_sizes=self.batch_sizes,
                                       num_workers=self.num_workers,
                                       dataset_name=self.dataset_name,
                                       dataset_save_dir=self.dataset_save_dir)
        dataloader = dataloaders['forget']
        sample_input, _ = next(iter(dataloader))
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
                   model_type='original',
                   id=self.id)

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
        if self.wandb_enabled:
            wandb.log({'total_run_mins': job_run_time})

        # Step 4: Save the unlearned model
        save_model(unlearned_model,
                   output_dir=self.output_dir,
                   unlearning_algorithm=self.unlearner_name,
                   model_name=self.model_name,
                   seed=self.seed,
                   model_type='unlearned',
                   id=self.id,
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
    if app.wandb_enabled is not None:
        wandb.init(
            project=app.project_name,
            id=app.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume='never'  # Always make sure the unlearning run ID is new
        )
    app.run()
    if app.wandb_enabled:
        wandb.finish()


if __name__ == '__main__':
    main()
