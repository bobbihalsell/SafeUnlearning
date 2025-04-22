import os
import time
import timm
import torch
import torch.nn as nn
import torchvision

import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import MissingMandatoryValue
import wandb

from utils import set_seed, setup_device
from unlearning.importmodel import ImportModel  #TODO: change once importmodel is refactored
from attacks.utils import safe_dataclass_load, SaveImage, calculate_metrics, load_from_directory
from datasets import DATASETS_TO_PARAMS
from attacks.config_validation import InputValidator
from attacks.GGL.reconstructor import GGLReconstructor
from attacks.InvertGrad.reconstructor import InvertGradReconstructor,InvertGradConfig

DEFAULT_SEED = 42

class ReconstructorApp(InputValidator):
    def __init__(self, config: DictConfig):
        # Perform input validation first
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)

        self.device = setup_device()
        print(config.keys())
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        # Output directory
        self.output_dir = config.get('output_dir', './artifacts/')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_original_model(self):
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
    
    def load_unlearned_model(self):
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
    
    def initalise_image_params(self):
        self.image_mean, self.image_std, self.image_size = DATASETS_TO_PARAMS[self.dataset_name]
        self.image_size = [3, self.image_size, self.image_size]

    def initialize_reconstructor(self, unlearned_model, original_model):
        """ Initialize the correct unlearner from user specification."""

        loss_fn = nn.CrossEntropyLoss()
        if self.reconstructor_name == 'ggl':
            reconstructor = GGLReconstructor(
                labels = self.labels,
                lr = self.reconstructor_lr,
                exp_name = self.experiment_name,
                original_model = original_model,
                target_model=unlearned_model, 
                loss_fn = loss_fn,
                **self.reconstructor_params
            )

        elif self.reconstructor_name == 'invertgrad':
            reconstructor = InvertGradReconstructor(
                device = self.device,
                original_model = original_model,
                unlearned_model = unlearned_model,
                config = safe_dataclass_load(InvertGradConfig, self.reconstructor_params),
                seed = self.seed
            )
        else:
            raise ValueError(f'reconstructor {self.reconstructor_name}'
                             ' not supported.')
        self.reconstructor = reconstructor
        return reconstructor



    def save_results(self):
        saver = SaveImage(self.reconstructor_name,
                          self.seed,
                          self.experiment_name,
                          self.image_mean,
                          self.image_std,
                          self.output_dir)
        return saver



    def run(self):
        print('running...')
        # Step 1 : read yaml files
        # Step 2 : load unlearned model 
        unlearned_model = self.load_model_from_disk(self.unlearned_weights)

        # Step 3: Initialize the pretrained model
        original_model = self.load_model_from_disk(self.original_weights)

        # Step 3: Initialize wandb
        if self.wandb_enabled:
            config = self.reconstructor_params.copy() 
            config['type'] = self.reconstructor_name
            config['reconstructor_lr'] = self.reconstructor_lr  
            config.update(self.extra_config)

            if self.run_id is None:
                self.run_id = wandb.util.generate_id()
            
            wandb.init(
                project = self.wandb_project,
                id = self.run_id,
                config = config,
                group = self.experiment_name
                )
        

        # Step 4: Reconstruction
        self.initalise_image_params()
        
        reconstructor = self.initialize_reconstructor(unlearned_model, original_model)
        print('reconstructor initialized')

        start_time = time.time()
        if self.reconstructor_name == 'ggl':
            z_res, reconstruction, losses = reconstructor.reconstruct(**self.unlearner_params)
        elif self.reconstructor_name == 'invertgrad':
            reconstruction, losses = reconstructor.reconstruct(labels = self.labels,
                                                               num_images = self.reconstructor_params['num_images'],
                                                            image_size= self.image_size,
                                                            image_mean= self.image_mean,
                                                            image_std=self.image_std,
                                                            lr= self.reconstructor_lr,
                                                            verbose=self.verbose)
        total_time = time.time() - start_time

        print(f"Reconstruction completed in {total_time:.2f} seconds")
        print(f"Reconstruction Losses: {losses}")


        # # Step 5: Save the reconstructed image
        image_saver = self.save_results()
        image_saver.save_png(reconstruction,
                             normalize=False)
        
        # # Step 6: Calculate metrics
        ref_batch = load_from_directory(self.forget_root)
        psnr_value, mse_value = calculate_metrics(reconstruction, ref_batch, self.image_size, self.verbose)


        # # Step 7: Save metrics
        if self.wandb_enabled:
            table_psnr = wandb.Table(columns=["img_id", "psnr", "best_ref_index"])
            for i, (p, idx) in enumerate(psnr_value):
                table_psnr.add_data(i, p, idx)

            table_mse = wandb.Table(columns=["img_id", "mse", "best_ref_index"])
            for i, (mse, idx) in enumerate(mse_value):
                table_mse.add_data(i, mse, idx)

            psnr_max = max([p for p, _ in psnr_value])
            mse_min = min([mse for mse, _ in mse_value])

            wandb.log({
                'reconstruction_time': total_time,
                'losses': losses,
                'reconstruction': wandb.Image(reconstruction),
                'psnr_table': table_psnr,
                'mse_table': table_mse,
                'max_psnr': psnr_max,
                'min_mse': mse_min
            })
            wandb.finish()
        
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
    app = ReconstructorApp(cfg)
    app.run()

if __name__ == '__main__':
    main()

    # Run pip install -e .
    # Run python src/attacks/main_reconstructor.py data=cifar10 reconstructor=invertgrad