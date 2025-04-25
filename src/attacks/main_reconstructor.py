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
from importmodel import ImportModel
from attacks.utils import safe_dataclass_load, SaveImage, calculate_metrics, load_from_directory
from datasets import DATASETS_TO_PARAMS
from attacks.config_validation import ReconstructorValidator
from attacks.GGL.reconstructor import GGLReconstructor
from attacks.InvertGrad.reconstructor import InvertGradReconstructor,InvertGradConfig

DEFAULT_SEED = 42

class ReconstructorApp(ReconstructorValidator):
    """
    ReconstructorApp class for performing model reconstruction in unlearning attacks.
    """
    def __init__(self, config: DictConfig):
        """
    
        Initializes the ReconstructorApp by performing input validation, setting up the device,
        initializing seeds, and preparing directories for storing results.

        Args:
            config (DictConfig): Configuration dictionary containing settings for the reconstruction.
        
        Attributes:
            device (torch.device): Device to use for computation (e.g., 'cpu' or 'cuda').
            output_dir (str): Directory where results will be saved.
            forget_root (str): Directory for evaluation metrics during unlearning.
            original_model (nn.Module): The original model before unlearning.
            unlearned_model (nn.Module): The model after unlearning.
            reconstructor (object): The reconstructor instance to perform the attack (e.g., GGL or InvertGrad).
        
        """
        # Perform input validation first
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        set_seed(self.seed)

        # Output directory
        self.output_dir = config.get('output_dir', './artifacts/')
        os.makedirs(self.output_dir, exist_ok=True)

        # Forget root for eval metrics
        self.forget_root = f"{self.dataset_save_dir}/forget"
    
    def initialise_model(self):
        """
        Loads the original and unlearned models from the specified checkpoints.

        The method utilizes the ImportModel class to load both the original and unlearned models
        based on the specified parameters.

        Raises:
            ValueError: If the model type or checkpoint paths are not correctly specified.
        """
        
        original_import = ImportModel(self.load_method,
                                      self.model_name,
                                      self.num_classes,
                                      self.init_path,
                                      self.original_model_ckpt_path,
                                      self.model_kwargs
                                     )
        self.original_model = original_import.model
        unlearned_import = ImportModel(self.load_method,
                                      self.model_name,
                                      self.num_classes,
                                      self.init_path,
                                      self.unlearned_model_ckpt_path,
                                      self.model_kwargs
                                       )
        self.unlearned_model = unlearned_import.model
    
    def initalise_image_params(self):
        """
        Initializes the image parameters (mean, std, and size) based on the dataset being used.

        The method retrieves dataset-specific parameters from the DATASETS_TO_PARAMS mapping.

        Attributes set:
            image_mean (tuple): Mean of the dataset images for normalization.
            image_std (tuple): Standard deviation of the dataset images for normalization.
            image_size (list): Image size (channels, height, width) for processing.
        """
       
        self.image_mean, self.image_std, self.image_size = DATASETS_TO_PARAMS[self.dataset_name]
        self.image_size = [3, self.image_size, self.image_size]

    def initialize_reconstructor(self, unlearned_model, original_model):
        """ Initialize the correct unlearner from user specification.
        
        Args:
            unlearned_model (nn.Module): The model after unlearning.
            original_model (nn.Module): The original model before unlearning.
        
        Returns:
            reconstructor (object): The reconstructor object initialized for the specified attack.
        
        Raises:
            ValueError: If an unsupported attack type is specified.
        
        """

        loss_fn = nn.CrossEntropyLoss()
        if self.attack_name == 'ggl':
            reconstructor = GGLReconstructor(
                labels = self.labels,
                lr = self.attack_lr,
                exp_name = self.experiment_name,
                original_model = original_model,
                target_model=unlearned_model, 
                loss_fn = loss_fn,
                **self.attack_params
            )

        elif self.attack_name == 'invertgrad':
            reconstructor = InvertGradReconstructor(
                device = self.device,
                original_model = original_model,
                unlearned_model = unlearned_model,
                config = safe_dataclass_load(InvertGradConfig, self.attack_params),
                seed = self.seed
            )
        else:
            raise ValueError(f'reconstructor {self.attack_name}'
                             ' not supported.')
        self.reconstructor = reconstructor
        return reconstructor



    def save_results(self):
        """
        Creates a SaveImage instance for saving the reconstructed image and logging results.

        Returns:
            SaveImage: The image saver instance that handles saving and logging.
        """
        saver = SaveImage(self.attack_name,
                          self.seed,
                          self.experiment_name,
                          self.image_mean,
                          self.image_std,
                          self.output_dir)
        return saver



    def run(self):
        """
        Runs the reconstruction process by loading models, performing the attack, saving results, and calculating metrics.

        Raises:
            ValueError: If any error occurs during the model reconstruction process.
        """
        print('Running reconstruction...')
        # Step 1 : read yaml files
        # Step 2 : Load models model 
        self.initialise_model()

        # Step 3: Initialize wandb
        if self.wandb_enabled:
            config = self.attack_params.copy()
            config['type'] = self.attack_name
            config['reconstructor_lr'] = self.attack_lr
            config.update(self.extra_config)

            if self.run_id is None:
                self.run_id = wandb.util.generate_id()
            
            wandb.init(
                project = self.project_name,
                id = self.run_id,
                config = config,
                group = self.experiment_name
                )


        # Step 4: Reconstruction
        self.initalise_image_params()
        
        reconstructor = self.initialize_reconstructor(self.unlearned_model, self.original_model)
        print('Reconstructor initialized')

        start_time = time.time()
        if self.attack_name == 'ggl':
            z_res, reconstruction, losses = reconstructor.reconstruct(**self.unlearner_params)
        elif self.attack_name == 'invertgrad':
            reconstruction, losses = reconstructor.reconstruct(labels = self.labels,
                                                               num_images = self.attack_params['num_images'],
                                                            image_size= self.image_size,
                                                            image_mean= self.image_mean,
                                                            image_std=self.image_std,
                                                            lr= self.attack_lr,
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
            config_path="../../configs",
            config_name="attacks")
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