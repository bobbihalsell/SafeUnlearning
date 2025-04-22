from cfg_validator import InputValidator, ConfigError

import os


class ReconstructorValidator(InputValidator):
    def __init__(self, config):
        super().__init__(config)

        self._validate_required_sections()
        # self._validate_dataset_params()
        # self._validate_model_params()
        # self._validate_reconstructor_params()
        

    def _validate_required_sections(self):
        """Ensure unlearning-specific sections exist."""
        required_sections = ['model', 'dataset', 'reconstructor']
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )
            else:
                print(f"Found required section: {section}")
    
    def _validate_reconstructor_params(self):
        print("validating reconstructor params")
        recon_config = self.reconstructor_file
        print(f"Recon config: {recon_config}")
        method = self._require(recon_config, 'type', 'reconstructor_name')
        print("===================")
        print(self.reconstructor_name)

        self.labels = self._load_labels(recon_config.get('unlearned_labels', None))

        if self.reconstructor_name == 'ggl':
            self._validate_ggl_params(recon_config)
        elif self.reconstructor_name == 'invertgrad':
            self._validate_invertgrad_params(recon_config)
        else:
            raise ConfigError(f"Unsupported unlearning method: {method}")
        
        self._require(recon_config, 'lr', 'reconstructor_lr')


    def _validate_ggl_params(self, config):
        self.recon_params = self._require(config, 'cfg', 'reconstructor_params')

        required_params = ['num_updates', 'unlearning_method', 
                           'batch_size', 'loss_models', 'budget',
                           'search_dim', 'use_tanh', 
                           'gp_optim', 'use_scheduler', 
                           'initial_lr', 'unlearned_labels']
        for param in required_params:
            self._require(self.recon_params, param)

        self.unlearner_params = config.get('unlearner', {})
          
    def _validate_invertgrad_params(self, config):
        self.recon_params = self._require(config, 'cfg', 'reconstructor_params')

        required_params = ['grad_diff_lr', 'signed', 'boxed',
                           'cost_fn', 'indices', 'weights',
                           'optim', 'num_runs', 'recon_iterations',
                           'total_variation', 'init', 'lr_decay',
                           'scoring_choice', 'eval', 'filter']
        
        for param in required_params:
            self._require(self.recon_params, param)
        
        self.num_images = config.get('num_images', None)
        if self.labels is None and self.num_images is None:
            raise ConfigError('Either labels or num_images must be provided.')

    
    def _validate_wandb_params(self):
        self._require(self.wandb_file, 'project_name')
        if self.wandb_file.get('extra_config', None) is not None:
            self.extra_config = self._require(self.wandb_file, 'extra_config')
        if self.wandb_file.get('run_id', None) is not None:
            self.run_id = self._require(self.wandb_file, 'run_id')


        
    def _load_labels(self, labels_cfg):
        if isinstance(labels_cfg, str) and os.path.isfile(labels_cfg):
            with open(labels_cfg, 'r') as f:
                return [int(line.strip()) for line in f if line.strip()]
        elif isinstance(labels_cfg, list):
            return labels_cfg
        else:
            return None    

class InputValidator:
    """ Validates most inputs to the configuration YAML file."""
    def __init__(self, config):
        assert isinstance(config, dict)



                   
        if self.wandb is not None:
            self.wandb_enabled = True
            self.wandb_project = self.wandb['project_name']
            self.run_id = self.wandb.get('run_id', None)
            self.extra_config = self.wandb.get('extra_config', {})
        else:
            self.wandb_enabled = False
        
        self._validate_experiment_params()
        self._validate_dataset_params()

        self._validate_reconstructor_params()
        
