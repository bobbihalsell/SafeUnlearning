from cfg_validator import InputValidator, ConfigError
import os


class GLiRValidator(InputValidator):
    """ Validator specifically for unlearning configurations."""
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections()
    
    def _validate_required_sections(self):
        """Ensure unlearning-specific sections exist."""
        required_sections = ['model', 'dataset', 'attack']
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )


class ReconstructorValidator(InputValidator):
    def __init__(self, config):
        super().__init__(config)

        self._validate_required_sections()

    def _validate_required_sections(self):
        """Ensure unlearning-specific sections exist."""
        required_sections = ['model', 'dataset', 'reconstructor']
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )
    
    def _validate_reconstructor_params(self):
        recon_config = self.reconstructor_file
        method = self._require(recon_config, 'type', 'reconstructor_name')


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
                           'initial_lr']
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
        self.extra_config = self._require(self.wandb_file, 'extra_config') if self.wandb_file.get('extra_config') else {}
        self.run_id = self._require(self.wandb_file, 'run_id') if self.wandb_file.get('run_id') else None


        
    def _load_labels(self, labels_cfg):
        if isinstance(labels_cfg, str) and os.path.isfile(labels_cfg):
            try:
                with open(labels_cfg, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    labels = [int(line) for line in lines]
                if not labels:
                    raise ConfigError(f"Label file {labels_cfg} is empty.")
                return labels
            except ValueError:
                raise ConfigError(f"Invalid label format in {labels_cfg}. Expected integers.")
        
        elif isinstance(labels_cfg, list):
            if all(isinstance(label, int) for label in labels_cfg):
                return labels_cfg
            else:
                raise ConfigError(f"Invalid label format in {labels_cfg}. Expected a list of integers.")
        else:
            return None

