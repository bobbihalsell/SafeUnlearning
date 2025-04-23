from utils import ConfigError


class InputValidator:
    """ Base validator for configuration YAML files."""
    def __init__(self, config):
        assert isinstance(config, dict)
        self.config = config

        for key in config:
            if isinstance(config[key], dict):
                cfg = config[key]
                setattr(self, f'{key}_file', cfg)
                validate_method = f'_validate_{key}_params'
                if hasattr(self, validate_method):
                    getattr(self, validate_method)()
            else:
                setattr(self, key, config[key])
        self.wandb_enabled = hasattr(self, 'wandb_file')

    def _require(self, model_config, key, alias=None):
        if key not in model_config:
            raise ConfigError(f"Missing required config key: '{key}'")
        if alias is None:
            alias = key
        setattr(self, alias, model_config[key]) 
        return model_config[key]
    
    def _validate_required_sections(self, required_sections):
        """Ensure unlearning-specific sections exist."""
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )
        
    def _validate_unlearner_params(self):
        """Validate parameters required by each unlearner type."""
        config = self.unlearner_file
        
        # Require basic unlearner fields
        unlearner_name = self._require(config, 'name', 'unlearner_name')
        self._require(config, 'evaluate')
        cfg = self._require(config, 'cfg', 'unlearn_params')
        
        required_params = ['epochs', 'lr', 'weight_decay', 'use_l2_penalty']
        
        # Common parameters for unlearners
        for param in required_params:
            if unlearner_name == 'scrub' and param == 'epochs':
                continue
            self._require(cfg, param)
        
        # Add specific parameters based on unlearner type
        if unlearner_name == 'neggradplus':
            self._require(cfg, 'beta')
        
        elif unlearner_name == 'scrub':
            required_scrub_params = ['min_epochs', 
                                     'max_epochs', 
                                     'alpha', 
                                     'gamma']
            for param in required_scrub_params:
                self._require(cfg, param)
        
        elif unlearner_name in {'euk', 'cfk'}:
            self._require(cfg, 'k')
            if unlearner_name == 'euk':
                self._require(cfg, 'reinit_method')
        
        elif unlearner_name == 'gradproj':
            required_gradproj_params = ['recalc_freq', 
                                        'num_components', 
                                        'redirection_strength', 
                                        'max_grad_norm']
            for param in required_gradproj_params:
                self._require(cfg, param)
    
    def _validate_dataset_params(self):
        config = self.dataset_file
        self._require(config, 'name', 'dataset_name')
        valid_dataset_names = {'cifar5', 'cifar10', 'cifar100', 'imagenet'}        
        if self.dataset_name not in valid_dataset_names:
            raise ConfigError(f'Dataset support only for '
                              f'{", ".join(valid_dataset_names)}. '
                              f'Received {self.dataset_name}')
        self._require(self.dataset_file, 'num_classes')
        self._require(config, 'save_path', 'dataset_save_dir')

    def _validate_model_params(self):
        """Check model-related config params from the YAML file."""
        config = self.model_file
        load_method = self._require(config, 'load_method')
        
        if load_method == 'class':
            self._validate_class_params(config)
        elif load_method == 'torchhub':
            self._validate_torchhub_params(config)
        elif load_method == 'torchvision':
            self._validate_torchvision_params(config)
        elif load_method == 'timm':
            self._validate_timm_params(config)
        else:
            raise ConfigError(f'Invalid model loading method: {load_method}')
        
        for arg in ['init_path', 'model_kwargs']:
            if not hasattr(self, arg):
                setattr(self, arg, None)
        for key in config:
            if key.endswith('_ckpt_path'):
                setattr(self, key, config[key])
        self.model_ckpt_path = config.get('model_ckpt_path', None)
        if not hasattr(self, 'pretrained'):
            setattr(self, 'pretrained', True)
        
    def _validate_class_params(self, config):
        self._require(config, 'class_path', 'init_path')
        self._require(config, 'class_name', 'model_name')
        self._require(config, 'model_kwargs', 'model_kwargs')
    
    def _validate_torchhub_params(self, config):
        self._require(config, 'repo_path', 'init_path')
        self._require(config, 'model_name')
    
    def _validate_torchvision_params(self, config):
        self._require(config, 'model_name', 'model_name')
    
    def _validate_timm_params(self, config):
        self._require(config, 'model_name', 'model_name')

    def _validate_wandb_params(self):
        self._require(self.wandb_file, 'project_name')
        self._require(self.wandb_file, 'run_id')
        self.wandb_extra_config = self.wandb_file.get('extra_config', {})
