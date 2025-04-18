from unlearning.utils import ConfigError


class GLiRValidator:
    def __init__(self, config):
        assert isinstance(config, dict)

        self.output_dir = config['output_dir']
        
        self.threshold = config['attack']['threshold']
        self.num_params = config['attack']['num_params']
        self.small_var_lim = config['attack']['small_var_lim']
        self.test_size = config['attack']['test_size']

        self.dataset_name = config['dataset']['name']
        self.dataset_save_dir = config['dataset']['save_path']
        self.background_ratio = config['dataset']['background_ratio']
       
        model_config = config['model']
        self.model_loading = model_config['model_loading']

        self.verbose = config['verbose']
        self.visualise = config['visualise']

        self._validate_attack_params()
        self._validate_dataset_params()
        self._validate_model_params(model_config)
    
    def _validate_attack_params(self):
        if not (0 <= self.threshold <= 1):
            raise ConfigError("attack.threshold must be between 0 and 1 "
                              "(inclusive)")
        
        if self.small_var_lim <= 0:
            raise ConfigError("attack.small_var_lim must be positive")

    def _validate_dataset_params(self):
        valid_dataset_names = {'cifar5',
                               'cifar10',
                               'cifar100',
                               'imagenet'}
        if self.dataset_name not in valid_dataset_names:
            # raise ConfigError('Dataset support only for '
            #                   f'{', '.join(valid_dataset_names)}. '
            #                   f'Received {self.dataset_name}')
            raise ConfigError(
                f"Dataset support only for {', '.join(valid_dataset_names)}. "
                f"Received {self.dataset_name}"
                )

        if self.dataset_save_dir is None:
            raise ConfigError('A path to '
                              'load the dataset from must be provided. '
                              'dataset_save_dir cannot be empty.')
        
        if not (0 <= self.background_ratio <= 1):
            raise ConfigError("dataset.background_ratio must be between 0 and "
                              "1 (inclusive)")
        
    def _require(self, model_config, key):
        if key not in model_config:
            raise ConfigError(f"Missing required config key: '{key}'")
        return model_config[key]

    def _validate_model_params(self, model_config):
        """ Check some of the model-related config params from the YAML file.

        This validates only some of the model parameters, as it is more
        efficient to use the EAFP approach for model name and
        initialization checking."""
        if self.model_loading == 'class':
            self._validate_class_params(model_config)
        elif self.model_loading == 'torchhub':
            self._validate_torchhub_params(model_config)
        elif self.model_loading == 'torchvision':
            self._validate_torchvision_params(model_config)
        elif self.model_loading == 'timm':
            self._validate_timm_params(model_config)
        else:
            raise ConfigError('Invalid model loading ')

    def _validate_class_params(self, model_config):
        self.init_path = self._require(model_config, 'class_path')
        self.model_name = self._require(model_config, 'class_name')
        self.model_kwargs = self._require(model_config, 'model_kwargs')
        self.original_model_ckpt_path = model_config.get(
            'original_model_ckpt_path', 
            None
            )
        self.unlearned_model_ckpt_path = self._require(
            model_config,
            'unlearned_model_ckpt_path'
            )
        self.num_classes = None

    def _validate_torchhub_params(self, model_config):
        self.init_path = self._require(model_config, 'repo_path')
        self.model_name = self._require(model_config, 'model_name')
        self.original_model_ckpt_path = model_config.get(
            'original_model_ckpt_path', 
            None
            )
        self.unlearned_model_ckpt_path = self._require(
            model_config,
            'unlearned_model_ckpt_path'
            )
        self.model_kwargs = None
        self.num_classes = None

    def _validate_torchvision_params(self, model_config):
        self.model_name = self._require(model_config, 'model_name')
        self.num_classes = self._require(model_config, 'num_classes')
        self.original_model_ckpt_path = model_config.get(
            'original_model_ckpt_path', 
            None
            )
        self.unlearned_model_ckpt_path = self._require(
            model_config,
            'unlearned_model_ckpt_path'
            )
        self.init_path = None
        self.model_kwargs = None

    def _validate_timm_params(self, model_config):
        self.model_name = self._require(model_config, 'model_name')
        self.num_classes = self._require(model_config, 'num_classes')
        self.original_model_ckpt_path = model_config.get(
            'original_model_ckpt_path', 
            None
            )
        self.unlearned_model_ckpt_path = self._require(
            model_config,
            'unlearned_model_ckpt_path'
            )
        self.init_path = None
        self.model_kwargs = None
