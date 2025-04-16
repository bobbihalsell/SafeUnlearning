from unlearning.utils import ConfigError


class InputValidator:
    """ Validates most inputs to the configuration YAML file."""
    def __init__(self, config):
        assert isinstance(config, dict)

        self.output_dir = config['output_dir']

        self.unlearner_name = config['unlearner']['name']
        self.unlearn_params = config['unlearner']['cfg']
        self.evaluate = config['unlearner']['evaluate']

        self.dataset_name = config['dataset']['name']
        self.dataset_save_dir = config['dataset']['save_path']
        self.dataset_cfg = config['dataset']['cfg']
        self.num_workers = self.dataset_cfg['num_workers']
        self.batch_sizes = self.dataset_cfg['batch_sizes']

        model_config = config['model']
        self.load_method = model_config['load_method']

        self.verbose = config['verbose']

        self._validate_unlearner_params()
        self._validate_dataset_params()
        self._validate_model_params(model_config)

    def _validate_unlearner_params(self):
        """
        Validate parameters required by each unlearner type.
        Raises exception if required parameters are missing.
        """
        # common parameters for unlearners
        required_params = ['epochs',
                           'lr',
                           'weight_decay',
                           'use_l2_penalty',
                           ]

        missing_params = []
        for param in required_params:
            if param not in self.unlearn_params:
                if self.unlearner_name == 'scrub' and param == 'epochs':
                    continue
                missing_params.append(param)

        if missing_params:
            raise ConfigError('Missing required parameters for unlearning: '
                              f'{', '.join(missing_params)}')

        # Add specific parameters based on unlearner type
        if self.unlearner_name == 'neggradplus':
            try:
                self.unlearn_params['beta']
            except KeyError:
                raise ConfigError('Missing required parameter beta'
                                  ' for NegGradPlus unlearner')

        elif self.unlearner_name == 'scrub':
            # Check for required SCRUB-specific parameters
            required_scrub_params = ['min_epochs', 'max_epochs',
                                     'alpha', 'gamma']
            missing_scrub_params = []
            for param in required_scrub_params:
                if param not in self.unlearn_params:
                    missing_scrub_params.append(param)
            if missing_scrub_params:
                raise ConfigError(
                    'Missing required params for SCRUB unlearner:'
                    f' {', '.join(missing_scrub_params)}'
                                  )

        elif self.unlearner_name in {'euk', 'cfk'}:
            # Check for k parameter
            try:
                self.unlearn_params['k']
            except KeyError:
                raise ConfigError(f'Missing required parameter k for '
                                  f'{self.unlearner_name.upper()} unlearner')
            if self.unlearner_name == 'euk':
                # Check for EUk-specific parameters
                if 'reinit_method' not in self.unlearn_params:
                    raise ConfigError('Missing required parameter '
                                      'reinit_method for EUk unlearner')

    def _validate_dataset_params(self):
        valid_dataset_names = {'cifar5',
                               'cifar10',
                               'cifar100',
                               'imagenet'}
        if self.dataset_name not in valid_dataset_names:
            raise ConfigError('Dataset support only for '
                              f'{', '.join(valid_dataset_names)}. '
                              f'Received {self.dataset_name}')
        if self.dataset_save_dir is None:
            raise ConfigError('A path to '
                              'load the dataset from must be provided. '
                              'dataset_save_dir cannot be empty.')

    def _require(self, model_config, key):
        if key not in model_config:
            raise ConfigError(f"Missing required config key: '{key}'")
        return model_config[key]

    def _validate_model_params(self, model_config):
        """ Check some of the model-related config params from the YAML file.

        This validates only some of the model parameters, as it is more
        efficient to use the EAFP approach for model name and
        initialization checking."""
        if self.load_method == 'class':
            self._validate_class_params(model_config)
        elif self.load_method == 'torchhub':
            self._validate_torchhub_params(model_config)
        elif self.load_method == 'torchvision':
            self._validate_torchvision_params(model_config)
        elif self.load_method == 'timm':
            self._validate_timm_params(model_config)
        else:
            raise ConfigError('Invalid model loading ')

    def _validate_class_params(self, model_config):
        self.init_path = self._require(model_config, 'class_path')
        self.model_name = self._require(model_config, 'class_name')
        self.model_kwargs = self._require(model_config, 'model_kwargs')
        self.model_ckpt_path = model_config.get('model_ckpt_path', None)
        self.num_classes = None

    def _validate_torchhub_params(self, model_config):
        self.init_path = self._require(model_config, 'repo_path')
        self.model_name = self._require(model_config, 'model_name')
        self.model_ckpt_path = model_config.get('model_ckpt_path', None)
        self.model_kwargs = None
        self.num_classes = None

    def _validate_torchvision_params(self, model_config):
        self.model_name = self._require(model_config, 'model_name')
        self.num_classes = self._require(model_config, 'num_classes')
        self.model_ckpt_path = model_config.get('model_ckpt_path', None)
        self.init_path = None
        self.model_kwargs = None

    def _validate_timm_params(self, model_config):
        self.model_name = self._require(model_config, 'model_name')
        self.num_classes = self._require(model_config, 'num_classes')
        self.model_ckpt_path = model_config.get('model_ckpt_path', None)
        self.init_path = None
        self.model_kwargs = None
