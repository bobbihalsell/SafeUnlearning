from attacks.utils import ConfigError
import os


class InputValidator:
    """ Validates most inputs to the configuration YAML file."""
    def __init__(self, config):
        assert isinstance(config, dict)

        self.experiment_name = config['experiment']['name']
        self.seed = config['experiment']['seed']
        self.model_name = config['experiment']['model_name']

        self.oiginal_weights = config['paths']['original_weights']
        self.unlearned_weights = config['paths']['unlearned_weights']

        data_config = config['data']
        self.labels = data_config['labels']
        self.dataset_name = data_config['dataset_name']
        self.image_mean = data_config['image_mean']
        self.image_std = data_config['image_std']
        self.image_size = data_config['image_size']
        self.batch_size = data_config['batch_size']
        #self.num_workers = data_config['num_workers] don't know if we need that?
        self.num_classes = data_config['num_classes']

        self.reconstructor_name = config['Reconstructor']['type']
        self.reconstructor_lr = config['Reconstructor']['lr']

        if self.reconstructor_name == 'ggl':
            reconstructor = config['Reconstructor']['GGL']
            self.reconstructor_params = config["Reconstructor"]['GGL']
            self.num_updates = reconstructor['num_updates']
            self.unlearning_method = reconstructor['unlearning_method']
            self.alpha = reconstructor['alpha']
            self.gamma = reconstructor['gamma']
            self.min_epochs = reconstructor['min_epochs']
            self.max_epochs = reconstructor['max_epochs']
            self.batch_size = reconstructor['batch_size']
            self.loss_models = reconstructor['loss_models'] # 'l1', 'l2', 'weighted', or 'interpolated'
            self.budget = reconstructor['budget'] # Budget for Bayesian Optimization
            self.search_dim = reconstructor['search_dim'] # Dimension of the latent space
            self.use_tanh = reconstructor['use_tanh'] # Whether to apply tanh activation to the latent vector

        elif self.reconstructor_name == 'inversegrad':
            reconstructor = config['Reconstructor']['InverseGrad']
            self.reconstructor_params = config["Reconstructor"]['InverseGrad']
            self.grad_lr = reconstructor['grad_lr']
            self.rec_experiments = reconstructor['rec_experiments']
            self.rec_epochs = reconstructor['rec_epochs']
            self.boxed = reconstructor['boxed']
            self.cost_fn = reconstructor['cost_fn']
            self.indices = reconstructor['indices']
            self.weights = reconstructor['weights']
            self.optim = reconstructor['optim']
            self.total_variation = reconstructor['total_vaariation']
            self.init = reconstructor['init']
            self.lr_decay = reconstructor['lr_decay']
            self.scoring_choice = reconstructor['scoring_choice']

        self.unlearner_name = config['Unlearner']['name']
        self.unlearn_params = config['Unlearner']['cfg']
        

        self.verbose = config['verbose']

        self._validate_unlearner_params()
        #self._validate_dataset_params()
        #self._validate_model_params()

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
                raise ConfigError('Missing required params for SCRUB unlearner:'
                                  f' {', '.join(missing_scrub_params)}')

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

    def _validate_model_params(self):
        """ Check some of the model-related config params from the YAML file.

        This validates only some of the model parameters, as it is more
        efficient to use the EAFP approach for model name and
        initialization checking."""
        if not os.path.exists(self.unlearned_model_ckpt_path):
            raise ConfigError("Model weights file not found: "
                              f"{self.unlearned_model_ckpt_path}")
