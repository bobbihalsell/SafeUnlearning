from utils import ConfigError


class DatasetValidator:
    """ Unpack and validate DatasetInitializer's parameters passed from config."""
    def __init__(self, config):
        self._unpack_dataset_parameters(config=config)
        self._validate_dataset_args()
        if self.dataset_load_method == 'torchvision':
            self._validate_torchvision_args()
        self._unpack_forget_parameters(config=config)
        self._validate_forget_args()

    def _unpack_dataset_parameters(self, config: dict):
        """ Set the dataset-related keys in the config as attr/value pairs.
        Args:
            config (dict): The overall configuration dictionary from OmegaConf.
        Returns:
            None
        """
        dataset_cfg = config['dataset']
        self.dataset_load_method = dataset_cfg['load_method']

        for key, value in dataset_cfg.items():
            if key != 'method':  # Already set as self.dataset_load_method
                setattr(self, key, value)
            if key == 'name':
                setattr(self, 'dataset_name', value)

    def _unpack_forget_parameters(self, config: dict):
        """ Set the forget-related keys in the config as attr/value pairs.
        Args:
            config (dict): The overall configuration dictionary from OmegaConf.
        Returns:
            None
        """
        forget_cfg = config['forget']
        self.forget_method = forget_cfg['method']
        for key, value in forget_cfg.items():
            if key != 'method':  # Already set as self.forget_method
                setattr(self, key, value)

    def _validate_torchvision_args(self):
        """
        Validate that torchvision-related arguments are of correct types.

        Raises:
            ConfigError: If `dataset_name` or `binaries_download_dir` is not a string.
        """
        if not isinstance(self.dataset_name, str):
            raise ConfigError("dataset_name must be a string")
        if not isinstance(self.binaries_download_dir, str):
            raise ConfigError("binaries_download_dir must be a string")

    def _validate_dataset_args(self):
        """
        Validate dataset-related configuration arguments.

        Raises:
            ConfigError: If any value is of the wrong type or out of the expected range.
        """
        if self.init_path is None:
            self.init_path = 'tmp_dir'
        if not isinstance(self.proportion, (float, int)):
            raise ConfigError("proportion must be a float or int")
        if not (0 <= self.proportion <= 1):
            raise ConfigError("proportion must be between 0 and 1")
        if not isinstance(self.val_ratio, (float, int)):
            raise ConfigError("val_ratio must be a float or int")
        if not (0 <= self.val_ratio <= 1):
            raise ConfigError("val_ratio must be between 0 and 1")
        if not isinstance(self.init_path, str):
            raise ConfigError("init_path must be a string")
        if not isinstance(self.save_path, str):
            raise ConfigError("save_path must be a string")

    def _validate_forget_args(self):
        """
        Validate forgetting method arguments based on the selected method.
        
        Raises:
            ConfigError: If types do not match expectations or if method is unknown.
        """
        if self.forget_method == 'class':
            if not isinstance(self.forget_idx, list):
                raise ConfigError(
                    'forget_idx should be a list of labels. '
                    f'Received {type(self.forget_idx)}.'
                )
        elif self.forget_method == 'classnum':
            if not isinstance(self.forget_idx, dict):
                raise ConfigError(
                    'forget_idx should be a dict of label/count pairs. '
                    f'Received {type(self.forget_idx)}.'
                )
        elif self.forget_method == 'random_n':
            if not isinstance(self.forget_size, int):
                raise ConfigError(
                    'forget_size should be an int. '
                    f'Received {type(self.forget_size)}.'
                )
        elif self.forget_method == 'filename':
            if not isinstance(self.forget_filenames, list):
                raise ConfigError(
                    'forget_filenames should be a list of ints. '
                    f'Received {type(self.forget_filenames)}.'
                )
        else:
            raise ConfigError(
                'Unknown forget_method. '
                'Allowed: class, classnum, random_n, filename. '
                f'Received {self.forget_method}')
