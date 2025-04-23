from cfg_validator import InputValidator, ConfigError


class GLiRValidator(InputValidator):
    """ Validator specifically for unlearning configurations."""
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections(['model', 'dataset', 'attack'])
    
    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        self._require(self.dataset_file, 'background_ratio')
        
    def _validate_attack_params(self):
        self._require(self.attack_file, 'threshold')
        if not (0 <= self.threshold <= 1):
            raise ConfigError("threshold must be between 0 and 1 "
                              "(inclusive)")
        self._require(self.attack_file, 'num_params')
        if self.num_params <= 0 or not isinstance(self.num_params, int):
            raise ConfigError("num_params must be positive integer")
        self._require(self.attack_file, 'small_var_lim')
        if self.small_var_lim <= 0:
            raise ConfigError("small_var_lim must be positive")
        self._require(self.attack_file, 'test_size')
        if (self.test_size is not None 
           and not (0 <= self.test_size <= 1)):
            raise ConfigError("test_size must be between 0 and 1 "
                              "(inclusive)")