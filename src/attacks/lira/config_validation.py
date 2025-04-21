from unlearning.config_validation import InputValidator
from unlearning.utils import ConfigError


class LiRAValidator(InputValidator):
    """ Validator specifically for unlearning configurations."""
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections()
    
    def _validate_required_sections(self):
        """Ensure unlearning-specific sections exist."""
        required_sections = ['model', 'dataset', 'unlearner@original', 
                             'unleaner', 'attack']
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        self._require(self.dataset_file, 'val_ratio')
        self._require(self.dataset_file, 'forget_ratio')
        cfg = self._require(self.dataset_file, 'cfg', 'dataset_cfg')
        for key, value in cfg.items():
            setattr(self, key, value)
        
    def _validate_attack_params(self):
        self._require(self.attack_file, 'num_splits')
        self._require(self.attack_file, 'num_forgets')

    def _validate_trainer_params(self):
        """Validate parameters."""
        config = self.trainer_file
        self._require(config, 'evaluate', 'train_evaluate')
        cfg = self._require(config, 'cfg', 'trainer_cfg')
        
        required_params = ['epochs', 'lr', 'weight_decay', 'use_l2_penalty']
        
        # Common parameters for unlearners
        for param in required_params:
            self._require(cfg, param, f'trainer_{param}')