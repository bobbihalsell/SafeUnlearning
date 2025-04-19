from unlearning.utils import ConfigError
from cfg_validator import InputValidator


class UnlearningValidator(InputValidator):
    """ Validator specifically for unlearning configurations."""
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections()
        if not hasattr(self, 'wandb_config'):
            self.wandb_config = None
    
    def _validate_required_sections(self):
        """Ensure unlearning-specific sections exist."""
        required_sections = ['model', 'dataset', 'unlearner']
        for section in required_sections:
            if not hasattr(self, f'{section}_file'):
                raise ConfigError(
                    f"Missing required config section: '{section}'"
                    )

    def _validate_dataset_params(self, config):
        super()._validate_dataset_params(config)
        cfg = self._require(config, 'cfg', 'dataset_cfg')
        for key, value in cfg.items():
            setattr(self, key, value)