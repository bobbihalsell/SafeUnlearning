from src.unlearning.config_validation import InputValidator
from src.unlearning.utils import ConfigError


class LiRAValidator(InputValidator):
    def __init__(self, config):
        super().__init__(config)

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        if self.dataset_cfg['num_splits'] is None:
            raise ConfigError('Missing required parameter num_splits for LiRA')

        if self.dataset_cfg['num_forgets'] is None:
            raise ConfigError('Missing required parameter num_forgets for LiRA')
