from unlearning.config_validation import InputValidator
from unlearning.utils import ConfigError


class LiRAValidator(InputValidator):
    def __init__(self, config):
        self.attack_cfg = config["attack"]["cfg"]
        super().__init__(config)

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        if self.attack_cfg["num_splits"] is None:
            raise ConfigError("Missing required parameter num_splits for LiRA")

        if self.attack_cfg["num_forgets"] is None:
            raise ConfigError("Missing required parameter num_forgets for LiRA")
