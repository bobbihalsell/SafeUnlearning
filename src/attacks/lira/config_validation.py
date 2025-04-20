from unlearning.config_validation import InputValidator
from unlearning.utils import ConfigError


class LiRAValidator(InputValidator):
    def __init__(self, config):
        self.forget_ratio = config["dataset"]["forget_ratio"]
        self.val_ratio = config["dataset"]["val_ratio"]
        self.num_splits = config["attack"]["cfg"]["num_splits"]
        self.num_forgets = config["attack"]["cfg"]["num_splits"]
        super().__init__(config)

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        if self.num_splits is None:
            raise ConfigError("Missing required parameter num_splits for LiRA")

        if self.num_forgets is None:
            raise ConfigError("Missing required parameter num_forgets for LiRA")
