from cfg_validator import InputValidator


class UnlearningValidator(InputValidator):
    """ 
    Validator specifically for unlearning configurations.
    Inherits from `InputValidator` and ensures that the configuration file contains
    all necessary fields related to the model, dataset, and unlearning method.

    Attributes:
        wandb_config (Optional[dict]): Weights & Biases configuration, if provided.
    
    """
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections(['model', 'dataset', 'unlearner'])
        if not hasattr(self, 'wandb_config'):
            self.wandb_config = None

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        cfg = self._require(self.dataset_file, 'cfg', 'dataset_cfg')
        for key, value in cfg.items():
            setattr(self, key, value)

    def _validate_model_params(self):
        super()._validate_model_params()
        self.pretrained = self.model_file.get('pretrained', True)