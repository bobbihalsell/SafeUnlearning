import numpy as np
from omegaconf import DictConfig, OmegaConf
from cfg_validator import InputValidator


class DatasetValidator(InputValidator):
    def __init__(self, config: DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)
        self._validate_required_sections(['dataset', 'forget'])
        if not hasattr(self, 'wandb_config'):
            self.wandb_config = None
        seed = config.get('seed', 42)
        np.random.seed(seed)
            
    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        self._require(self.dataset_file, 'save_path')
        self._require(self.dataset_file, 'init_path')
        self._require(self.dataset_file, 'proportion')
        self._require(self.dataset_file, 'val_ratio')

    def _validate_forget_params(self, ):
        super()._validate_dataset_params()
        self._require(self.forget_file, 'method', 'forget_method')
        self._require(self.forget_file, 'forget_idx')
        self.retain_size = self.forget_file.get('retain_size', None)
        self.num = self.forget_file.get('num', None)