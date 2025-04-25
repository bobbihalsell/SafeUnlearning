from omegaconf import DictConfig, OmegaConf
from cfg_validator import InputValidator


class TrainValidator(InputValidator):
    """ Validator specifically for training configurations."""
    def __init__(self, config: DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)
        self._validate_required_sections(['model', 'dataset',
                                          'trainer'])

    def _validate_dataset_params(self):
        super()._validate_dataset_params()
        cfg = self._require(self.dataset_file, 'cfg', 'dataset_cfg')
        for key, value in cfg.items():
            setattr(self, key, value)

    def _validate_model_params(self):
        super()._validate_model_params()
        self.pretrained = self.model_file.get('pretrained', None)
        self.checkpoint_path = self.model_file.get('original_model_ckpt_path',
                                                   None)
        self.from_checkpoint = (self.checkpoint_path is not None)

    def _validate_trainer_params(self):
        self._require(self.trainer_file, 'model_save_dir')
        self._require(self.trainer_file['cfg'], 'epochs')
        self._require(self.trainer_file['cfg'], 'lr')
        self._require(self.trainer_file['cfg'], 'weight_decay')

    def _validate_wandb_params(self):
        self._require(self.wandb_file, 'project_name')
        self._require(self.wandb_file, 'run_id')
