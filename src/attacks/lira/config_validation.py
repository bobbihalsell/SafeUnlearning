from cfg_validator import InputValidator


class LiRAValidator(InputValidator):
    """ Validator specifically for unlearning configurations."""
    def __init__(self, config):
        super().__init__(config)
        self._validate_required_sections(['model', 'dataset', 'trainer',
                                          'unlearner', 'attack'])

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