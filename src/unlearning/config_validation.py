class InputValidator:
    def __init__(self, config):
        assert isinstance(config, dict)
        self.unlearner_name = config['unlearner']['name']
        self.unlearn_params = config['unlearner']['cfg']
        self._validate_unlearner_params()

    def _validate_unlearner_params(self):
        """
        Validate parameters required by each unlearner type.
        Raises exception if required parameters are missing.
        """
        if not (self.unlearn_params['loss_fn'] == 'cross_entropy'):
            raise AttributeError('loss_fn must be cross_entropy, '
                                 f'received {self.unlearn_params['loss_fn']}')
        # common parameters for unlearners
        required_params = ['epochs',
                           'lr',
                           'weight_decay',
                           'use_l2_penalty',
                           'loss_fn']

        missing_params = []
        for param in required_params:
            if param not in self.unlearn_params:
                if self.unlearner_name == 'scrub' and param == 'epochs':
                    continue
                missing_params.append(param)

        if missing_params:
            raise ValueError('Missing required parameters for unlearning: '
                             f'{', '.join(missing_params)}')

        # Add specific parameters based on unlearner type
        if self.unlearner_name == 'neggradplus':
            try:
                self.unlearn_params['beta']
            except KeyError:
                raise ValueError('Missing required parameter beta'
                                 ' for NegGradPlus unlearner')

        elif self.unlearner_name == 'scrub':
            # Check for required SCRUB-specific parameters
            required_scrub_params = ['min_epochs', 'max_epochs',
                                     'alpha', 'gamma']
            missing_scrub_params = []
            for param in required_scrub_params:
                if param not in self.unlearn_params:
                    missing_scrub_params.append(param)
            if missing_scrub_params:
                raise ValueError('Missing required params for SCRUB unlearner:'
                                 f' {', '.join(missing_scrub_params)}')

        elif self.unlearner_name in ['euk', 'cfk']:
            # Check for k parameter
            try:
                self.unlearn_params['k']
            except KeyError:
                raise ValueError(f'Missing required parameter k for '
                                 f'{self.unlearner_name.upper()} unlearner')
            if self.unlearner_name == 'euk':
                # Check for EUk-specific parameters
                if 'reinit_method' not in self.unlearn_params:
                    raise ValueError('Missing required parameter '
                                     'reinit_method for EUk unlearner')

    # def _validate_dataset_params(self):
    #     assert 
