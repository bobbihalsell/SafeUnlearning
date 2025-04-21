import os
import json 
import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import MissingMandatoryValue
from attacks.GLiR.glir_utils import calculate_metrics
from attacks.GLiR.gradient_attack import GLiR
from attacks.GLiR.datahandler import DataHandler
from attacks.GLiR.config_validation import GLiRValidator
from importmodel import ImportModel
from utils import set_seed, setup_device


class GLiRApp(GLiRValidator):
    """
    Implement the white-box mia described in 
    https://arxiv.org/abs/2306.07273
    """
    def __init__(self, config: DictConfig):
        # Perform input validation first
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)

        self.device = setup_device()
        print(f'Using device: {self.device}')
        self.seed = config['seed']
        set_seed(self.seed)

        # Output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def initialise_data(self):
        self.data = DataHandler(
            self.dataset_name,
            self.dataset_save_dir,
            self.background_ratio,
            self.test_size
        )
        self.background, _ = self.data.prepare_background_points()
        self.querypoints = self.data.prepare_querypoints()
        return

    def initialise_attack(self):
        original_import = ImportModel(self.load_method,
                                      self.model_name, 
                                      self.num_classes,
                                      self.init_path, 
                                      self.original_model_ckpt_path,
                                      self.model_kwargs, 
                                      )
        self.original_model = original_import.model
        unlearned_import = ImportModel(self.load_method,
                                       self.model_name, 
                                       self.num_classes,
                                       self.init_path, 
                                       self.original_model_ckpt_path,
                                       self.model_kwargs, 
                                       )
        self.unlearned_model = unlearned_import.model

        self.attack = GLiR(
                        model_before=self.original_model, 
                        model_after=self.unlearned_model,
                        num_params=self.num_params, 
                        small_var_lim=self.small_var_lim,
                        device=self.device
                        )  
        
    def initialise_baseline(self):
        self.attack.establish_baseline(self.background)
        return

    def perform_attack(self):
        """
        Perform the attack on the query points and save results.

        Args: 
            savepath: Directory to save plots and results
            threshold: The threshold to classify predictions based on 
                attack scores
        """
        # Ensure the save path exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Unpack the labels and data points 
        # labels = [z for _, z in self.querypoints]
        # data = [d for d, _ in self.querypoints]
        data, labels = zip(*self.querypoints)
        data = list(data)
        labels = list(labels)

        # Classify points using the attack method
        print('classifying...')
        classifications, p_vals, test_statistics = self.attack.classify_set(
            data, 
            self.threshold,
            teststatistic=True
            )

        # Evaluate the attack
        print('evaluating...')
        metrics = calculate_metrics(classifications, labels)
        one_take_pvals = [1 - p for p in p_vals]

        # Save the ROC curve and GLIR distribution plot
        print('visualising...')
        self.attack.plot_roc_curve(one_take_pvals, labels,
                                   savepath=self.output_dir)
        self.attack.plot_glir_distribution(p_values=p_vals, 
                                           test_statistics=test_statistics,
                                           labels=labels,
                                           savepath=self.output_dir, 
                                           alpha=self.threshold)

        # Save metrics as JSON
        metrics_path = os.path.join(self.output_dir, "attack_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            print(f"Metrics saved at: {metrics_path}")

        # Print results to the console
        print(f"Accuracy: {metrics['Accuracy']}")
        print(f"Precision: {metrics['Precision']}")
        print(f"Recall: {metrics['Recall']}")

    def run(self):
        print("Peparing query points and background points...")
        self.initialise_data()
        
        # Load models
        print("Loading models and initialising attack...")
        self.initialise_attack()

        print("Creating the baseline...")
        self.initialise_baseline()

        print("Performing the attack...")
        self.perform_attack()

        print("Attack complete")
        return 


@hydra.main(version_base=None,
            config_path="config",
            config_name="config")
def main(cfg: DictConfig):
    # Print the config for the user first
    print('============ Run Configuration ============')
    print(OmegaConf.to_yaml(cfg))
    print('============================================')
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            'Missing the following required arguments in the configuration: '
            f'{missing_keys}. \n'
            'Hint: python file.py key=value sets the appropriate value.')
    app = GLiRApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
