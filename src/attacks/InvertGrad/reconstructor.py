# Adapted from JonasGeiping/invertinggradients, licensed under MIT
# Source: https://github.com/JonasGeiping/invertinggradients/

from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import torch
import torchvision.transforms as transforms

from attacks.InvertGrad.medianfilt import MedianPool2d
from attacks.InvertGrad.reconstruction_cost import (reconstruction_costs,
                                                    total_variation,
                                                    DistillKL)
from utils import set_seed

# === Valid Configuration Options ===
OPTIONS_BOOL = [True, False]

OPTIONS_FILTERS = [None, 'median']
OPTIONS_COST_FNS = ['sim', 'l2', 'l1']
OPTIONS_INDICES = [
    'def', 'batch', 'topk-1', 'top10', 'top50',
    'first', 'first4', 'first5', 'first10', 'first50',
    'last5', 'last10', 'last50'
]
OPTIONS_WEIGHTS = ['equal', 'none']
OPTIONS_SCORING = ['loss', 'tv', 'pixelmean', 'pixelmedian']
OPTIONS_OPTIMIZERS = ['adam', 'sgd', 'LBFGS', 'adamw']


@dataclass
class InvertGradConfig:
    """
    Configuration for the Invert Gradient Reconstructor.
    Attributes:
        grad_diff_lr (float): Learning rate for gradient difference.
        signed (bool): Whether to use signed gradients.
        boxed (bool): Whether to box the input data.
        cost_fn (str): Distance function to use for reconstruction.
        indices (str): Layers to use in calculating reconstruction cost.
        weights (str): Weighting of the layers in the reconstruction cost.
        optim (str): Optimizer to use for reconstruction.
        num_runs (int): Number of runs for reconstruction. The best result
            will be chosen based on the scoring choice.
        recon_iterations (int): Number of iterations in a reconstruction.
        total_variation (float): Total variation regularization parameter.
        init (str): Initialization method for the input image.
        lr_decay (bool): Whether to use learning rate decay.
        scoring_choice (str): Scoring method to use for reconstruction.
        eval (bool): Whether to use evaluation mode for the model.
        filter (bool): Whether to apply median filtering to the reconstructed
          image every 500 iterations.

    """
    grad_diff_lr: float = 1e-4
    signed: bool = False
    boxed: bool = True
    cost_fn: str = 'sim'
    indices: str = 'def'
    weights: str = 'equal'
    optim: str = 'adam'
    num_runs: int = 1
    recon_iterations: int = 4800
    total_variation: float = 1e-1
    init: str = 'randn'
    lr_decay: bool = True
    scoring_choice: str = 'loss'
    eval: bool = True
    filter: Optional[str] = None

    def __post_init__(self):
        # Force conversion to float if the value is passed as a string
        if not isinstance(self.total_variation, float):
            self.total_variation = float(self.total_variation)


class InvertGradReconstructor():
    """
    Implements a gradient inversion attack as introduced in
    "Inverting Gradients -- How Easy Is It to Break Privacy
    in Federated Learning?"(https://arxiv.org/abs/2003.14053v1)

    This class reconstructs input images from gradients by optimizing
    dummy inputs via gradient descent. The optimization process attempts
    to match the gradients produced by the reconstructed input with those
    of the original unlearned image.

    This adaptation of the original algorithm takes in the gradient difference
    between the original model and the unlearned model.The loss function used
    for optimization is a combination of cross-entropy loss and
    Kullback-Leibler divergence, which helps to ensure that the reconstructed
    image is similar to the original image in terms of both pixel values and
    class probabilities.
    """

    def __init__(self,
                 device,
                 original_model,
                 unlearned_model,
                 config: InvertGradConfig = InvertGradConfig(),
                 seed=42
                 ):
        """
        Initialize with algorithm setup.

        Args:
            device (torch.device): Device to run the reconstruction on (CPU
                or GPU).
            original_model (torch.nn.Module): The original model before
                unlearning.
            unlearned_model (torch.nn.Module): The model after unlearning.
            config (InvertGradConfig): Configuration object for the
                reconstruction process.
        """
        self.config = config
        print(f"filter: {self.config.filter}")

        # Validate the configuration parameters
        self._validate_config()

        self.original_model = original_model
        self.unlearned_model = unlearned_model
        self.device = device

        self.loss_fn_ce = torch.nn.CrossEntropyLoss()
        self.loss_fn = DistillKL(2)
        self.input_gradient = self._gradient_difference(config.grad_diff_lr)
        set_seed(seed)

    def reconstruct(self,
                    labels,
                    num_images,
                    image_size=[3, 32, 32],
                    image_mean=[0.5, 0.5, 0.5],
                    image_std=[0.5, 0.5, 0.5],
                    lr=0.1,
                    verbose=True):
        """
        Reconstruct image from gradient. The reconstruction process runs for a
        specified number of trials and returns the best result based on the
        scoring choice.
        Args:
            labels (torch.Tensor): Labels for the images to be reconstructed.
            num_images (int): Number of images to reconstruct.
            image_size (list): Size of the images to be reconstructed.
            image_mean (list): Mean values for normalization.
            image_std (list): Standard deviation values for normalization.
            lr (float): Learning rate for optimization.
            verbose (bool): Whether to print progress messages.
        Returns:
            torch.Tensor: Reconstructed images.
            float: Score of the reconstructed images.
        """

        self.image_size = tuple(int(x) for x in image_size)
        self.dm = image_mean[0]
        self.ds = image_std[0]
        self.lr = lr
        self.verbose = verbose

        if labels is not None:
            labels = torch.as_tensor(labels, device=self.device)
            self.num_images = labels.shape[0]
        else:
            self.num_images = num_images

        if self.num_images > 1 and self.config.scoring_choice in ['pixelmean',
                                                                  'pixelmedian']:
            raise ValueError('Pixel mean/median scoring choice can only be'
                             ' performed on one image.')

        self.normalizer = transforms.Normalize(
            mean=image_mean,
            std=image_std
        )

        if self.config.eval:
            self.original_model.eval()

        # initalize input data
        input_data = self.input_gradient
        stats = defaultdict(list)
        x = self._init_images()
        scores = torch.zeros(self.config.num_runs, device=self.device)

        if labels is None:
            self.reconstruct_label = True

            def loss_fn(pred, labels):
                labels = torch.nn.functional.softmax(labels, dim=-1)
                return torch.mean(torch.sum(-labels *
                                            torch.nn.functional.
                                            log_softmax(pred, dim=-1), 1))
            self.loss_fn_ce = loss_fn
        else:
            self.reconstruct_label = False

        try:
            for trial in range(self.config.num_runs):
                x_trial, labels = self._run_trial(x[trial].to(self.device),
                                                  input_data, labels)

                # Store the trial result
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial.to(self.device)

        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result
        if self.config.scoring_choice in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels,
                                                    input_data, stats)
        else:
            scores = scores[torch.isfinite(scores)]  
            optimal_index = torch.argmin(scores)
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

            if self.verbose:
                print(f'Optimal result score: {scores[optimal_index]:2.4f}')

        return x_optimal.detach(), stats['opt']

    def _gradient_difference(self,
                             grad_lr=1e-4):
        """
        Calculate the gradient difference between the original and
        unlearned models.
        Args:
            grad_lr (float): Learning rate for gradient difference.
        Returns:
            list: List of gradient differences for each parameter.
        """

        param_old = [p.clone().detach() for p in
                     self.original_model.parameters()]
        param_new = [p.clone().detach() for p in
                     self.unlearned_model.parameters()]

        return [(new.detach() - old.detach()) / grad_lr for old, new
                in zip(param_old, param_new)]

    def _init_images(self):
        """
        Initialize the input images based on the specified
        initialization method.
        Returns:
            torch.Tensor: Initialized input images.
        """
        if self.config.init == 'randn':
            return torch.randn((self.config.num_runs, self.num_images,
                                *self.image_size), device=self.device)
        elif self.config.init == 'rand':
            return (torch.rand((self.config.num_runs, self.num_images,
                                *self.image_size), device=self.device) - 0.5) * 2
        elif self.config.init == 'zeros':
            return torch.zeros((self.config.num_runs, self.num_images,
                                *self.image_size), device=self.device)
        else:
            raise ValueError("Invalid initialization method. Choose from ['randn', 'rand', 'zeros'].")

    def _run_trial(self,
                   x_trial,
                   input_data,
                   labels):
        """
        Run a single trial of the reconstruction process.
        Args:
            x_trial (torch.Tensor): Input image for the trial.
            input_data (list): List of input gradients.
            labels (torch.Tensor): Labels for the images to be reconstructed.
        Returns:
            torch.Tensor: Reconstructed image for the trial.
            torch.Tensor: Labels for the reconstructed image.
        """

        x_trial.requires_grad = True

        if self.reconstruct_label:
            output_test = self.original_model(x_trial)

            # Create trainable label vector to reconstruct
            labels = (
                torch.randn(output_test.shape[1])
                .to(self.device)
                .requires_grad_(True)
            )

        # Set up the optimizer with the labels
        optimizer = self._set_optimizer(x_trial, labels)

        recon_iterations = self.config.recon_iterations
        if self.config.lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(recon_iterations / 2.667),
                    int(recon_iterations / 1.6),
                    int(recon_iterations / 1.142),
                ],
                gamma=0.5,
            )
        try:
            # Take a copy of the model to compute KL loss
            self.model_copy = deepcopy(self.original_model)

            for iteration in range(recon_iterations):

                if self.config.boxed:
                    x_trial.data = torch.clamp(x_trial.data, 0, 1)

                closure = self._gradient_closure(optimizer, x_trial,
                                                 input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config.lr_decay:
                    scheduler.step()

                    if (iteration + 1 == recon_iterations or
                       iteration % 500 == 0) and self.verbose:
                        print(f'It: {iteration}. Rec. loss:{rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config.filter == 'median':
                            x_trial.data = MedianPool2d(
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    same=False)(x_trial)
                        else:
                            pass

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self,
                          optimizer,
                          x_trial,
                          input_gradient,
                          label):
        """
        Closure function for the optimizer. This function computes the loss and
        gradients for the current trial.
        Args:
            optimizer (torch.optim.Optimizer): Optimizer for the trial.
            x_trial (torch.Tensor): Input image for the trial.
            input_gradient (list): List of input gradients.
            label (torch.Tensor): Labels for the images to be reconstructed.
        Returns:
            function: Closure function for the optimizer.
        """

        def closure():
            # Compute the loss and gradients
            optimizer.zero_grad()
            self.original_model.zero_grad()

            loss_ce = self.loss_fn_ce(
                self.original_model(self.normalizer(x_trial)),
                label
            )

            loss_kl = self.loss_fn(
                self.original_model(self.normalizer(x_trial)),
                self.model_copy(self.normalizer(x_trial))
            )
            # Combine the losses
            loss = loss_ce + loss_kl

            gradient = torch.autograd.grad(
                                        loss,
                                        self.original_model.parameters(),
                                        create_graph=True
                                        )
            rec_loss = reconstruction_costs(
                                        [gradient], input_gradient,
                                        cost_fn=self.config.cost_fn,
                                        indices=self.config.indices,
                                        weights=self.config.weights
                                        )
          
            # Add total variation regularization if specified
            if self.config.total_variation > 0:
                rec_loss += self.config.total_variation * total_variation(x_trial)

            rec_loss.backward()

            if self.config.signed:
                x_trial.grad.sign_()

            return rec_loss
        return closure

    def _score_trial(self,
                     x_trial,
                     input_gradient,
                     label):
        """
        Score the trial based on the specified scoring choice. Uses only the 
        cross-entropy loss. Returns 0 for pixelmean and pixelmedian.
        Args:
            x_trial (torch.Tensor): Input image for the trial.
            input_gradient (list): List of input gradients.
            label (torch.Tensor): Labels for the images to be reconstructed.
        Returns:
            float: Score of the trial.
        """

        if self.config.scoring_choice == 'loss':
            self.original_model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn_ce(self.original_model((x_trial)), label)

            gradient = torch.autograd.grad(
                                        loss,
                                        self.original_model.parameters(),
                                        create_graph=False
                                        )

            return reconstruction_costs(
                                    [gradient], input_gradient,
                                    cost_fn=self.config.cost_fn,
                                    indices=self.config.indices,
                                    weights=self.config.weights)
        elif self.config.scoring_choice == 'tv':
            return total_variation(x_trial)
        elif self.config.scoring_choice in ['pixelmean', 'pixelmedian']:
            return 0.0

    def _average_trials(self, x, labels, input_data, stats):
        """
        Average the trials and compute the optimal result based on the
        specified scoring choice.
        
        Args:
            x (torch.Tensor): Input images for the trials.
            labels (torch.Tensor): Labels for the images to be reconstructed.
            input_data (list): List of input gradients.
            stats (dict): Dictionary to store statistics.
        Returns:
            torch.Tensor: Reconstruction images from all runs and combined
                reconstructed image.
            dict: Dictionary of statistics.
        """
        if self.config.scoring_choice == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
            x = x.squeeze(1)
            # Combine the original image with the optimal result
            x_with_opt = torch.vstack([x, x_optimal])

        elif self.config.scoring_choice == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)
            x = x.squeeze(1)
            # Combine the original image with the optimal result
            x_with_opt = torch.vstack([x, x_optimal])

        self.original_model.zero_grad()
        if self.reconstruct_label:
            labels = self.original_model(x_optimal).softmax(dim=1)

        # Calculate the loss for the optimal result and score it
        loss = self.loss_fn_ce(self.original_model(x_optimal), labels)

        gradient = torch.autograd.grad(loss, self.original_model.parameters(),
                                       create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config.cost_fn,
                                            indices=self.config.indices,
                                            weights=self.config.weights)
        if self.verbose:
            print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_with_opt, stats

    def _set_optimizer(self,
                       x_trial,
                       labels=None):
        """
        Set up the optimizer for the trial.
        Args:
            x_trial (torch.Tensor): Input image for the trial.
        Returns:
            torch.optim.Optimizer: Optimizer for the trial.
        """
        params = [x_trial]
        if self.reconstruct_label:
            params.append(labels)

        if self.config.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.lr)
        elif self.config.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.lr,
                                        momentum=0.9, nesterov=True)
        elif self.config.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS(params)
        elif self.config.optim == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optim}")
     
        return optimizer
    
    def _validate_config(self):
        """
            Validate the configuration parameters.
            Raises:
                ValueError: If any of the configuration parameters are invalid.
        """
        if self.config.grad_diff_lr <= 0:
            raise ValueError("Gradient difference learning rate must be positive.")

        if self.config.recon_iterations <= 0:
            raise ValueError("Reconstruction iterations must be positive.")

        if self.config.total_variation < 0:
            raise ValueError("Total variation must be non-negative.")

        if self.config.lr_decay not in OPTIONS_BOOL:
            raise ValueError("Learning rate decay must be a boolean value.")

        if self.config.signed not in OPTIONS_BOOL:
            raise ValueError("Signed gradients must be a boolean value.")

        if self.config.boxed not in OPTIONS_BOOL:
            raise ValueError("Boxed input data must be a boolean value.")

        if self.config.eval not in OPTIONS_BOOL:
            raise ValueError("Evaluation mode must be a boolean value.")

        if self.config.filter not in OPTIONS_FILTERS:
            raise ValueError(f"Filter must be one of: {OPTIONS_FILTERS}")

        if self.config.cost_fn not in OPTIONS_COST_FNS:
            raise ValueError(f"Invalid cost function. Choose from: {OPTIONS_COST_FNS}")

        if self.config.indices not in OPTIONS_INDICES:
            raise ValueError(f"Invalid indices. Choose from: {OPTIONS_INDICES}")

        if self.config.weights not in OPTIONS_WEIGHTS:
            raise ValueError(f"Invalid weights. Choose from: {OPTIONS_WEIGHTS}")

        if self.config.scoring_choice not in OPTIONS_SCORING:
            raise ValueError(f"Invalid scoring choice. Choose from: {OPTIONS_SCORING}")

        if self.config.optim not in OPTIONS_OPTIMIZERS:
            raise ValueError(f"Invalid optimizer. Choose from: {OPTIONS_OPTIMIZERS}")
