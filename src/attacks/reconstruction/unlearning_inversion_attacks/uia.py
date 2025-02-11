""" Method from https://arxiv.org/abs/2404.03233 by Hu et. al in a paper."""
import torch
from utils import total_variation
from torchvision import transforms
from tqdm import trange


DEFAULT_CONFIG = dict(boxed=True,
                      cost_fn='cossim',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',  # How you want to initialize an image
                      filter='none',
                      lr_decay=True
                      )
# cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
# cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
# cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
# cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
# stl10_mean = [0.44671064615249634, 0.4398098886013031, 0.4066464304924011]
# stl10_std = [0.26034098863601685, 0.2565772831439972, 0.2712673842906952]


class UIAttack:
    """ Perform an Unlearning Inversion Attack given 2 models."""
    def __init__(self, original_model, unlearned_model,
                 config: dict, mean_std=(0.0, 1.0), num_images=1):
        if not isinstance(original_model, torch.nn.Module):
            raise TypeError("original_model must be an instance of torch.nn.Module")
        if not isinstance(unlearned_model, torch.nn.Module):
            raise TypeError("unlearned_model must be an instance of torch.nn.Module")

        self.original_model = original_model
        self.unlearned_model = unlearned_model

        self._validate_models()
        self.config = self._validate_config(config)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.mean_std = mean_std
        self.num_images = num_images
        # Dict of torch device used by the current system
        self.setup = dict(device=next(self.original_model.parameters()).device,
                          dtype=next(self.original_model.parameters()).dtype)

    def _validate_models(self):
        """Ensure both models have the same number and shape of parameters."""
        original_params = list(self.original_model.state_dict().keys())
        unlearned_params = list(self.unlearned_model.state_dict().keys())

        if original_params != unlearned_params:
            raise ValueError("""Models have different parameter structures.
                             Support for unlearning methods which change model
                             architecture is not yet enabled.
                             """)

        for key in original_params:
            if (self.original_model.state_dict()[key].shape !=
                    self.unlearned_model.state_dict()[key].shape):
                raise ValueError(f"Shape mismatch for parameter: {key}")

    def _validate_config(self, config):
        for key in DEFAULT_CONFIG.keys():
            if config.get(key) is None:
                config[key] = DEFAULT_CONFIG[key]
        for key in config.keys():
            if DEFAULT_CONFIG.get(key) is None:
                raise ValueError(f'Deprecated key in config dict: {key}!')
        return config

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'],
                                self.num_images,
                                *img_shape), **self.setup)  # **self.setup tells the tensor to use GPU/CPU and a specific dtype

        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def get_parameter_difference(self):
        """ Get the gradient (parameter diff) between the 2 models."""
        param_differences = {
            name: param_unlearned - param_original
            for (name, param_original), (_, param_unlearned) in zip(
                self.original_model.state_dict().items(),
                self.unlearned_model.state_dict().items()
            )
        }

        return list(param_differences.values())

    def _reconstruction_costs(self, gradients, input_gradient, cost_fn='l2'):
        """ Compute gradient cost between gradients and reference gradients.
        Lower is better.

        Args:
            gradients (list): A list of gradients.
            input_gradient (list): A list of parameter-wise differences
            between an original and an unlearned model.
            cost_fn (str): (Default) L2 distance.
                Options: 'l2', 'l1', 'max', 'cossim', 'dotsim'

        Returns:
            reconstruction_cost (float): 
                A distance metric between the gradient pairs.
        """
        if len(gradients) != len(input_gradient):
            raise Exception('Input gradients of unequal length.')

        total_costs = 0
        for trial_gradient in gradients:
            pnorm = [0, 0]
            costs = 0
            for i in range(len(gradients)):
                if cost_fn == 'l2':
                    costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()
                elif cost_fn == 'l1':
                    costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum()
                elif cost_fn == 'max':
                    costs += ((trial_gradient[i] - input_gradient[i]).abs()).max()
                elif cost_fn == 'dotsim':
                    costs -= (trial_gradient[i] * input_gradient[i]).sum()
                    pnorm[0] += trial_gradient[i].pow(2).sum()
                    pnorm[1] += input_gradient[i].pow(2).sum()
                elif cost_fn == 'cossim':
                    costs += 1 - torch.nn.functional.cosine_similarity(
                        trial_gradient[i].flatten(),
                        input_gradient[i].flatten(),
                        dim=0,
                        eps=1e-10
                        )
                else:
                    raise ValueError('Unsupported cost function.')
            if cost_fn == 'dot':
                costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

            # Accumulate final costs
            total_costs += costs
        # Mean over all trial gradients
        return total_costs / len(gradients)

    def _gradient_closure(self, normalizer, optimizer,
                          x_trial, input_gradient, label,
                          cost_fn='l2'):
        def closure():
            optimizer.zero_grad()
            self.original_model.zero_grad()
            # Compute the loss w.r.t inputs
            loss = self.loss_fn(self.original_model(normalizer(x_trial)),
                                label)
            gradient = torch.autograd.grad(loss,
                                           self.original_model.parameters(),
                                           create_graph=True)

            rec_loss = self._reconstruction_costs([gradient],
                                                  input_gradient,
                                                  cost_fn=cost_fn)
            if self.config['total_variation'] > 0:
                rec_loss += (self.config['total_variation'] *
                             total_variation(x_trial))
            # Compute the gradients with respect to each of the x_trial params
            rec_loss.backward()
            return rec_loss

        return closure

    def reconstruct(self, X_forget, y_forget, img_shape=(3, 32, 32),
                    dryrun=False, eval=True, tol=None):
        """Reconstruct image."""
        if eval:
            self.original_model.eval()
            self.unlearned_model.eval()

        param_diff = self.get_parameter_difference()

        stats = {}
        # Initialize tensor of randomly initialized images
        x = self._init_images(img_shape)
        history_list = []
        scores = torch.zeros(self.config['restarts'])

        if y_forget is None:  # Label reconstruction
            self.reconstruct_label = True

            def loss_fn(pred, y_forget):  # NLL loss
                y_forget = torch.nn.functional.softmax(y_forget, dim=-1)
                return torch.mean(torch.sum(- y_forget * torch.nn.functional.log_softmax(pred, dim=-1), 1))
            self.loss_fn = loss_fn
        else:
            assert y_forget.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels, history = self._run_trial(x[trial],
                                                           param_diff,
                                                           X_forget,
                                                           y_forget,
                                                           dryrun=dryrun)
                # Score how well gradient of reconstructed x_trial images match param_diff.
                scores[trial] = self._score_trial(x_trial,
                                                  param_diff,
                                                  labels)
                # Update x[trial] with the checkpoint from this restart.
                x[trial] = x_trial
                history_list.append(history)
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        print('Choosing optimal result ...')
        scores = scores[torch.isfinite(scores)]
        # Get x with minimum cost between the gradient and param difff
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        stats['opt'] = scores
        x_optimal = x[optimal_index]

        return x, stats, history_list, x_optimal

    def _run_trial(self, x_trial, input_gradient, data, labels, dryrun=False):
        x_trial.requires_grad = True  # So that x can be modified
        history = []

        if self.reconstruct_label:
            output_test = self.original_model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels],
                                             lr=self.config['lr'])
            elif self.config['optim'] == 'adamw':
                optimizer = torch.optim.AdamW([x_trial, labels],
                                              lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':
                optimizer = torch.optim.SGD([x_trial, labels],
                                            lr=self.config['lr'],
                                            momentum=0.9,
                                            nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial,
                                               labels])
            else:
                raise ValueError("Only 'adam', 'adamw', 'sgd' and 'LBFGS' optimizers supported.")
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial],
                                             lr=self.config['lr'])
            elif self.config['optim'] == 'adamw':
                optimizer = torch.optim.AdamW([x_trial],
                                              lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':
                optimizer = torch.optim.SGD([x_trial],
                                            lr=self.config['lr'],
                                            momentum=0.9,
                                            nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        normalizer = transforms.Normalize(dm.tolist(), ds.tolist())
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,
                                                                         max_iterations // 1.142], gamma=0.5)
        try:
            tqdm_range = trange(1, max_iterations + 1, desc='Loss', leave=True)
            for iteration in tqdm_range:
                # Project into image space
                if self.config['boxed']:
                    x_trial.data = torch.clamp(x_trial.data, 0, 1)
                # Compute the gradient of x_trial w.r.t the original model
                closure = self._gradient_closure(normalizer,
                                                 optimizer,
                                                 x_trial,
                                                 input_gradient,
                                                 labels)
                # Compute loss, backpropagate and update x_trial.
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    if (iteration == max_iterations) or iteration % 100 == 0:
                        history.append(x_trial.detach().cpu().clone())

                tqdm_range.set_description(f'Loss: {rec_loss.item():2.4f}. MSE: {(x_trial.data - data).pow(2).mean().item()}')
                tqdm_range.refresh()
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels, history

    def _score_trial(self, x_trial, input_gradient, label):
        """ Return a score for a reconstruction attempt x_trial.

        Score reconstructed x_trial inputs using similarity between
        gradient of x_trials w.r.t original model and the parameter difference
        as a proxy for reconstruction quality.

        Args:
            x_trial (tensor): Reconstructed images over the trials.
            input_gradient (list): Parameter difference between 2 models
            label: x_trial labels

        Returns:
            The reconstruction costs for x_trial.
        """
        self.original_model.zero_grad()
        x_trial.grad = None
        loss = self.loss_fn(self.original_model(x_trial), label)
        gradient = torch.autograd.grad(loss,
                                       self.original_model.parameters(),
                                       create_graph=False)

        return self._reconstruction_costs([gradient],
                                          input_gradient,
                                          cost_fn=self.config['cost_fn'])
