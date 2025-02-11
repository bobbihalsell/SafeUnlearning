""" Method from https://arxiv.org/abs/2404.03233 by Hu et. al in a paper."""
import torch


DEFAULT_CONFIG = dict(boxed=True,
                      cost_fn='cossim',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',  # How you want to initialize an image
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')


class UIAttack:
    """ Perform an Unlearning Inversion Attack given 2 models."""
    def __init__(self, original_model, unlearned_model, config:dict):
        if not isinstance(original_model, torch.nn.Module):
            raise TypeError("original_model must be an instance of torch.nn.Module")
        if not isinstance(unlearned_model, torch.nn.Module):
            raise TypeError("unlearned_model must be an instance of torch.nn.Module")

        self.original_model = original_model
        self.unlearned_model = unlearned_model

        self._validate_models()
        self.config = self._validate_config(config)

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

    def _validate_config(self, config):  # Just to initialize missing keys to the default and that there are no extra fields provided by the user that the code cannot support.
        for key in DEFAULT_CONFIG.keys():
            if config.get(key) is None:
                config[key] = DEFAULT_CONFIG[key]
        for key in config.keys():
            if DEFAULT_CONFIG.get(key) is None:  # Quite a nice logic though for software robustness, worth noting and maybe taking awawy
                raise ValueError(f'Deprecated key in config dict: {key}!')
        return config

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


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
