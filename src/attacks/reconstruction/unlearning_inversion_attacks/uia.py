""" Method from https://arxiv.org/abs/2404.03233 by Hu et. al in a paper."""
import torch


class UIAttack:
    """ Perform an Unlearning Inversion Attack given 2 models."""
    def __init__(self, original_model, unlearned_model):
        if not isinstance(original_model, torch.nn.Module):
            raise TypeError("original_model must be an instance of torch.nn.Module")
        if not isinstance(unlearned_model, torch.nn.Module):
            raise TypeError("unlearned_model must be an instance of torch.nn.Module")

        self.original_model = original_model
        self.unlearned_model = unlearned_model

        self._validate_models()

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

    def reconstruct(self):
        # Run experiments to reconstruct features given parameter difference
        pass
