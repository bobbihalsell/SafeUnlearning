import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_costs(gradients,
                         input_gradient,
                         cost_fn='l2',
                         indices='def',
                         weights='equal'
                         ):
    """
    Calculate the reconstruction costs based on the specified cost function.
    Args:
        gradients (list): List of gradients for the reconstruction.
        input_gradient (list): List of input gradients.
        cost_fn (str): Distance function to use for reconstruction.
            Options: 'l2', 'l1', 'max', 'sim', 'simlocal'.
            l2: L2 distance
            l1: L1 distance
            max: max distance
            sim: cosine similarity
            simlocal: local cosine similarity
        indices (str): Layers to use in calculating reconstruction cost.
            Options: 'def', 'batch', 'topk-1', 'top10', 'top50', 'first',
            'first4','first5', 'first10', 'first50', 'last5', 'last10',
            'last50'.
            def: all layers
            batch: random 8 layers
            topk-1: top 4 layers
            top10/top50: top 10/50 layers
            first/first4/first5/first10/first50: first 4/5/10/50 layers
            last5/last10/last50: last 5/10/50 layers
        weights (str): Weighting of the layers in the reconstruction process.
            Options: 'linear', 'exp', 'equal'.
            linear: linearly decreasing weights
            exp: exponentially decreasing weights
            equal: equal weights

    Returns:
        torch.Tensor: Reconstruction costs.
    """
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(
            torch.stack([p.norm() for p in input_gradient], dim=0),
            4
        )
    elif indices == 'top10':
        _, indices = torch.topk(
            torch.stack([p.norm() for p in input_gradient], dim=0),
            10
        )
    elif indices == 'top50':
        _, indices = torch.topk(
            torch.stack([p.norm() for p in input_gradient], dim=0),
            50
        )
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError("Invalid indices option. Choose from ['def', 'batch',"
                         "'topk-1', 'top10', 'top50', 'first', 'first4', "
                         "'first5', 'first10', 'first50', 'last5', 'last10',"
                         " 'last50']")

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype,
                               device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype,
                               device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    elif weights == 'equal':
        weights = input_gradient[0].new_ones(len(input_gradient))
    else:
        raise ValueError('Weights must be one of [linear, exp, equal]')

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += (
                    1 - torch.nn.functional.cosine_similarity(
                        trial_gradient[i].flatten(),
                        input_gradient[i].flatten(),
                        dim=0,
                        eps=1e-10
                    )
                ) * weights[i]
            else:
                raise ValueError('Cost function must be one of [l2, l1, max,'
                                 ' sim, simlocal]')              
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs

    return total_costs / len(gradients)


def total_variation(x):
    """
    Anisotropic total variation regularization.

    This function computes the total variation of an image tensor.
    The total variation is a measure of the smoothness of the image,
    and it is defined as the sum of the absolute differences
    between adjacent pixels in the x and y directions.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Total variation of the input tensor.
    """
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    return dx + dy


class DistillKL(nn.Module):
    """
    Kullback-Leibler Divergence Loss for Distillation.
    This class implements the Kullback-Leibler divergence loss function
    for distillation, as described in the paper "Distilling the Knowledge
    in a Neural Network"
    https://arxiv.org/pdf/1503.02531).

    Higher temperatures lead to softer probability distributions.
    Args:
        T (float): Temperature parameter for scaling the logits.
    """

    def __init__(self,
                 T):
        """
        Initialize the Kullback-Leibler divergence loss.
        Args:
            T (float): Temperature parameter for scaling the logits.
        """
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self,
                y_s,
                y_t):
        """
        Forward pass for the Kullback-Leibler divergence loss.
        Args:
            y_s (torch.Tensor): Model y logits.
            y_t (torch.Tensor): Model t logits.
        Returns:
            torch.Tensor: Kullback-Leibler divergence loss.
        """
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(
            p_s, p_t, reduction='sum'
        ) * (self.T**2) / y_s.shape[0]

        return loss