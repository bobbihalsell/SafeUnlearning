"""Mechanisms for image reconstruction from parameter gradients."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from InverseGrad.modules import MetaMonkey

from InverseGrad.metrics import total_variation as TV
from InverseGrad.medianfilt import MedianPool2d

from copy import deepcopy

@dataclass
class InverseGradConfig:
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
    filter: str = 'none'

    def __post_init__(self): #TODO: this can be enforced by hydra
    # Force conversion to float if the value is passed as a string
        if not isinstance(self.total_variation, float):
            self.total_variation = float(self.total_variation)


class InverseGradReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, device, original_model, unlearned_model,  
                 config: InverseGradConfig = InverseGradConfig()):
        """Initialize with algorithm setup."""
        self.config = config
        self.original_model = original_model
        self.unlearned_model = unlearned_model
        self.setup = device

        self.loss_fn_ce = torch.nn.CrossEntropyLoss()
        self.loss_fn = DistillKL(2)
        self.iDLG = True
        self.input_gradient = self._gradient_difference(config.grad_diff_lr)



    def _gradient_difference(
            self,
            grad_lr = 1e-4):
            param_old = [p.clone().detach() for p in self.original_model.parameters()]
            param_new = [p.clone().detach() for p in self.unlearned_model.parameters()]
            return [(new.detach() - old.detach()) / grad_lr for old, new in zip(param_old, param_new)]

    def reconstruct(self, labels, image_size=[3, 32, 32], 
                    num_images=1, image_mean = [0.5, 0.5, 0.5],
                    image_std = [0.5, 0.5, 0.5],
                    lr = 0.1, verbose = True):
        """Reconstruct image from gradient."""

        self.num_images = num_images
        self.image_size = tuple(int(x) for x in image_size)
        self.dm = image_mean[0]
        self.ds = image_std[0]
        self.lr = lr
        self.verbose = verbose
        labels = torch.tensor(labels, device=self.setup)
        
        
        if eval:
            self.original_model.eval()

        input_data = self.input_gradient
        stats = defaultdict(list)
        x = self._init_images()
        scores = torch.zeros(self.config.num_runs, device=self.setup)

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn_ce = loss_fn
                self.loss_fn = DistillKL(2)
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config.num_runs):
                x_trial, labels = self._run_trial(x[trial].to(self.setup), input_data, labels)
                # Finalize
                print(f'Finalizing trial {trial} ...')
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                print(f'Score: {scores[trial]:2.4f}')
                x[trial] = x_trial.to(self.setup)


        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config.scoring_choice in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        return x_optimal.detach(), stats

    def _init_images(self):
        if self.config.init == 'randn':
            return torch.randn((self.config.num_runs, self.num_images, *self.image_size), device = self.setup)
        elif self.config.init == 'rand':
            return (torch.rand((self.config.num_runs, self.num_images, *self.image_size), device =self.setup) - 0.5) * 2
        elif self.config.init == 'zeros':
            return torch.zeros((self.config.num_runs, self.num_images, *self.image_size), device = self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.original_model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config.optim == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.lr)
            elif self.config.optim == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr = self.lr, momentum=0.9, nesterov=True)
            elif self.config.optim== 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            elif self.config.optim == 'adamw':
                optimizer = torch.optim.AdamW([x_trial, labels], lr=self.lr)
            else:
                raise ValueError()
        else:
            if self.config.optim == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.lr)
            elif self.config.optim == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr = self.lr, momentum=0.9, nesterov=True)
            elif self.config.optim== 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            elif self.config.optim == 'adamw':
                optimizer = torch.optim.AdamW([x_trial, labels], lr=self.lr)
            else:
                raise ValueError()

        recon_iterations = self.config.recon_iterations
        if self.config.lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[recon_iterations // 2.667, recon_iterations // 1.6,

                                                                         recon_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            self.model_copy = copy.deepcopy(self.original_model)
            for iteration in range(recon_iterations):
                closure = self._gradient_closure(optimizer, x_trial.to(self.setup), input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config.lr_decay:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config.boxed:
                        lower = -self.dm / self.ds
                        upper = (1 - self.dm) / self.ds
                        x_trial.data = torch.clamp(x_trial, min=lower, max=upper)
                    # if self.debug['boxed']:
                    #     self.dm, self.ds = torch.tensor(self.dm).to(x_trial.device), torch.tensor(self.ds).to(x_trial.device)
                    #     x_trial.data = torch.max(torch.min(x_trial, (1 - self.dm) / self.ds), -self.dm / self.ds)


                    if (iteration + 1 == recon_iterations) or iteration % 500 == 0 and self.verbose:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config.filter== 'none':
                            pass
                        elif self.config.filter== 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()


        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.original_model.zero_grad()
            loss_ce = self.loss_fn_ce(self.original_model(x_trial.to(self.setup)), label)
            loss_kl = self.loss_fn(self.original_model(x_trial.to(self.setup)), self.model_copy(x_trial.to(self.setup)))
            loss = loss_ce + loss_kl

            gradient = torch.autograd.grad(loss, self.original_model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config.cost_fn, indices=self.config.indices,
                                            weights=self.config.weights)

            if self.config.total_variation> 0:
                rec_loss += self.config.total_variation * TV(x_trial)
            rec_loss.backward()
            if self.config.signed:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config.scoring_choice == 'loss':
            self.original_model.zero_grad()
            x_trial.grad = None
            loss_ce = self.loss_fn_ce(self.original_model(x_trial), label)
            loss_kl = self.loss_fn(self.original_model(x_trial), self.model_copy(x_trial))
            loss =  loss_ce + loss_kl

            gradient = torch.autograd.grad(loss, self.original_model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config.cost_fn, indices=self.config.indices,
                                        weights=self.config.weights)
        elif self.config.scoring_choice== 'tv':
            return TV(x_trial)
        elif self.config.scoring_choice in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config.scoring_choice} ...')
        if self.config.scoring_choice == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config.scoring_choice == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.original_model.zero_grad()
        if self.reconstruct_label:
            labels = self.original_model(x_optimal).softmax(dim=1) #TODO: change loss here
        loss = self.loss(self.original_model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.original_model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config.cost_fn,
                                            indices=self.config.indices,
                                            weights=self.config.weights)
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class MultiStepReconstructor(InverseGradReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config: InverseGradConfig = InverseGradConfig(), num_images=1, use_updates=True, batch_size=0):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.original_model.zero_grad()
            parameters = loss_steps(self.original_model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config.cost_fn, indices=self.config.indices,
                                            weights=self.config.weights)

            if self.config.total_variation > 0:
                rec_loss += self.config.total_variation * TV(x_trial)
            rec_loss.backward()
            if self.config.signed:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config.scoring_choice == 'loss':
            self.original_model.zero_grad()
            parameters = loss_steps(self.original_model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config.cost_fn, indices=self.config.indices,
                                        weights=self.config.weights)
        elif self.config.scoring_choice== 'tv':
            return TV(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss_ce = loss_fn(outputs, labels_).sum()
    
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param + lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
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
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
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
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)

