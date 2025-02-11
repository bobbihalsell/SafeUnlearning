import torch
from tqdm import trange
from collections import defaultdict, OrderedDict
from recovery.nn import MetaMonkey
from .metrics import total_variation as TV
from .metrics import InceptionScore
from copy import deepcopy


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)  # Make sure the user passed in a dictionary with all the right arguments.
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std  # A tuple
        self.num_images = num_images  # Number of images per attempt (restarts)

        if self.config['scoring_choice'] == 'inception': #  Inception Score (IS) is an algorithm used to assess the quality of images created by a generative image model (from Wikipedia)
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')  # Take the mean loss per sample

    def reconstruct(self, input_gradient, data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)  # Initialize a big tensor of all our randomly initialized images.
        dm, ds = self.mean_std
        history_list = []
        scores = torch.zeros(self.config['restarts'])  # We'll store a score for each attempt


        if labels is None:  # Label reconstruction (Black-box case in the paper)
            
            # DLG label recovery
            # However this also improves conditioning for some LBFGS cases
            self.reconstruct_label = True

            def loss_fn(pred, labels):  # NLL loss
                labels = torch.nn.functional.softmax(labels, dim=-1)
                return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
            self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images  # Ensure we have labels for each image
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                # Run a trial passing in the batch of randomly initialized images x[trial], 
                # data is usually X_unlearn, the features of the forget set
                # Labels is usually y_unlearn: The true labels of the forget set
                # I think dryrun is a means to cut the trials early, if the reconstruction takes a while usually.
                x_trial, labels, history = self._run_trial(x[trial], input_gradient, data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_gradient, labels)
                x[trial] = x_trial
                history_list.append(history)
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_gradient, stats)

        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)  # Minimises the cost between the gradient and input gradient
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores
            x_optimal = x[optimal_index]


        print(f'Total time: {time.time()-start_time}.')
        return x, stats, history_list, x_optimal

    def _init_images(self, img_shape):  # Initialize a random image
        if self.config['init'] == 'randn':  # Random initialization of image values according to normal dist.
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)  # **self.setup tells the tensor to use GPU/CPU and a specific dtype
        elif self.config['init'] == 'rand':  # Random according to unif distribution
            # But we allow for values in -1 to 1 via the shifting and scaling operation (x-0.5)*2.
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':  # All 0s initialization
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup) #*img_shape unpacks a tuple of args, so it unpacks the image dims.
        # It initializes a tensor of shape (number of tries, number of images per try, shape of the images)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_gradient, data, labels, dryrun=False):
        x_trial.requires_grad = True  # For label construction we do need to modify x to find x where the model is extremely confident in a class, as stated in the paper. So we need the grad of loss w.r.t x.
        history = []

        if self.reconstruct_label:  # Then we'd have to randomly initialize some label to start.
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'adamw':
                optimizer = torch.optim.AdamW([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:  # Then we just want to reconstruct the image
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'adamw':
                optimizer = torch.optim.AdamW([x_trial, labels], lr=self.config['lr'])  # TODO I think there's a bug here
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
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

                                                                         max_iterations // 1.142], gamma=0.5)   # 3/8 5/8 7/8
        try:
            tqdm_range = trange(1, max_iterations + 1, desc='Loss', leave=True)
            for iteration in tqdm_range:
                # Project into image space
                if self.config['boxed']:
                    x_trial.data = torch.clamp(x_trial.data, 0, 1)  # To ensure we don't allow the image values to exceed allowable pixel values.
                # Compute the gradient of x_trial w.r.t the model (usually the finetuned model)
                closure = self._gradient_closure(normalizer, optimizer, x_trial, input_gradient, labels)  # This is a callable
                # .step( . ) is going to update x_trial, and maybe labels if we're performing label reconstruction.
                rec_loss = optimizer.step(closure)  # The closure is invoked to compute loss and perform backprop. 
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    if (iteration == max_iterations) or iteration % 100 == 0:
                        history.append(x_trial.detach().cpu().clone())  # Append the x_trial at the time step just to show the iterative reconstruction history

                    if iteration % 500 == 0:  # Every 500 steps,
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()
                tqdm_range.set_description(f'Loss: {rec_loss.item():2.4f}. MSE: {(x_trial.data - data).pow(2).mean().item()}')
                tqdm_range.refresh()
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels, history

    def _gradient_closure(self, normalizer, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(normalizer(x_trial)), label)  # Compute the loss w.r.t inputs
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)  # Compute the gradients, and allow for further gradient computations on the gradients themselves with graph=True
            rec_loss = reconstruction_costs([gradient], input_gradient, 
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])  # Calculate the reconstruction loss between some given input gradient vector and the gradient vector for this current model.

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)  # Add on TV normalization to the loss if necessary.
            rec_loss.backward()
            if self.config['signed']:  # If we want to instead of considering the value of the gradients, only consider their sign e.g.(+1, -1)
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats

def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """ Compute the gradient cost between gradients and reference input gradients.
    Example: Cosine similarity.

    Input gradient is the given difference between the original and unlearned model parameters.
    So gradients would be the current gradients. This is usually passed in as [gradients], a list of len 1.
    """
    # Compute the gradient cost between grads and reference grads.
    if isinstance(indices, list):
        pass
    elif indices == 'def':  # Default, check all input gradient values.
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
    # Allows us to weight different parameters of the gradient differently.
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
    return total_costs / len(gradients)  # Normalize by the number of gradients
