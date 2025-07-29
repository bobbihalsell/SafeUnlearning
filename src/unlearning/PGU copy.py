import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unlearning.base import BaseUnlearner
from unlearning.unlearn_utils import l2_penalty

# Assuming these are available from your utils/unlearn_utils.py and utils/general_utils.py
# For now, I'll place them as internal functions for clarity, but they can remain external.
# from unlearning.base import BaseUnlearner # Assuming BaseUnlearner is in your project
# from unlearning.unlearn_utils import (
#     compute_svd,
#     compute_retain_svd,
#     freeze_norm_stats,
#     get_entropy,
#     # l2_penalty if you intend to use it, but the paper doesn't mention it for this method
# )

# --- Your unlearn_utils.py functions integrated for self-containment ---
# (You can keep these in utils/unlearn_utils.py and import them if preferred)

# A simplified create_feature_extractor for demonstration. You'll need your actual implementation.


class PGUnlearner(BaseUnlearner):
    """
    Implements the unlearning method based on Core Gradient Space (CGS)
    and gradient projection orthogonal to the retaining data's CGS,
    as described in the provided paper excerpt.
    """

    def __init__(
        self,
        device,
        evaluate: bool = False,
        wandb_enabled: bool = False,
        verbose: bool = True,
    ):
        super().__init__(device, evaluate, wandb_enabled, verbose)
        # Small constant epsilon for numerical stability in the loss function
        self.epsilon = 1e-6

    def _cgs_loss(self, outputs, labels, lambda_val: float, epsilon: float = 1e-6):
        """
        Calculates the custom loss function for CGS unlearning (Equation 1).
        L = sum_{i in D_f} sum_{c=1}^C (-y_{i,c} log(1 - p_{i,c} + epsilon) - lambda * p_{i,c} log(p_{i,c}))

        Args:
            outputs (torch.Tensor): Model's raw outputs (logits) for the batch.
            labels (torch.Tensor): True labels for the batch.
            lambda_val (float): Lambda parameter for the entropy term.
                                > 0 for entropy maximization, < 0 for entropy minimization.
            epsilon (float): Small constant for numerical stability.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        # Convert labels to one-hot encoding
        num_classes = outputs.shape[1]
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        # Calculate probabilities pi,c
        probabilities = F.softmax(outputs, dim=1)

        # First term: "reverse" cross-entropy loss
        # -y_i,c * log(1 - p_i,c + epsilon)
        term1 = -labels_one_hot * torch.log(1 - probabilities + epsilon)
        loss1 = torch.sum(term1) # Sum over classes and samples

        # Second term: entropy regularization
        # -lambda * p_i,c * log(p_i,c)
        # Note: torch.distributions.Categorical(probs=probabilities).entropy() calculates H(P) = -sum(p_i log p_i)
        # So, -lambda * (sum_c (p_i,c log p_i,c)) is lambda * entropy
        # Paper's term: -λ * pi,c log(pi,c) -> sum_c(-lambda * p_i,c log(p_i,c)) = -lambda * H(P)
        # So we want -lambda * entropy.
        entropy_term = -lambda_val * torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1)
        loss2 = torch.sum(entropy_term) # Sum over samples

        total_loss = loss1 + loss2
        return total_loss / outputs.shape[0] # Average over batch size

    # def _get_entropy(model, loader, ignore_first_k=0):
    #     with torch.no_grad():
    #         model.eval()
    #         entropies = []
    #         for batch_idx, (images, labels) in enumerate(loader):
    #             labels = labels.to(device)
    #             images = images.to(device)

    #             outputs = model(images)
    #             outputs = outputs[:, ignore_first_k:]
    #             scores = F.softmax(outputs, dim=1)
    #             entropy = torch.log(torch.sum(-scores*torch.log(scores), dim=1))
    #             entropies.append(entropy)
                
    #         entropies = torch.cat(entropies, dim=0).cpu().numpy().tolist()
    #         return entropies

    def _compute_svd(model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
        if conv_fea_dict is None:
            conv_fea_dict = model.conv_fea_dict
        if linear_fea_dict is None:
            linear_fea_dict = model.linear_fea_dict
        start_time = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        
        for md in model.modules():
            name = md.__class__.__name__
            if name == 'Dropout':
                md.training = True

        fea_dict = {}
        for val in {**conv_fea_dict, **linear_fea_dict}.values():
            fea_dict[val] = val

        printer(fea_dict)
        tmp_fea_dict = {**fea_dict}
        if 'input' in fea_dict:
            features = {'input': []}
            tmp_fea_dict.pop('input')
        else:
            features = {}

        fea_ext = create_feature_extractor(model, tmp_fea_dict)

        covar, svd = {}, {}
        for key in {**conv_fea_dict, **linear_fea_dict}.keys():
            covar[key] = 0
        with torch.no_grad():
            for _ in range(int(epochs)):
                for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    if 'input' in fea_dict:
                        features['input'] = imgs

                    feats = fea_ext(imgs)
                    for fea_name in feats:
                        features[fea_name] = feats[fea_name].detach()

                    for layer in conv_fea_dict: 
                        ks = eval(f'model.{layer}').kernel_size
                        padding = eval(f'model.{layer}').padding
                        f = features[conv_fea_dict[layer]]
                        patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
                        fea_dim = patch.shape[1]
                        patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                        covar[layer] += torch.mm(patch.permute(1, 0), patch)

                    for layer in linear_fea_dict:
                        f = features[linear_fea_dict[layer]].double().squeeze()
                        covar[layer] += torch.mm(f.permute(1, 0), f)
            
            for layer in covar:
                stime = time.time()
                U, S, _ = torch.svd(covar[layer]/epochs)
                svd[layer] = {'U': U, 'S': torch.sqrt(S)}
                print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f}')

        process_time = time.time() - start_time
        print(f'Processing time: {process_time:.04f}')
        return svd

    def _compute_retain_svd(full_svd, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
        if conv_fea_dict is None:
            conv_fea_dict = model.conv_fea_dict
        if linear_fea_dict is None:
            linear_fea_dict = model.linear_fea_dict
        start_time = time.time()
        fea_dict = {}
        for val in {**conv_fea_dict, **linear_fea_dict}.values():
            fea_dict[val] = val

        max_dim = 0
        for k in {**conv_fea_dict, **linear_fea_dict}:
            w = eval(f'model.{k}.weight')
            fea_dim = w.numel() // w.shape[0]
            max_dim = max(max_dim, fea_dim) 

        # min_epochs = max(epochs, math.ceil(max_dim/len(data_loader.dataset)))
        min_epochs = epochs

        tmp_fea_dict = {**fea_dict}
        if 'input' in fea_dict:
            features = {'input': []}
            tmp_fea_dict.pop('input')
        else:
            features = {}

        fea_ext = create_feature_extractor(model, tmp_fea_dict)

        covar, retain_svd = {}, {}
        for key in {**conv_fea_dict, **linear_fea_dict}.keys():
            covar[key] = 0
        with torch.no_grad():
            for _ in range(int(min_epochs)):
                for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)
                    if 'input' in fea_dict:
                        features['input'] = imgs

                    feats = fea_ext(imgs)
                    for fea_name in feats:
                        features[fea_name] = feats[fea_name].detach()

                    for layer in conv_fea_dict: 
                        ks = eval(f'model.{layer}').kernel_size
                        padding = eval(f'model.{layer}').padding
                        f = features[conv_fea_dict[layer]]
                        patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
                        fea_dim = patch.shape[1]
                        patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                        covar[layer] += torch.mm(patch.permute(1, 0), patch)

                    for layer in linear_fea_dict:
                        f = features[linear_fea_dict[layer]].double().squeeze()
                        covar[layer] += torch.mm(f.permute(1, 0), f)
            
            process_time = time.time() - start_time
            printer(f'Processing time: {process_time:.04f}')
            for layer in covar:
                stime = time.time()
                U, S = full_svd[layer]['U'].to(device), full_svd[layer]['S'].to(device)
                M = torch.mm(torch.mm(U, torch.diag(S**2)), U.t())

                M1 = M - covar[layer]/min_epochs
                U1_, S1sq_, _ = torch.svd(M1)
                retain_svd[layer] = {'U': U1_, 'S': torch.sqrt(S1sq_)}
                printer(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - M: {M.shape}')

        process_time = time.time() - start_time
        print(f'Processing time: {process_time:.04f}')
        return retain_svd
        
    def _freeze_norm_stats(net):
        try:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

        except ValueError:  
            print("error with BatchNorm")
            return

    def evaluate(model, data_loader, description='', num_forget_classes=None):
        ce_losses = AverageMeter()
        accuracies = AverageMeter()
        with torch.no_grad():
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            loss_function = nn.CrossEntropyLoss()
            for batch_idx, (images, labels) in enumerate(data_loader):
                labels = labels.to(device)
                images = images.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)
                ce_losses.update(loss.item(), images.shape[0])
                if num_forget_classes is not None:
                    labels = labels - num_forget_classes
                    outputs = outputs[:, num_forget_classes:]
                acc = accuracy(outputs, labels)
                accuracies.update(acc[0].item(), images.shape[0])

            print('[{}] Loss: {:.04f} - Acc: {:.04f}'.format(description, ce_losses.avg, accuracies.avg))
        return ce_losses.avg, accuracies.avg



    

    def unlearn(self, model: nn.Module, data_dict: Dict[str, DataLoader], **kwargs):
        model.to(self.device)
        unlearned_model, scheduler = self._setup_unlearning(model, data_dict, **kwargs)

        # Hyperparameters specific to CGS
        # retained_var: gamma_l threshold for CGS. 
        self.retained_var_threshold = kwargs.get('retained_var_threshold', 0.95)
        # lambda_val: Lambda from the loss function. Your `loss2_w` acts as this.
        self.lambda_val = kwargs.get('lambda_val', 1.0) # Default to positive for entropy maximization
        # Trust Region Gradient Projection for de-poisoning (last linear layer)
        self.apply_trgp = kwargs.get('apply_trgp', False)
        # If apply_trgp is True, we need to know the last linear layer name
        self.last_linear_layer_name = kwargs.get('last_linear_layer_name', 'fc') # Common name for classifier layer

        # Ensure required data is available
        if "forget" not in data_dict.keys():
            raise ValueError("'forget' data must be in data_dict for CGS unlearning.")
        if "train" not in data_dict.keys():
             raise ValueError("'train' (or equivalent full dataset) data must be in data_dict to compute full SVD.")


        # 1. Compute Full Data Covariance and Forget Covariance
        full_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/svd_full_data.pt'
        os.makedirs(f'exp/svd/{cfg.model}_{cfg.dataset}/', exist_ok=True)
        if not os.path.isfile(full_svd_file):
            self.full_svd = self._compute_svd(
            model, 
            data_dict["train"], 
            epochs=1 # epochs=kwargs.get('svd_full_epochs', 1), # Use more epochs for more accurate SVD
            )
            torch.save(self.full_svd, full_svd_file)
        else:
            self.full_svd = torch.load(full_svd_file)

        current_svd = copy.deepcopy(full_svd)

        # Start unlearning
        unl_start_time = time.time()

        ### SKIPPING AUGMENTATION CODE###
        # print('forget_train_noaug')
        # learn_cfg.dataset.train.params.forget_range = unlearn_step.forget_range
        # learn_cfg.dataset.train.params.ignore_range = unlearn_step.get('ignore_range', [])
        # learn_cfg.dataset.train.params.data_section = 'forget'
        # forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
        #                                                         **learn_cfg.dataset.train.params)
        # forget_train_noaug_loader = DataLoader(forget_train_noaug, shuffle=False, num_workers=6, batch_size=128)

        # forget_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
        #                                                         **learn_cfg.dataset.train.params)
        # forget_train_aug_loader = DataLoader(forget_train_aug, shuffle=False, num_workers=6, batch_size=128)

        # learn_cfg.dataset.train.params.forget_range = [[0, unlearn_step.forget_range[0][1]]]*len(unlearn_step.forget_range)
        # all_forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
        #                                                             **learn_cfg.dataset.train.params)
        # all_forget_train_noaug_loader = DataLoader(all_forget_train_noaug, shuffle=False, num_workers=6, batch_size=128)

        # print('retain_train_noaug')
        ###############################

        # GET RETAIN DATA SVD
        retain_svd_file = f'exp/svd/{cfg.model}_{cfg.dataset}/retain_{e}.pt'
        retain_svd = compute_retain_svd(current_svd, model, data_dict['forget'], epochs=1)

        current_svd = retain_svd

        P = {}
        for layer in retain_svd:
            # retained_var += cfg.get('retained_var_step', 0)*unlearn_step_idx
            # if retained_var == 1.0:
            #     print(f'Skip layer: {layer}')
            #     continue
            # print(f'Retained var: {retained_var:.04f}')
            k = torch.sum((torch.cumsum(retain_svd[layer]['S'], dim=0) / torch.sum(retain_svd[layer]['S'])) <= retained_var)
            # logger.info(f"{k.item()}\t {retain_svd[layer]['S'].shape[0]}\t {100*k.item()/retain_svd[layer]['S'].shape[0]:.06f}")
            
            M = retain_svd[layer]['U'][:, :k]
            P[layer] = torch.mm(M, M.t()).to(device).float()

        offset = 0.1
        wd = 0
        num_bins = 100

        alpha = np.exp(np.log(end_lr/start_lr) / num_epochs)
        base_lr = start_lr

        retain_train_res, forget_train_res, val_res = [], [], []

        exp_dir = f'{cfg.work_dir}/results/{suffix}'
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f'{exp_dir}/ckp/', exist_ok=True)
        os.makedirs(f'{exp_dir}/entropy_retrain_unlearn/', exist_ok=True)

        name_mapping = {}
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            # Unlearn Training
            unlearn_model.train()
            unlearn_model.apply(_freeze_norm_stats)
            for batch_idx, (images, labels) in enumerate(data_dict['forget']):
                if epoch == 0:
                    break
                labels = labels.to(device)
                images = images.to(device)

                optimizer.zero_grad()
                outputs = unlearn_model(images)
                
                # loss = loss_function(outputs, labels)
                logits = F.softmax(outputs, dim=1)
                loss1, loss2 = 0, 0
                for j in range(images.shape[0]):
                    logit = logits[j][labels[j]]
                    loss1 += -torch.log(1 - ((logit) - offset))
                    
                loss2 = -torch.sum(-logits*torch.log(logits + 1e-15), dim=1).mean()
                loss1 /= images.shape[0]
                loss = loss1_w*loss1 + loss2_w*loss2

                losses1.update(loss1.item(), images.shape[0])
                losses2.update(loss2.item(), images.shape[0])
                ce_losses.update(loss.item(), images.shape[0])
                loss.backward()

                with torch.no_grad():
                    for name, param in unlearn_model.named_parameters():
                        if name not in name_mapping:
                            P_name = name
                            for i in range(20):
                                P_name = P_name.replace(f'.{i}', f'[{i}]')
                            P_name = P_name.replace('.weight', '')
                            name_mapping[name] = P_name
                        else:
                            P_name = name_mapping[name]

                        if P_name not in P:
                            param.grad.data.fill_(0)
                            continue

                        # print(name, P_name)
                        sz = param.grad.data.shape[0]
                        reg_grad = param.grad.data.add(param.data, alpha=wd)
                        reg_grad = reg_grad - torch.mm(reg_grad.view(sz,-1), P[P_name]).view(param.size())
        
                        lr = base_lr
                        param.data -= lr * reg_grad
            
            # Evaluate
            forward_pass_elapsed = time.time() - epoch_start_time
            if self.wandb_enabled:
                self._log_forward_pass_time_in_wandb(
                    epoch=e + 1, time=forward_pass_elapsed
                )
            if self.verbose:
                self._print_forward_pass_metrics(e, forward_pass_elapsed)
            if self.evaluate:
                self._evaluate_all_splits(
                    model=unlearned_model,
                    data_dict=data_dict,
                    epoch=e + 1,
                )

            if scheduler is not None:
                scheduler.step()

        return unlearned_model, self.logs