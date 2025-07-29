import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.fx as fx
from tqdm import tqdm

from unlearning.base import BaseUnlearner
from unlearning.unlearn_utils import l2_penalty

import os
import torch.fx as fx

from PGUutils import *
import copy
import shutil

from collections import OrderedDict
from torchvision.models.feature_extraction import get_graph_node_names


# class PGUnlearner(BaseUnlearner):
#     """
#     Implements the unlearning method based on Core Gradient Space (CGS)
#     and gradient projection orthogonal to the retaining data's CGS,
#     as described in the provided paper excerpt.
#     """

#     def __init__(
#         self,
#         device,
#         evaluate: bool = False,
#         wandb_enabled: bool = False,
#         verbose: bool = True,
#     ):
#         super().__init__(device, evaluate, wandb_enabled, verbose)
#         # Small constant epsilon for numerical stability in the loss function
#         self.epsilon = 1e-6

#     def _get_feature_dict(self, model):
#         """
#         Automatically generate conv_fea_dict and linear_fea_dict for a given model.
#         """
#         try:
#             _, eval_nodes = get_graph_node_names(model)
#             conv_fea_dict = OrderedDict()
#             linear_fea_dict = OrderedDict()
            
#             # Traverse the model's modules and keep track of the module names
#             for name, module in model.named_modules():
#                 if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
#                     conv_fea_dict[name] = name
#                 elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
#                     linear_fea_dict[name] = name
            
#             print(f"Generated {len(conv_fea_dict)} conv layers and {len(linear_fea_dict)} linear layers")
#             return conv_fea_dict, linear_fea_dict
#         except Exception as e:
#             print(f"Error in _get_feature_dict: {e}")
#             raise

#     def _get_module_by_name(self, model, layer_name):
#         """
#         Get a module by its name from the model.
#         """
#         try:
#             parts = layer_name.split('.')
#             module = model
#             for part in parts:
#                 if part.isdigit():
#                     module = module[int(part)]
#                 else:
#                     module = getattr(module, part)
#             return module
#         except Exception as e:
#             print(f"Error getting module {layer_name}: {e}")
#             raise

#     def _compute_svd(self, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
#         """
#         Compute SVD for the input representations of each layer.
#         """
#         print(f"Starting _compute_svd with {len(data_loader)} batches")
        
#         # Auto-generate feature dictionaries if not provided
#         if conv_fea_dict is None or linear_fea_dict is None:
#             if hasattr(model, 'conv_fea_dict') and hasattr(model, 'linear_fea_dict'):
#                 conv_fea_dict = model.conv_fea_dict if conv_fea_dict is None else conv_fea_dict
#                 linear_fea_dict = model.linear_fea_dict if linear_fea_dict is None else linear_fea_dict
#             else:
#                 conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
#                 print(f"Auto-generated conv_fea_dict: {list(conv_fea_dict.keys())}")
#                 print(f"Auto-generated linear_fea_dict: {list(linear_fea_dict.keys())}")
        
#         start_time = time.time()
#         model.eval()
        
#         # Enable dropout if needed
#         for md in model.modules():
#             if md.__class__.__name__ == 'Dropout':
#                 md.training = True

#         # Setup feature extraction
#         fea_dict = {}
#         for val in {**conv_fea_dict, **linear_fea_dict}.values():
#             fea_dict[val] = val

#         print(f"Feature dict: {list(fea_dict.keys())}")
#         tmp_fea_dict = {**fea_dict}
#         if 'input' in fea_dict:
#             features = {'input': []}
#             tmp_fea_dict.pop('input')
#         else:
#             features = {}

#         fea_ext = create_feature_extractor(model, tmp_fea_dict)

#         # Initialize covariance matrices
#         covar, svd = {}, {}
#         for key in {**conv_fea_dict, **linear_fea_dict}.keys():
#             covar[key] = 0
            
#         print(f"Initialized covariance for layers: {list(covar.keys())}")

#         with torch.no_grad():
#             for epoch in range(int(epochs)):
#                 print(f"Processing epoch {epoch + 1}/{epochs}")
#                 for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader, desc=f"SVD Epoch {epoch+1}")):
#                     imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    
#                     if 'input' in fea_dict:
#                         features['input'] = imgs

#                     # Extract features
#                     feats = fea_ext(imgs)
#                     for fea_name in feats:
#                         features[fea_name] = feats[fea_name].detach()

#                     # Process conv layers
#                     for layer in conv_fea_dict:
#                         try:
#                             # FIXED: Use _get_module_by_name instead of eval
#                             layer_module = self._get_module_by_name(model, layer)
#                             ks = layer_module.kernel_size
#                             padding = layer_module.padding
                            
#                             f = features[conv_fea_dict[layer]]
#                             patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
#                             fea_dim = patch.shape[1]
#                             patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
#                             covar[layer] += torch.mm(patch.permute(1, 0), patch)
#                         except Exception as e:
#                             print(f"Error processing conv layer {layer}: {e}")
#                             raise

#                     # Process linear layers
#                     for layer in linear_fea_dict:
#                         try:
#                             f = features[linear_fea_dict[layer]].double().squeeze()
#                             if f.ndim == 1:
#                                 f = f.unsqueeze(0)
#                             covar[layer] += torch.mm(f.permute(1, 0), f)
#                         except Exception as e:
#                             print(f"Error processing linear layer {layer}: {e}")
#                             raise

#             # Compute SVD for each layer
#             print("Computing SVD for each layer...")
#             for layer in covar:
#                 try:
#                     stime = time.time()
#                     U, S, _ = torch.svd(covar[layer] / epochs)
#                     svd[layer] = {'U': U, 'S': torch.sqrt(S)}
#                     print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - Shape: {U.shape}')
#                 except Exception as e:
#                     print(f"Error computing SVD for layer {layer}: {e}")
#                     raise

#         process_time = time.time() - start_time
#         print(f'SVD Processing time: {process_time:.04f}')
#         print(f"DEBUG: _compute_svd created SVD for layers: {list(svd.keys())}")
        
#         return svd

#     def _compute_retain_svd(self, full_svd, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
#         """
#         Compute retain SVD by subtracting forget data contribution from full SVD.
#         """
#         print(f"Starting _compute_retain_svd with {len(data_loader)} batches")
#         print(f"DEBUG: _compute_retain_svd received full_svd with layers: {list(full_svd.keys())}")
        
#         # Auto-generate feature dictionaries if not provided
#         if conv_fea_dict is None or linear_fea_dict is None:
#             if hasattr(model, 'conv_fea_dict') and hasattr(model, 'linear_fea_dict'):
#                 conv_fea_dict = model.conv_fea_dict if conv_fea_dict is None else conv_fea_dict
#                 linear_fea_dict = model.linear_fea_dict if linear_fea_dict is None else linear_fea_dict
#             else:
#                 conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
        
#         expected_layers = {**conv_fea_dict, **linear_fea_dict}
#         print(f"DEBUG: _compute_retain_svd expects layers: {list(expected_layers.keys())}")
        
#         # Check if all expected layers are in full_svd
#         missing_layers = []
#         for layer in expected_layers:
#             if layer not in full_svd:
#                 missing_layers.append(layer)
        
#         if missing_layers:
#             print(f"ERROR: Missing layers in full_svd: {missing_layers}")
#             print(f"Available layers in full_svd: {list(full_svd.keys())}")
#             raise KeyError(f"Missing layers in full_svd: {missing_layers}")
        
#         start_time = time.time()
#         fea_dict = {}
#         for val in {**conv_fea_dict, **linear_fea_dict}.values():
#             fea_dict[val] = val

#         # Calculate max dimension for epochs calculation
#         max_dim = 0
#         for k in {**conv_fea_dict, **linear_fea_dict}:
#             try:
#                 layer_module = self._get_module_by_name(model, k)
#                 w = layer_module.weight
#                 fea_dim = w.numel() // w.shape[0]
#                 max_dim = max(max_dim, fea_dim)
#             except Exception as e:
#                 print(f"Error calculating dimensions for layer {k}: {e}")
#                 raise

#         min_epochs = epochs

#         # Setup feature extraction
#         tmp_fea_dict = {**fea_dict}
#         if 'input' in fea_dict:
#             features = {'input': []}
#             tmp_fea_dict.pop('input')
#         else:
#             features = {}

#         fea_ext = create_feature_extractor(model, tmp_fea_dict)

#         # Initialize covariance matrices
#         covar, retain_svd = {}, {}
#         for key in {**conv_fea_dict, **linear_fea_dict}.keys():
#             covar[key] = 0

#         with torch.no_grad():
#             for epoch in range(int(min_epochs)):
#                 print(f"Processing retain SVD epoch {epoch + 1}/{min_epochs}")
#                 for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader, desc=f"Retain SVD Epoch {epoch+1}")):
#                     imgs = imgs.to(self.device)
#                     lbls = lbls.to(self.device)
                    
#                     if 'input' in fea_dict:
#                         features['input'] = imgs

#                     # Extract features
#                     feats = fea_ext(imgs)
#                     for fea_name in feats:
#                         features[fea_name] = feats[fea_name].detach()

#                     # Process conv layers
#                     for layer in conv_fea_dict:
#                         try:
#                             layer_module = self._get_module_by_name(model, layer)
#                             ks = layer_module.kernel_size
#                             padding = layer_module.padding
                            
#                             f = features[conv_fea_dict[layer]]
#                             patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
#                             fea_dim = patch.shape[1]
#                             patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
#                             covar[layer] += torch.mm(patch.permute(1, 0), patch)
#                         except Exception as e:
#                             print(f"Error processing conv layer {layer}: {e}")
#                             raise

#                     # Process linear layers
#                     for layer in linear_fea_dict:
#                         try:
#                             f = features[linear_fea_dict[layer]].double().squeeze()
#                             if f.ndim == 1:
#                                 f = f.unsqueeze(0)
#                             covar[layer] += torch.mm(f.permute(1, 0), f)
#                         except Exception as e:
#                             print(f"Error processing linear layer {layer}: {e}")
#                             raise

#             # Compute retain SVD using paper's formula
#             print("Computing retain SVD for each layer...")
#             for layer in covar:
#                 try:
#                     stime = time.time()
#                     U, S = full_svd[layer]['U'].to(self.device), full_svd[layer]['S'].to(self.device)
#                     M = torch.mm(torch.mm(U, torch.diag(S**2)), U.t())

#                     M1 = M - covar[layer] / min_epochs
#                     U1_, S1sq_, _ = torch.svd(M1)
#                     retain_svd[layer] = {'U': U1_, 'S': torch.sqrt(S1sq_)}
#                     print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - M: {M.shape}')
#                 except Exception as e:
#                     print(f"Error computing retain SVD for layer {layer}: {e}")
#                     raise

#         process_time = time.time() - start_time
#         print(f'Retain SVD Processing time: {process_time:.04f}')
        
#         return retain_svd

#     def _freeze_norm_stats(self, net):
#         """
#         Freeze batch normalization statistics.
#         """
#         try:
#             for m in net.modules():
#                 if isinstance(m, nn.BatchNorm2d):
#                     m.eval()
#                 if isinstance(m, nn.BatchNorm1d):
#                     m.eval()
#         except ValueError:
#             print("error with BatchNorm")
#             return

#     def unlearn(self, model: nn.Module, data_dict: Dict[str, DataLoader], **kwargs):
#         """
#         Main unlearning method.
#         """
#         print("Starting PGU unlearning...")
#         model.to(self.device)
#         unlearned_model, scheduler = self._setup_unlearning(model, data_dict, **kwargs)

#         # Generate feature dictionaries once
#         if hasattr(model, 'conv_fea_dict') and hasattr(model, 'linear_fea_dict'):
#             conv_fea_dict = model.conv_fea_dict
#             linear_fea_dict = model.linear_fea_dict
#         else:
#             conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
#             print(f"Auto-generated conv_fea_dict: {list(conv_fea_dict.keys())}")
#             print(f"Auto-generated linear_fea_dict: {list(linear_fea_dict.keys())}")

#         # Hyperparameters specific to CGS
#         self.retained_var_threshold = kwargs.get('retained_var_threshold', 0.95)
#         self.lambda_val = kwargs.get('lambda_val', 1.0)
#         self.apply_trgp = kwargs.get('apply_trgp', False)
#         self.last_linear_layer_name = kwargs.get('last_linear_layer_name', 'fc')

#         # Ensure required data is available
#         if "forget" not in data_dict.keys():
#             raise ValueError("'forget' data must be in data_dict for CGS unlearning.")
#         if "train" not in data_dict.keys():
#             raise ValueError("'train' (or equivalent full dataset) data must be in data_dict to compute full SVD.")

#         # Compute Full Data SVD - include model info in filename to avoid cache conflicts
#         model_name = model.__class__.__name__
#         layer_hash = str(hash(str(sorted(conv_fea_dict.keys()) + sorted(linear_fea_dict.keys()))))[:8]
#         full_svd_file = f'exp/svd/svd_full_data_{model_name}_{layer_hash}.pt'
#         os.makedirs(f'exp/svd/', exist_ok=True)
        
#         if not os.path.isfile(full_svd_file):
#             print("Computing full SVD (this may take a while)...")
#             self.full_svd = self._compute_svd(
#                 model, 
#                 data_dict["train"], 
#                 conv_fea_dict=conv_fea_dict,
#                 linear_fea_dict=linear_fea_dict,
#                 epochs=1
#             )
#             torch.save(self.full_svd, full_svd_file)
#             print(f"Full SVD saved to {full_svd_file}")
#         else:
#             print(f"Loading full SVD from {full_svd_file}")
#             self.full_svd = torch.load(full_svd_file)

#         current_svd = copy.deepcopy(self.full_svd)

#         # Compute Retain SVD
#         print("Computing retain SVD...")
#         retain_svd = self._compute_retain_svd(
#             current_svd, 
#             model, 
#             data_dict['forget'], 
#             conv_fea_dict=conv_fea_dict,
#             linear_fea_dict=linear_fea_dict,
#             epochs=1
#         )

#         current_svd = retain_svd

#         # Build projection matrices
#         print("Building projection matrices...")
#         P = {}
#         for layer in retain_svd:
#             k = torch.sum((torch.cumsum(retain_svd[layer]['S'], dim=0) / torch.sum(retain_svd[layer]['S'])) <= self.retained_var_threshold)
#             print(f"Layer {layer}: selected {k.item()}/{retain_svd[layer]['S'].shape[0]} components")
            
#             M = retain_svd[layer]['U'][:, :k]
#             P[layer] = torch.mm(M, M.t()).to(self.device).float()
            
#         print(f"Available projection matrices: {list(P.keys())}")
        
#         # Debug: show a few key mappings
#         print("Key projection matrix shapes:")
#         for layer_name in list(P.keys())[:3]:
#             print(f"  {layer_name}: {P[layer_name].shape}")

#         # Training parameters
#         offset = 0.05
#         alpha = np.exp(np.log(self.end_lr/self.start_lr) / self.epochs)
#         base_lr = self.start_lr

#         # Setup directories
#         exp_dir = f'{self.work_dir}/results/{self.exp_name}'
#         if os.path.isdir(exp_dir):
#             shutil.rmtree(exp_dir)
#         os.makedirs(exp_dir, exist_ok=True)
#         os.makedirs(f'{exp_dir}/ckp/', exist_ok=True)
#         os.makedirs(f'{exp_dir}/entropy_retrain_unlearn/', exist_ok=True)

#         # Main training loop
#         name_mapping = {}
#         for e in range(self.epochs):
#             epoch_start_time = time.time()
#             print(f"Epoch {e+1}/{self.epochs}")
            
#             if e == 0:
#                 print("  Skipping training for first epoch (evaluation only)")
            
#             # Unlearn Training
#             unlearned_model.train()
#             unlearned_model.apply(self._freeze_norm_stats)
            
#             for batch_idx, (images, labels) in enumerate(data_dict['forget']):
#                 if e == 0:
#                     break
                    
#                 labels = labels.to(self.device)
#                 images = images.to(self.device)

#                 self.optimizer.zero_grad()
#                 outputs = unlearned_model(images)
                
#                 # Compute loss
#                 logits = F.softmax(outputs, dim=1)
#                 loss1, loss2 = 0, 0
#                 for j in range(images.shape[0]):
#                     logit = logits[j][labels[j]]
#                     loss1 += -torch.log(1 - ((logit) - offset))
                    
#                 loss2 = -torch.sum(-logits*torch.log(logits + 1e-15), dim=1).mean()
#                 loss1 /= images.shape[0]
#                 loss = self.loss1_w*loss1 + self.loss2_w*loss2

#                 loss.backward()

#                 # Apply gradient projection
#                 with torch.no_grad():
#                     for name, param in unlearned_model.named_parameters():
#                         if param.grad is None:
#                             continue
                            
#                         # Map parameter name to layer name
#                         if name not in name_mapping:
#                             # Extract layer name from parameter name
#                             # e.g., "layer1.0.conv1.weight" -> "layer1.0.conv1"
#                             if '.weight' in name:
#                                 layer_name = name.replace('.weight', '')
#                             elif '.bias' in name:
#                                 layer_name = name.replace('.bias', '')
#                             else:
#                                 layer_name = name
#                             name_mapping[name] = layer_name
#                             if e == 1:  # Only print mapping once
#                                 print(f"Mapped parameter {name} to layer {layer_name}")
#                         else:
#                             layer_name = name_mapping[name]

#                         # Skip if no projection matrix for this layer
#                         if layer_name not in P:
#                             if e == 1:  # Only print once per epoch
#                                 print(f"No projection matrix for layer {layer_name}, skipping gradient projection")
#                             continue

#                         # Get projection matrix
#                         proj_matrix = P[layer_name]
                        
#                         # Apply weight decay
#                         reg_grad = param.grad.data.add(param.data, alpha=self.weight_decay)
                        
#                         # Apply projection based on parameter type
#                         if len(param.shape) == 4:  # Conv layer weights [out_ch, in_ch, h, w]
#                             sz = param.grad.data.shape[0]  # output channels
#                             grad_flat = reg_grad.view(sz, -1)  # [out_ch, in_ch*h*w]
                            
#                             # Check dimension compatibility
#                             if grad_flat.shape[1] != proj_matrix.shape[0]:
#                                 if e == 1:  # Only print once per epoch
#                                     print(f"Dimension mismatch for {layer_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
#                                 continue
                                
#                             # Apply projection: grad - grad * P
#                             proj_grad = torch.mm(grad_flat, proj_matrix)
#                             reg_grad = reg_grad - proj_grad.view(param.size())
                            
#                         elif len(param.shape) == 2:  # Linear layer weights [out_features, in_features]
#                             sz = param.grad.data.shape[0]  # output features
#                             grad_flat = reg_grad.view(sz, -1)  # [out_features, in_features]
                            
#                             # Check dimension compatibility
#                             if grad_flat.shape[1] != proj_matrix.shape[0]:
#                                 if e == 1:  # Only print once per epoch
#                                     print(f"Dimension mismatch for {layer_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
#                                 continue
                                
#                             # Apply projection: grad - grad * P
#                             proj_grad = torch.mm(grad_flat, proj_matrix)
#                             reg_grad = reg_grad - proj_grad.view(param.size())
                            
#                         elif len(param.shape) == 1:  # Bias terms
#                             # Skip bias terms or handle differently
#                             pass
                        
#                         # Update parameters
#                         lr = base_lr
#                         param.data -= lr * reg_grad
            
#             # Evaluate
#             forward_pass_elapsed = time.time() - epoch_start_time
#             if self.wandb_enabled:
#                 self._log_forward_pass_time_in_wandb(
#                     epoch=e + 1, time=forward_pass_elapsed
#                 )
#             if self.verbose:
#                 self._print_forward_pass_metrics(e, forward_pass_elapsed)
#             if self.evaluate:
#                 self._evaluate_all_splits(
#                     model=unlearned_model,
#                     data_dict=data_dict,
#                     epoch=e + 1,
#                 )
            
#             self.optimizer.step()
#             if hasattr(self, 'scheduler'):
#                 self.scheduler.step()

#         print("PGU unlearning completed!")
#         return unlearned_model, self.logs


class PGUnlearner(BaseUnlearner):
    """
    Fixed implementation of PGU with proper dimension handling and comprehensive layer coverage.
    """

    def __init__(
        self,
        device,
        evaluate: bool = False,
        wandb_enabled: bool = False,
        verbose: bool = True,
    ):
        super().__init__(device, evaluate, wandb_enabled, verbose)
        self.epsilon = 1e-6

    def _get_feature_dict(self, model):
        """
        Enhanced feature dictionary generation with better layer coverage.
        """
        try:
            _, eval_nodes = get_graph_node_names(model)
            conv_fea_dict = OrderedDict()
            linear_fea_dict = OrderedDict()
            
            # Traverse all modules, not just those in eval_nodes
            for name, module in model.named_modules():
                # Include Conv2d layers
                if isinstance(module, torch.nn.Conv2d):
                    conv_fea_dict[name] = name
                # Include Linear layers  
                elif isinstance(module, torch.nn.Linear):
                    linear_fea_dict[name] = name
            
            print(f"Generated {len(conv_fea_dict)} conv layers and {len(linear_fea_dict)} linear layers")
            print(f"Conv layers: {list(conv_fea_dict.keys())}")
            print(f"Linear layers: {list(linear_fea_dict.keys())}")
            return conv_fea_dict, linear_fea_dict
        except Exception as e:
            print(f"Error in _get_feature_dict: {e}")
            raise

    def _get_module_by_name(self, model, layer_name):
        """Get a module by its name from the model."""
        try:
            parts = layer_name.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except Exception as e:
            print(f"Error getting module {layer_name}: {e}")
            raise

    def _compute_svd_for_parameters(self, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
        """
        Compute SVD based on parameter gradients rather than input activations.
        This ensures dimensional consistency between SVD and gradient projection.
        """
        print(f"Computing SVD based on parameter gradients...")
        
        if conv_fea_dict is None or linear_fea_dict is None:
            conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
        
        model.train()
        
        # Collect gradients for each layer
        layer_gradients = {}
        all_layers = {**conv_fea_dict, **linear_fea_dict}
        
        for layer_name in all_layers:
            layer_gradients[layer_name] = []
        
        # Collect gradients over multiple batches
        total_batches = 0
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"Collecting gradients epoch {epoch+1}")):
                if total_batches >= 50:  # Limit to prevent memory issues
                    break
                    
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                model.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                
                # Collect gradients for each target layer
                for layer_name in all_layers:
                    try:
                        module = self._get_module_by_name(model, layer_name)
                        if hasattr(module, 'weight') and module.weight.grad is not None:
                            # Flatten the gradient
                            grad_flat = module.weight.grad.data.view(module.weight.shape[0], -1)
                            layer_gradients[layer_name].append(grad_flat.cpu().clone())
                    except Exception as e:
                        print(f"Error collecting gradient for {layer_name}: {e}")
                        continue
                
                total_batches += 1
        
        # Compute covariance matrices from gradients
        svd_results = {}
        for layer_name, grads in layer_gradients.items():
            if len(grads) == 0:
                print(f"No gradients collected for {layer_name}")
                continue
                
            # Concatenate all gradient samples
            all_grads = torch.cat(grads, dim=0).double()  # [total_samples, input_dim]
            
            if all_grads.shape[0] < 2:
                print(f"Insufficient gradient samples for {layer_name}")
                continue
            
            # Compute covariance of the input dimensions (second dimension)
            # This gives us the covariance of the input space that the gradients operate on
            input_grads = all_grads.t()  # [input_dim, total_samples]
            covar = torch.mm(input_grads, input_grads.t()) / (all_grads.shape[0] - 1)
            
            # SVD
            try:
                U, S, _ = torch.svd(covar)
                svd_results[layer_name] = {'U': U, 'S': torch.sqrt(S)}
                print(f'Layer {layer_name}: SVD shape {U.shape}, grad samples: {all_grads.shape[0]}')
            except Exception as e:
                print(f"Error computing SVD for {layer_name}: {e}")
                continue
        
        return svd_results

    def _compute_retain_svd_from_gradients(self, full_svd, model, forget_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
        """
        Compute retain SVD by subtracting forget data gradient contribution.
        """
        print(f"Computing retain SVD from gradients...")
        
        if conv_fea_dict is None or linear_fea_dict is None:
            conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
        
        # Compute forget data gradient covariances
        forget_svd = self._compute_svd_for_parameters(model, forget_loader, conv_fea_dict, linear_fea_dict, epochs)
        
        # Compute retain SVD as full - forget
        retain_svd = {}
        for layer_name in full_svd:
            if layer_name not in forget_svd:
                print(f"Layer {layer_name} not in forget SVD, using full SVD")
                retain_svd[layer_name] = full_svd[layer_name]
                continue
                
            try:
                # Get matrices
                U_full, S_full = full_svd[layer_name]['U'].to(self.device), full_svd[layer_name]['S'].to(self.device)
                U_forget, S_forget = forget_svd[layer_name]['U'].to(self.device), forget_svd[layer_name]['S'].to(self.device)
                
                # Reconstruct covariance matrices
                M_full = torch.mm(torch.mm(U_full, torch.diag(S_full**2)), U_full.t())
                M_forget = torch.mm(torch.mm(U_forget, torch.diag(S_forget**2)), U_forget.t())
                
                # Compute difference (retain = full - forget)
                M_retain = M_full - M_forget
                
                # Ensure positive semi-definite
                M_retain = (M_retain + M_retain.t()) / 2
                
                # SVD of retain covariance
                U_retain, S_retain_sq, _ = torch.svd(M_retain)
                
                # Keep only positive eigenvalues
                positive_mask = S_retain_sq > 1e-8
                U_retain = U_retain[:, positive_mask]
                S_retain = torch.sqrt(S_retain_sq[positive_mask])
                
                retain_svd[layer_name] = {'U': U_retain, 'S': S_retain}
                print(f'Layer {layer_name}: Retain SVD shape {U_retain.shape}')
                
            except Exception as e:
                print(f"Error computing retain SVD for {layer_name}: {e}")
                # Fallback to full SVD
                retain_svd[layer_name] = full_svd[layer_name]
        
        return retain_svd

    def _build_projection_matrices(self, retain_svd, retained_var_threshold=0.95):
        """
        Build projection matrices with proper dimension handling.
        """
        print("Building projection matrices...")
        P = {}
        
        for layer_name, svd_data in retain_svd.items():
            U, S = svd_data['U'], svd_data['S']
            
            # Select components based on variance threshold
            cumsum_var = torch.cumsum(S, dim=0) / torch.sum(S)
            k = torch.sum(cumsum_var <= retained_var_threshold)
            k = max(1, min(k.item(), U.shape[1]))  # Ensure valid range
            
            print(f"Layer {layer_name}: selected {k}/{U.shape[1]} components (threshold: {retained_var_threshold})")
            
            # Build projection matrix P = U_k @ U_k^T
            U_k = U[:, :k]
            P_matrix = torch.mm(U_k, U_k.t()).to(self.device).float()
            P[layer_name] = P_matrix
            
            print(f"  Projection matrix shape: {P_matrix.shape}")
        
        return P

    def _apply_gradient_projection(self, model, projection_matrices, verbose=False):
        """
        Apply gradient projection with proper parameter-layer mapping.
        """
        # Build parameter to layer mapping
        param_to_layer = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # Extract layer name from parameter name
            layer_name = None
            if '.weight' in name:
                layer_name = name.replace('.weight', '')
            elif '.bias' in name:
                layer_name = name.replace('.bias', '')
            
            # Check if this layer has a projection matrix
            if layer_name in projection_matrices:
                param_to_layer[name] = layer_name
            elif verbose:
                if layer_name:
                    print(f"No projection matrix for parameter {name} (layer: {layer_name})")
        
        # Apply projections
        for param_name, layer_name in param_to_layer.items():
            param = dict(model.named_parameters())[param_name]
            if param.grad is None:
                continue
                
            proj_matrix = projection_matrices[layer_name]
            
            # Apply projection based on parameter shape
            if len(param.shape) >= 2:  # Weight parameters
                # Reshape to [out_features, in_features]
                out_features = param.shape[0]
                grad_flat = param.grad.data.view(out_features, -1)
                
                # Check dimension compatibility
                if grad_flat.shape[1] != proj_matrix.shape[0]:
                    if verbose:
                        print(f"Dimension mismatch for {param_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
                    continue
                
                # Apply projection: grad_new = grad - grad @ P
                proj_grad = torch.mm(grad_flat, proj_matrix)
                projected_grad = grad_flat - proj_grad
                
                # Reshape back to original parameter shape
                param.grad.data = projected_grad.view(param.shape)
                
                if verbose:
                    print(f"Applied projection to {param_name}: {grad_flat.shape} -> {projected_grad.shape}")

    def unlearn(self, model: nn.Module, data_dict: Dict[str, DataLoader], **kwargs):
        """
        Main unlearning method with fixed dimension handling.
        """
        print("Starting PGU unlearning with fixed dimension handling...")
        model.to(self.device)
        unlearned_model, scheduler = self._setup_unlearning(model, data_dict, **kwargs)

        # Hyperparameters
        self.retained_var_threshold = kwargs.get('retained_var_threshold', 0.95)
        self.lambda_val = kwargs.get('lambda_val', 1.0)

        # Ensure required data is available
        if "forget" not in data_dict.keys():
            raise ValueError("'forget' data must be in data_dict for PGU unlearning.")
        if "train" not in data_dict.keys():
            raise ValueError("'train' data must be in data_dict to compute full SVD.")

        # Generate feature dictionaries
        conv_fea_dict, linear_fea_dict = self._get_feature_dict(unlearned_model)

        # Compute SVDs based on gradients
        print("Computing full dataset SVD...")
        full_svd = self._compute_svd_for_parameters(
            unlearned_model, data_dict["train"], conv_fea_dict, linear_fea_dict, epochs=1
        )

        print("Computing retain SVD...")
        retain_svd = self._compute_retain_svd_from_gradients(
            full_svd, unlearned_model, data_dict['forget'], conv_fea_dict, linear_fea_dict, epochs=1
        )

        # Build projection matrices
        projection_matrices = self._build_projection_matrices(retain_svd, self.retained_var_threshold)

        print(f"Built projection matrices for {len(projection_matrices)} layers")

        # Training loop
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            unlearned_model.train()
            unlearned_model.apply(self._freeze_norm_stats)
            
            for batch_idx, (images, labels) in enumerate(data_dict['forget']):
                labels = labels.to(self.device)
                images = images.to(self.device)

                self.optimizer.zero_grad()
                outputs = unlearned_model(images)
                
                # Compute unlearning loss
                logits = F.softmax(outputs, dim=1)
                loss1, loss2 = 0, 0
                offset = 0.05
                
                for j in range(images.shape[0]):
                    logit = logits[j][labels[j]]
                    loss1 += -torch.log(1 - ((logit) - offset))
                    
                loss2 = -torch.sum(-logits*torch.log(logits + 1e-15), dim=1).mean()
                loss1 /= images.shape[0]
                loss = self.loss1_w*loss1 + self.loss2_w*loss2

                loss.backward()

                # Apply gradient projection
                self._apply_gradient_projection(
                    unlearned_model, 
                    projection_matrices, 
                    verbose=(epoch == 0 and batch_idx == 0)  # Verbose only for first batch
                )

                self.optimizer.step()
            
            # Evaluation
            if self.evaluate:
                self._evaluate_all_splits(
                    model=unlearned_model,
                    data_dict=data_dict,
                    epoch=epoch + 1,
                )

        print("PGU unlearning completed!")
        return unlearned_model, self.logs

    def _freeze_norm_stats(self, net):
        """Freeze batch normalization statistics."""
        try:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        except ValueError:
            print("error with BatchNorm")
            return
# import time
# from typing import Dict

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torch.fx as fx
# from tqdm import tqdm

# from unlearning.base import BaseUnlearner
# from unlearning.unlearn_utils import l2_penalty

# import os
# import torch.fx as fx

# from PGUutils import *
# import copy
# import shutil

# from collections import OrderedDict
# from torchvision.models.feature_extraction import get_graph_node_names

# # from unlearning.base import BaseUnlearner # Assuming BaseUnlearner is in your project
# # from unlearning.unlearn_utils import (
# #     compute_svd,
# #     compute_retain_svd,
# #     freeze_norm_stats,
# #     get_entropy,
# #     # l2_penalty if you intend to use it, but the paper doesn't mention it for this method
# # )

# # --- Your unlearn_utils.py functions integrated for self-containment ---
# # (You can keep these in utils/unlearn_utils.py and import them if preferred)

# # A simplified create_feature_extractor for demonstration. You'll need your actual implementation.


# class PGUnlearner(BaseUnlearner):
#     """
#     Implements the unlearning method based on Core Gradient Space (CGS)
#     and gradient projection orthogonal to the retaining data's CGS,
#     as described in the provided paper excerpt.
#     """

#     def __init__(
#         self,
#         device,
#         evaluate: bool = False,
#         wandb_enabled: bool = False,
#         verbose: bool = True,
#     ):
#         super().__init__(device, evaluate, wandb_enabled, verbose)
#         # Small constant epsilon for numerical stability in the loss function
#         self.epsilon = 1e-6

#     # def _get_entropy(model, loader, ignore_first_k=0):
#     #     with torch.no_grad():
#     #         model.eval()
#     #         entropies = []
#     #         for batch_idx, (images, labels) in enumerate(loader):
#     #             labels = labels.to(self.device)
#     #             images = images.to(self.device)

#     #             outputs = model(images)
#     #             outputs = outputs[:, ignore_first_k:]
#     #             scores = F.softmax(outputs, dim=1)
#     #             entropy = torch.log(torch.sum(-scores*torch.log(scores), dim=1))
#     #             entropies.append(entropy)
                
#     #         entropies = torch.cat(entropies, dim=0).cpu().numpy().tolist()
#     #         return entropies

#     def _get_feature_dict(self, model):
#         """
#         Automatically generate conv_fea_dict and linear_fea_dict for a given model.
#         """
#         from collections import OrderedDict
#         _, eval_nodes = get_graph_node_names(model)
#         conv_fea_dict = OrderedDict()
#         linear_fea_dict = OrderedDict()
        
#         # Traverse the model's modules and keep track of the module names
#         for name, module in model.named_modules():
#             if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
#                 conv_fea_dict[name] = name
#             elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
#                 linear_fea_dict[name] = name
        
#         return conv_fea_dict, linear_fea_dict
    

#     def _get_module_by_name(self, model, layer_name):
#         """
#         Get a module by its name from the model.
#         """
#         parts = layer_name.split('.')
#         module = model
#         for part in parts:
#             if part.isdigit():
#                 module = module[int(part)]
#             else:
#                 module = getattr(module, part)
#         return module


#     def _compute_svd(self, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
#         if conv_fea_dict is None or linear_fea_dict is None:
#             if hasattr(model, 'conv_fea_dict') and hasattr(model, 'linear_fea_dict'):
#                 # Use model's existing dictionaries
#                 conv_fea_dict = model.conv_fea_dict if conv_fea_dict is None else conv_fea_dict
#                 linear_fea_dict = model.linear_fea_dict if linear_fea_dict is None else linear_fea_dict
#             else:
#                 # Auto-generate dictionaries
#                 conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
#                 print(f"Auto-generated conv_fea_dict: {conv_fea_dict}")
#                 print(f"Auto-generated linear_fea_dict: {linear_fea_dict}")
#         start_time = time.time()
#         model.eval()
        
#         for md in model.modules():
#             name = md.__class__.__name__
#             if name == 'Dropout':
#                 md.training = True

#         fea_dict = {}
#         for val in {**conv_fea_dict, **linear_fea_dict}.values():
#             fea_dict[val] = val

#         print(fea_dict)
#         tmp_fea_dict = {**fea_dict}
#         if 'input' in fea_dict:
#             features = {'input': []}
#             tmp_fea_dict.pop('input')
#         else:
#             features = {}

#         fea_ext = create_feature_extractor(model, tmp_fea_dict)

#         covar, svd = {}, {}
#         for key in {**conv_fea_dict, **linear_fea_dict}.keys():
#             covar[key] = 0
#         with torch.no_grad():
#             for _ in range(int(epochs)):
#                 for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
#                     imgs, lbls = imgs.to(self.device), lbls.to(self.device)
#                     if 'input' in fea_dict:
#                         features['input'] = imgs

#                     feats = fea_ext(imgs)
#                     for fea_name in feats:
#                         features[fea_name] = feats[fea_name].detach()

#                     for layer in conv_fea_dict: 
#                         ks = eval(f'model.{layer}').kernel_size
#                         padding = eval(f'model.{layer}').padding
#                         f = features[conv_fea_dict[layer]]
#                         patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
#                         fea_dim = patch.shape[1]
#                         patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
#                         covar[layer] += torch.mm(patch.permute(1, 0), patch)

#                     for layer in linear_fea_dict:
#                         f = features[linear_fea_dict[layer]].double().squeeze()
#                         covar[layer] += torch.mm(f.permute(1, 0), f)
            
#             for layer in covar:
#                 stime = time.time()
#                 U, S, _ = torch.svd(covar[layer]/epochs)
#                 svd[layer] = {'U': U, 'S': torch.sqrt(S)}
#                 print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f}')

#         process_time = time.time() - start_time
#         print(f'Processing time: {process_time:.04f}')

#         print(f"DEBUG: _compute_svd created SVD for layers: {list(svd.keys())}")

#         return svd

#     def _compute_retain_svd(self, full_svd, model, data_loader, conv_fea_dict=None, linear_fea_dict=None, epochs=1):
#         # AUTO-GENERATE FEATURE DICTIONARIES IF NOT PROVIDED
#         if conv_fea_dict is None or linear_fea_dict is None:
#             if hasattr(model, 'conv_fea_dict') and hasattr(model, 'linear_fea_dict'):
#                 # Use model's existing dictionaries
#                 conv_fea_dict = model.conv_fea_dict if conv_fea_dict is None else conv_fea_dict
#                 linear_fea_dict = model.linear_fea_dict if linear_fea_dict is None else linear_fea_dict
#             else:
#                 # Auto-generate dictionaries
#                 conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
        
#         start_time = time.time()
#         fea_dict = {}
#         for val in {**conv_fea_dict, **linear_fea_dict}.values():
#             fea_dict[val] = val

#         max_dim = 0
#         for k in {**conv_fea_dict, **linear_fea_dict}:
#             # FIXED: Use get_module_by_name instead of eval
#             layer_module = self._get_module_by_name(model, k)
#             w = layer_module.weight
#             fea_dim = w.numel() // w.shape[0]
#             max_dim = max(max_dim, fea_dim) 

#         # min_epochs = max(epochs, math.ceil(max_dim/len(data_loader.dataset)))
#         min_epochs = epochs

#         tmp_fea_dict = {**fea_dict}
#         if 'input' in fea_dict:
#             features = {'input': []}
#             tmp_fea_dict.pop('input')
#         else:
#             features = {}

#         fea_ext = create_feature_extractor(model, tmp_fea_dict)

#         covar, retain_svd = {}, {}
#         for key in {**conv_fea_dict, **linear_fea_dict}.keys():
#             covar[key] = 0
#         with torch.no_grad():
#             for _ in range(int(min_epochs)):
#                 for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader)):
#                     imgs = imgs.to(self.device)
#                     lbls = lbls.to(self.device)
#                     if 'input' in fea_dict:
#                         features['input'] = imgs

#                     feats = fea_ext(imgs)
#                     for fea_name in feats:
#                         features[fea_name] = feats[fea_name].detach()

#                     for layer in conv_fea_dict: 
#                         # FIXED: Use get_module_by_name instead of eval
#                         layer_module = self._get_module_by_name(model, layer)
#                         ks = layer_module.kernel_size
#                         padding = layer_module.padding
#                         f = features[conv_fea_dict[layer]]
#                         patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
#                         fea_dim = patch.shape[1]
#                         patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
#                         covar[layer] += torch.mm(patch.permute(1, 0), patch)

#                     for layer in linear_fea_dict:
#                         f = features[linear_fea_dict[layer]].double().squeeze()
#                         covar[layer] += torch.mm(f.permute(1, 0), f)
            
#             process_time = time.time() - start_time
#             print(f'Processing time: {process_time:.04f}')
#             for layer in covar:
#                 stime = time.time()
#                 U, S = full_svd[layer]['U'].to(self.device), full_svd[layer]['S'].to(self.device)
#                 M = torch.mm(torch.mm(U, torch.diag(S**2)), U.t())

#                 M1 = M - covar[layer]/min_epochs
#                 U1_, S1sq_, _ = torch.svd(M1)
#                 retain_svd[layer] = {'U': U1_, 'S': torch.sqrt(S1sq_)}
#                 print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - M: {M.shape}')

#         process_time = time.time() - start_time
#         print(f'Processing time: {process_time:.04f}')

#         print(f"DEBUG: _compute_retain_svd received full_svd with layers: {list(full_svd.keys())}")
#         print(f"DEBUG: _compute_retain_svd expects layers: {list({**conv_fea_dict, **linear_fea_dict}.keys())}")
    
#         return retain_svd
        
#     def _freeze_norm_stats(self, net):
#         try:
#             for m in net.modules():
#                 if isinstance(m, nn.BatchNorm2d):
#                     m.eval()
#                 if isinstance(m, nn.BatchNorm1d):
#                     m.eval()

#         except ValueError:  
#             print("error with BatchNorm")
#             return

#     # def evaluate(model, data_loader, description='', num_forget_classes=None):
#     #     # ce_losses = AverageMeter()
#     #     # accuracies = AverageMeter()
#     #     with torch.no_grad():
#     #         model.eval()
#     #         loss_function = nn.CrossEntropyLoss()
#     #         for batch_idx, (images, labels) in enumerate(data_loader):
#     #             labels = labels.to(self.device)
#     #             images = images.to(self.device)

#     #             outputs = model(images)
#     #             loss = loss_function(outputs, labels)
#     #             # ce_losses.update(loss.item(), images.shape[0])
#     #             if num_forget_classes is not None:
#     #                 labels = labels - num_forget_classes
#     #                 outputs = outputs[:, num_forget_classes:]
#     #             acc = accuracy(outputs, labels)
#     #             # accuracies.update(acc[0].item(), images.shape[0])

#     #         print('[{}] Loss: {:.04f} - Acc: {:.04f}'.format(description, ce_losses.avg, accuracies.avg))
#     #     return ce_losses.avg, accuracies.avg
    

#     def unlearn(self, model: nn.Module, data_dict: Dict[str, DataLoader], **kwargs):
#         model.to(self.device)
#         unlearned_model, scheduler = self._setup_unlearning(model, data_dict, **kwargs)

#         if hasattr(model, 'conv_fea_dict') and hasattr(model, 'linear_fea_dict'):
#             conv_fea_dict = model.conv_fea_dict
#             linear_fea_dict = model.linear_fea_dict
#         else:
#             conv_fea_dict, linear_fea_dict = self._get_feature_dict(model)
#             print(f"Auto-generated conv_fea_dict: {conv_fea_dict}")
#             print(f"Auto-generated linear_fea_dict: {linear_fea_dict}")

#         # Hyperparameters specific to CGS
#         # retained_var: gamma_l threshold for CGS. 
#         self.retained_var_threshold = kwargs.get('retained_var_threshold', 0.95)
#         # lambda_val: Lambda from the loss function. Your `loss2_w` acts as this.
#         self.lambda_val = kwargs.get('lambda_val', 1.0) # Default to positive for entropy maximization
#         # Trust Region Gradient Projection for de-poisoning (last linear layer)
#         self.apply_trgp = kwargs.get('apply_trgp', False)
#         # If apply_trgp is True, we need to know the last linear layer name
#         self.last_linear_layer_name = kwargs.get('last_linear_layer_name', 'fc') # Common name for classifier layer

#         # Ensure required data is available
#         if "forget" not in data_dict.keys():
#             raise ValueError("'forget' data must be in data_dict for CGS unlearning.")
#         if "train" not in data_dict.keys():
#              raise ValueError("'train' (or equivalent full dataset) data must be in data_dict to compute full SVD.")


#         # 1. Compute Full Data Covariance and Forget Covariance
#         full_svd_file = f'exp/svd//svd_full_data.pt'
#         os.makedirs(f'exp/svd/', exist_ok=True)
#         if not os.path.isfile(full_svd_file):
#             self.full_svd = self._compute_svd(
#             model, 
#             data_dict["train"], 
#             epochs=1 # epochs=kwargs.get('svd_full_epochs', 1), # Use more epochs for more accurate SVD
#             )
#             torch.save(self.full_svd, full_svd_file)
#         else:
#             self.full_svd = torch.load(full_svd_file)

#         current_svd = copy.deepcopy(self.full_svd)

#         # Start unlearning
#         unl_start_time = time.time()

#         ### SKIPPING AUGMENTATION CODE###
#         # print('forget_train_noaug')
#         # learn_cfg.dataset.train.params.forget_range = unlearn_step.forget_range
#         # learn_cfg.dataset.train.params.ignore_range = unlearn_step.get('ignore_range', [])
#         # learn_cfg.dataset.train.params.data_section = 'forget'
#         # forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
#         #                                                         **learn_cfg.dataset.train.params)
#         # forget_train_noaug_loader = DataLoader(forget_train_noaug, shuffle=False, num_workers=6, batch_size=128)

#         # forget_train_aug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_train, 
#         #                                                         **learn_cfg.dataset.train.params)
#         # forget_train_aug_loader = DataLoader(forget_train_aug, shuffle=False, num_workers=6, batch_size=128)

#         # learn_cfg.dataset.train.params.forget_range = [[0, unlearn_step.forget_range[0][1]]]*len(unlearn_step.forget_range)
#         # all_forget_train_noaug = eval(learn_cfg.dataset.train.name)(root=learn_cfg.dataset.train.root, transform=transform_test, 
#         #                                                             **learn_cfg.dataset.train.params)
#         # all_forget_train_noaug_loader = DataLoader(all_forget_train_noaug, shuffle=False, num_workers=6, batch_size=128)

#         # print('retain_train_noaug')
#         ###############################

#         # GET RETAIN DATA SVD
#         # retain_svd_file = f'exp/svd/retain_{e}.pt'

#         if not os.path.isfile(full_svd_file):
#             self.full_svd = self._compute_svd(
#                 model, 
#                 data_dict["train"], 
#                 conv_fea_dict=conv_fea_dict,  # Pass the dictionaries
#                 linear_fea_dict=linear_fea_dict,  # Pass the dictionaries
#                 epochs=1
#             )
#             torch.save(self.full_svd, full_svd_file)
#         else:
#             self.full_svd = torch.load(full_svd_file)
        
#         current_svd = copy.deepcopy(self.full_svd)

    
#         # retain_svd = self._compute_retain_svd(current_svd, model, data_dict['forget'], epochs=1)
#         retain_svd = self._compute_retain_svd(
#         current_svd, 
#         model, 
#         data_dict['forget'], 
#         conv_fea_dict=conv_fea_dict,  # Pass the same dictionaries
#         linear_fea_dict=linear_fea_dict,  # Pass the same dictionaries
#         epochs=1
#     )

#         current_svd = retain_svd

#         P = {}
#         for layer in retain_svd:
#             # retained_var += cfg.get('retained_var_step', 0)*unlearn_step_idx
#             # if retained_var == 1.0:
#             #     print(f'Skip layer: {layer}')
#             #     continue
#             # print(f'Retained var: {retained_var:.04f}')
#             k = torch.sum((torch.cumsum(retain_svd[layer]['S'], dim=0) / torch.sum(retain_svd[layer]['S'])) <= self.retained_var)
#             # logger.info(f"{k.item()}\t {retain_svd[layer]['S'].shape[0]}\t {100*k.item()/retain_svd[layer]['S'].shape[0]:.06f}")
            
#             M = retain_svd[layer]['U'][:, :k]
#             P[layer] = torch.mm(M, M.t()).to(self.device).float()

#         offset = 0.05
#         num_bins = 100

#         alpha = np.exp(np.log(self.end_lr/self.start_lr) / self.epochs)
#         base_lr = self.start_lr

#         retain_train_res, forget_train_res, val_res = [], [], []

#         exp_dir = f'{self.work_dir}/results/{self.exp_name}'
#         if os.path.isdir(exp_dir):
#             shutil.rmtree(exp_dir)
#         os.makedirs(exp_dir, exist_ok=True)
#         os.makedirs(f'{exp_dir}/ckp/', exist_ok=True)
#         os.makedirs(f'{exp_dir}/entropy_retrain_unlearn/', exist_ok=True)

#         name_mapping = {}
#         for e in range(self.epochs):
#             epoch_start_time = time.time()
#             # Unlearn Training
#             unlearned_model.train()
#             unlearned_model.apply(self._freeze_norm_stats)
#             for batch_idx, (images, labels) in enumerate(data_dict['forget']):
#                 if e == 0:
#                     break
#                 labels = labels.to(self.device)
#                 images = images.to(self.device)

#                 self.optimizer.zero_grad()
#                 outputs = unlearned_model(images)
                
#                 # loss = loss_function(outputs, labels)
#                 logits = F.softmax(outputs, dim=1)
#                 loss1, loss2 = 0, 0
#                 for j in range(images.shape[0]):
#                     logit = logits[j][labels[j]]
#                     loss1 += -torch.log(1 - ((logit) - offset))
                    
#                 loss2 = -torch.sum(-logits*torch.log(logits + 1e-15), dim=1).mean()
#                 loss1 /= images.shape[0]
#                 loss = self.loss1_w*loss1 + self.loss2_w*loss2

#                 # losses1.update(loss1.item(), images.shape[0])
#                 # losses2.update(loss2.item(), images.shape[0])
#                 # ce_losses.update(loss.item(), images.shape[0])
#                 loss.backward()

#                 with torch.no_grad():
#                     for name, param in unlearned_model.named_parameters():
#                         if name not in name_mapping:
#                             P_name = name
#                             for i in range(20):
#                                 P_name = P_name.replace(f'.{i}', f'[{i}]')
#                             P_name = P_name.replace('.weight', '')
#                             name_mapping[name] = P_name
#                         else:
#                             P_name = name_mapping[name]

#                         if P_name not in P:
#                             param.grad.data.fill_(0)
#                             continue

#                         # print(name, P_name)
#                         sz = param.grad.data.shape[0]
#                         reg_grad = param.grad.data.add(param.data, alpha=self.weight_decay)
#                         reg_grad = reg_grad - torch.mm(reg_grad.view(sz,-1), P[P_name]).view(param.size())
        
#                         lr = base_lr
#                         param.data -= lr * reg_grad
            
#             # Evaluate
#             forward_pass_elapsed = time.time() - epoch_start_time
#             if self.wandb_enabled:
#                 self._log_forward_pass_time_in_wandb(
#                     epoch=e + 1, time=forward_pass_elapsed
#                 )
#             if self.verbose:
#                 self._print_forward_pass_metrics(e, forward_pass_elapsed)
#             if self.evaluate:
#                 self._evaluate_all_splits(
#                     model=unlearned_model,
#                     data_dict=data_dict,
#                     epoch=e + 1,
#                 )
#             self.optimizer.step()
#             if hasattr(self, 'scheduler'):
#                 self.scheduler.step()

#         return unlearned_model, self.logs