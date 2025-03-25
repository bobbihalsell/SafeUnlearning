import torch
import torch.nn as nn
import tqdm
import yaml 

import torch
import torch.nn as nn


from utils import setup_device
from torch_dummy import TorchDummyImage


class DGL:
    """
    Implements the Deep Gradient Leakage attack as described in https://arxiv.org/abs/2309.13016
    and https://github.com/hangyuzhu/privacy-attack-in-federated-learning/tree/main.
    """
    def __init__(
              self,
              original_model: nn.Module,
              original_params: dict,
              unlearned_params: dict
        ):
            """
            Args:
            original_model: The original model before unlearning.
            original_params: The original model parameters before unlearning.
            unlearned_params: The unlearned model parameters.
            """
            # if not isinstance(original_model, nn.Module):
            #     raise TypeError('original_model must be a nn.Module.')

            if not isinstance(original_params, dict):
                raise TypeError("original_params must be a dictionary.")
            if not isinstance(unlearned_params, dict):
                raise TypeError("unlearned_params must be a dictionary.")
            
            self.device = setup_device()  #TODO: this is a copy from unlearning utils
            self.original_model = original_model.to(self.device)
            self.original_params = original_params 
            self.unlearned_params = unlearned_params 
    
    def _gradient_difference(
              self,
              grad_lr = 1e-4):
        
        return [(new - old) / grad_lr for old, new in zip(self.original_params, self.unlearned_params)]
    
    def attack(
            self,
            dummy_data: TorchDummyImage,
            rec_criterion,
            rec_epochs: int,
            rec_lr = 1e-4,
            grad_lr = 1e-4):
        
        self.dummy_data = dummy_data
        self.rec_criterion = rec_criterion
        
        self.dummy_image = self.dummy_data.generate_dummy_input(self.device)
        with torch.no_grad():
            self.dummy_image.clamp_(0, 1)  # In-place clamp without breaking gradients

        self.dummy_label = self.dummy_data.generate_dummy_label(self.device)

        optimizer = torch.optim.AdamW([self.dummy_image, self.dummy_label], lr=rec_lr) # AdamW works the best

        diff_grads = self._gradient_difference(self.original_params, self.unlearned_params, grad_lr)
        diff_grads = [g.detach() for g in diff_grads]

        pbar = tqdm(range(rec_epochs),
                    total=rec_epochs)
    
        for _ in pbar:
            def closure():
                optimizer.zero_grad()
                self.original_model.zero_grad()  

                dummy_pred = self.original_model(self.dummy_image)
                dummy_loss = self.rec_criterion(dummy_pred, self.dummy_label)


                if torch.isnan(dummy_loss).any(): #TODO: better way of dealing with this
                    print("NaN detected in loss!")
                    exit()
            
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.original_model.parameters(), create_graph=True)
                grad_diff = 0
                cos_sim_loss = 0
                pnorm_dummy, pnorm_gt = 0, 0  # Normalization terms for cosine similarity


                for dummy_g, origin_g in zip(dummy_dy_dx, diff_grads):
                    dummy_g = dummy_g / (dummy_g.norm() + 1e-6)  # Normalize dummy gradients
                    origin_g = origin_g / (origin_g.norm() + 1e-6)  # Normalize ground truth gradients
        
                    grad_diff += ((dummy_g - origin_g)**2).sum()
                    cos_sim_loss -= (dummy_g * origin_g).sum()  # Compute the dot product between each pair of gradients (dummy vs. ground truth)
                    pnorm_dummy += (dummy_g ** 2).sum() # Compute the L2 norm (magnitude) of both gradient vectors:
                    pnorm_gt += (origin_g ** 2).sum()

                denominator = (pnorm_dummy.sqrt() * pnorm_gt.sqrt()).clamp(min=1e-6) 
                cos_sim_loss = 1 + cos_sim_loss / denominator   # Compute the cosine similarity loss
                if torch.isnan(cos_sim_loss).any():
                    cos_sim_loss = torch.tensor(0.0, device=cos_sim_loss.device) 


                tv_reg = torch.sum(torch.abs(self.dummy_image[:, :, :-1] - self.dummy_image[:, :, 1:])) + \
                torch.sum(torch.abs(self.dummy_image[:, :-1, :] - self.dummy_image[:, 1:, :]))
                total_loss = grad_diff  + 0.0001 * tv_reg  + 0.0001 * cos_sim_loss 

                total_loss.backward(retain_graph=True)

                return total_loss 
            loss = optimizer.step(closure)
  

            with torch.no_grad():
                self.dummy_image.clamp_(0, 1)  # In-place clamp without breaking gradients
            pbar.set_description("Loss {:.6}".format(loss))

        # save the dummy image
        self.dummy_data.append(self.dummy_image)

        # convert dummy label to integer
        rec_dummy_label = torch.argmax(self.dummy_label, dim=-1)

        # save the dummy label
        self.dummy_data.append_label(rec_dummy_label)

        return dummy_data, rec_dummy_label



if __name__=="__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Access parameters
    params = config["params"]
    num_classes = params["num_classes"]
    lr = params["lr"]
    model_old_path = params["model_old"]
    model_new_path = params["model_new"]
    budget = params["budget"]
    save_path = params["savepath"]

    print(params)

    dgl = DGL(model_old_path)


    # return_image(budget=budget, model=model_new, generator=generator, gt_grads=gt_grads, savepath=save_path)

