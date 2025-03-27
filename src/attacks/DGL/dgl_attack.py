import torch
import torch.nn as nn
from tqdm import tqdm
import yaml 

from torchvision.models import resnet18


import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn


from utils import setup_device, ImageSaver
from torch_dummy import TorchDummyImage


class DGL:
    """
    Implements the Deep Gradient Leakage attack as described in https://arxiv.org/abs/2309.13016
    and https://github.com/hangyuzhu/privacy-attack-in-federated-learning/tree/main.
    """
    def __init__(
              self,
              original_weights: dict,
              unlearned_weights: dict
        ):
            """
            Args:
            original_model: The original model before unlearning.
            original_params: The original model parameters before unlearning.
            unlearned_params: The unlearned model parameters.
            """

            ## MY RESNET
            self.original_model = resnet18(weights=None)
            self.original_model.fc = nn.Linear(512, 10)  
            self.original_model.load_state_dict(torch.load(original_weights))

            self.unlearned_model = resnet18(weights=None)
            self.unlearned_model.fc = nn.Linear(512, 10)  
            self.unlearned_model.load_state_dict(torch.load(unlearned_weights))

            self.device = setup_device()  #TODO: this is a copy from unlearning utils
            self.original_model = self.original_model.to(self.device)
            self.unlearned_model = self.unlearned_model.to(self.device)
    
    def _gradient_difference(
              self,
              grad_lr = 1e-4):
              param_old = [p.clone().detach() for p in self.original_model.parameters()]
              param_new = [p.clone().detach() for p in self.unlearned_model.parameters()]
              return [(new.detach() - old.detach()) / grad_lr for old, new in zip(param_old, param_new)]

    
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

        optimizer = torch.optim.AdamW([self.dummy_image, self.dummy_label], lr=rec_lr)
        diff_grads = self._gradient_difference(grad_lr)

        self.original_model.eval()

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
                    dummy_g = dummy_g / (dummy_g.norm() + 1e-4)  # Normalize dummy gradients
                    origin_g = origin_g / (origin_g.norm() + 1e-4)  # Normalize ground truth gradients
                    grad_diff += ((dummy_g - origin_g)**2).sum()


                    pnorm_dummy += (dummy_g ** 2).sum() # Compute the L2 norm of both gradient vectors:
                    pnorm_gt += (origin_g ** 2).sum()

                denominator = (pnorm_dummy.sqrt() * pnorm_gt.sqrt()).clamp(min=1e-6) 
                cos_sim_loss = 1 + cos_sim_loss / denominator   # Compute the cosine similarity loss
                if torch.isnan(cos_sim_loss).any():
                    cos_sim_loss = torch.tensor(0.0, device=cos_sim_loss.device) 


                tv_reg = torch.sum(torch.abs(self.dummy_image[:, :, :-1] - self.dummy_image[:, :, 1:])) + \
                torch.sum(torch.abs(self.dummy_image[:, :-1, :] - self.dummy_image[:, 1:, :]))
                total_loss = grad_diff  + 0.0001 * tv_reg  + 0.0001 * cos_sim_loss 
                total_loss = grad_diff 
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

        return dummy_data, rec_dummy_label, loss



if __name__=="__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Access parameters
    # params = config["params"]
    # exp_name = params["exp_name"]
    # num_classes = params["num_classes"]
    # original_weights_path = params["original_weights_path"]
    # unlearned_weights_path = params["unlearned_weights_path"]
    # rec_batch_size = params["rec_batch_size"]
    # rec_experiments = params["rec_experiments"]
    # rec_epochs = params["rec_epochs"]
    # rec_lr = params["rec_lr"]
    # grad_lr = params["grad_lr"]
    # image_shape = params["image_shape"]
    # normalise = params["normalise"]
    # image_mean = params["image_mean"]
    # image_std = params["image_std"]

    # save_path = params["savepath"]

    # print(params)

    for params in config["experiments"]:
        exp_name = params["exp_name"]
        num_classes = params["num_classes"]
        original_weights_path = params["original_weights_path"]
        unlearned_weights_path = params["unlearned_weights_path"]
        rec_batch_size = params["rec_batch_size"]
        rec_experiments = params["rec_experiments"]
        rec_epochs = params["rec_epochs"]
        rec_lr = params["rec_lr"]
        grad_lr = params["grad_lr"]
        image_shape = params["image_shape"]
        normalise = params["normalise"]
        image_mean = params["image_mean"]
        image_std = params["image_std"]
        save_path = params["savepath"]

        print(f"Running experiment: {exp_name}")
        print(params)

        dgl = DGL(original_weights_path, unlearned_weights_path)

        dummy = TorchDummyImage(
        image_shape=image_shape,
        batch_size=rec_batch_size,
        n_classes=num_classes,
        normalize=normalise,
        dm=image_mean,
        ds=image_std,
        device='cuda') #TODO remove this
        
        rec_criterion = nn.CrossEntropyLoss()  #TODO
        losses = []
        for i in range(rec_experiments):
            _, _, loss = dgl.attack(dummy, rec_criterion, rec_epochs, rec_lr, grad_lr)
            losses.append(float(loss.detach()))


        images = []
        images += dummy.history
        max_images = rec_experiments * rec_batch_size
        imgs = ImageSaver(save_path, max_images= max_images, num_cols=rec_batch_size) 

        imgs.save(images, exp_name, losses)


