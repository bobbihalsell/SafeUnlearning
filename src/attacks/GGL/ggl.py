import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import copy
from tqdm import tqdm
import nevergrad as ng
import lpips
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import nevergrad as ng
import importlib
import reconstructor
importlib.reload(reconstructor)
from reconstructor import NGReconstructor, BOReconstructor
import yaml
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images)

device = 'cuda'

setup = dict(device=device, dtype=torch.float)
dm = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]
ds = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]

generator= BigGAN.from_pretrained('biggan-deep-256').to(setup['device'])


def plot(tensor, save_path=None):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)

    plt.figure(figsize=(6, 6))  # Create new figure

    if tensor.shape[0] == 1:
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()  




def calculate_grads(num_classes=1000, dim=1280, lr=0.01, unlearned_model_path="../../../unlearning/gascent/mobilenet_unlearned_grad_ascent_1.pth"):
    # Initialize two MobileNetV2 models
    model_old = models.resnet18(pretrained=True)
    model_new = models.resnet18(pretrained=True)

    # Modify the classifier to match the checkpoint
    #model_old.classifier[1] = torch.nn.Linear(dim, num_classes)
    #model_new.classifier[1] = torch.nn.Linear(dim, num_classes)

    # Move models to the selected device
    model_old = model_old.to(device)
    model_new = model_new.to(device)

    # Print model architecture 
    print(model_old)

    # Load correct MobileNetV2 checkpoints
    #model_old.load_state_dict(torch.load(model_old_path, map_location=device)['model_state_dict'])
    #model_new.load_state_dict(torch.load("/vol/bitbucket/oap24/sandbox-machine-unlearning/src/sandbox/unlearning/SCRUB/mobilenet_unlearned_gascent.pth", map_location=device))
    model_new.load_state_dict(torch.load(unlearned_model_path, map_location=device))

    # Extract model parameters before and after unlearning
    param_old = [p.clone().detach() for p in model_old.parameters()]
    param_new = [p.clone().detach() for p in model_new.parameters()]

    # Compute gradients 
    gt_grads = [(new - old) / lr for old, new in zip(param_old, param_new)]

    return gt_grads, model_new, model_old


def return_image(budget, model, generator, gt_grads, savepath, reconstructor, strategy):
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 reduce=None, reduction='mean')
    model.eval()

    ng_rec = reconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                        num_classes=1000, search_dim=(128,), strategy=strategy, budget=budget, use_tanh=True, defense_setting=None)
    z_res, x_res, img_res_old, loss_res = ng_rec.reconstruct(gt_grads)

    plot(x_res, savepath)

if __name__=="__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Access parameters
    params = config["params"]
    num_classes = params["num_classes"]
    lr = params["lr"]
    #model_old_path = params["model_old"]
    model_new_path = params["model_new"]
    budget = params["budget"]
    save_path = params["savepath"]
    reconstr = params["reconstructor"]
    strategy = params["strategy"]

    if reconstr == "NGReconstructor":
        reconstr = NGReconstructor
    elif reconstr == "BOReconstructor":
        reconstr = BOReconstructor


    print(params)

    gt_grads, model_new, model_old = calculate_grads(num_classes, dim=1280, lr=lr, unlearned_model_path=model_new_path)

    return_image(budget=budget, model=model_new, generator=generator, gt_grads=gt_grads, savepath=save_path, reconstructor=reconstr, strategy=strategy)



# params:
# num classes
# model old 
# model new - eg: artifacts/unlearned/{model}_{method}_{epoch}.pth
# lr
# budget
# savepath
