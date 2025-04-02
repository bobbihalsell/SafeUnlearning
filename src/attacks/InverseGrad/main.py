
import torch
import torch.nn as nn
import yaml
from attack.reconstruction_algorithms import *
from attack.utils import *

from torchvision.models import resnet18

class InverseGrad:
    """
    Implements the Inverting Gradients attack as described in https://arxiv.org/abs/2003.14053.
    Based on the code from https://github.com/JonasGeiping/invertinggradients/.

    This attack assumes knowledge of ground-truth labels of unlearned samples. 
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
            labels,
            rec_epochs: int,
            rec_lr = 1e-4,
            grad_lr = 1e-4,
            image_shape = (3,32,32),
            image_mean = [0.5, 0.5,0.5],
            image_std = [0.5, 0.5,0.5]
            ):

            labels = torch.as_tensor(labels, device = self.device)

            config = dict(signed=True,
                    boxed=True,
                    cost_fn='sim',
                    indices='def',
                    weights='equal',
                    lr=rec_lr,
                    optim='adamw',
                    restarts=1,
                    max_iterations=rec_epochs,
                    total_variation=1e-1,
                    init='randn',
                    filter='none',
                    lr_decay=True,
                    scoring_choice='loss')
            
            num_images = len(labels)
        
            diff_grads = self._gradient_difference(grad_lr)

            image_mean = torch.as_tensor(image_mean, device = self.device)[:, None, None]
            image_std = torch.as_tensor(image_std, device = self.device)[:, None, None]

            rec_machine = GradientReconstructor(self.original_model, (image_mean, image_std), config=config, num_images=num_images)
            rec_machine.model.eval()
            output, score = rec_machine.reconstruct(diff_grads, labels, img_shape=image_shape)

            return output, score['opt']
    

if __name__=="__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        for params in config["experiments"]:
                exp_name = params["exp_name"]
                seed = params["seed"]  #TODO: 
                original_weights_path = params["original_weights_path"]
                unlearned_weights_path = params["unlearned_weights_path"]
                labels = params["labels"]
                rec_experiments = params["rec_experiments"]
                rec_epochs = params["rec_epochs"]
                rec_lr = params["rec_lr"]
                grad_lr = params["grad_lr"]
                image_shape = params["image_shape"]
                image_mean = params["image_mean"]
                image_std = params["image_std"]
                save_path = params["savepath"]

                
                print(f"Running experiment: {exp_name}")
                print(params)

                set_seed(seed)

                inversegrad = InverseGrad(original_weights_path, unlearned_weights_path)
                for exp_num in range(rec_experiments):
                        output, loss = inversegrad.attack(labels, rec_epochs, rec_lr, grad_lr, 
                                                        image_shape, image_mean, image_std)
                        num_images = len(labels)
                        img = ImageSaver(save_path, num_images)
                        filename = f"inverse_grad_{exp_name}_expNum{exp_num}" # TODO: change this
                        img.save(output, filename, loss, image_mean, image_std)