import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from pytorch_pretrained_biggan import BigGAN, BigGANConfig
from torch.autograd import Variable
import torchvision.models as models
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from attacks.GGL.turbo import Turbo1
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torchvision

from unlearning.scrub import SCRUB

#import save_results
#import SCRUB
#import get_transform
#import set_seed


def get_transform(dataset_name):
    if dataset_name.lower() == "imagenet":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ModelDiffReconstructorBO():
    def __init__(self, original_model, target_model, generator, loss_fn, unlearning_method, type='l1', num_classes=1000, num_updates=None, lr=0.01, search_dim=128, use_tanh=False, budget=500, alpha=None, gamma=None, min_epochs=None, max_epochs=None):
        """
        original_model: The original model (pre-update).
        target_model: The target model (unlearned model).
        generator: The BigGAN generator.
        loss_fn: The loss function to compute the loss on model output.
        num_classes: The number of classes for classification.
        num_updates: The number of SGD updates to perform on the original model using the generated image.
        lr: Learning rate for SGD updates.
        search_dim: The dimension of the latent space (size of z).
        use_tanh: Whether to apply tanh activation to the latent vector z.
        budget: The maximum number of evaluations for Bayesian Optimization.
        """
        self.original_model = original_model
        self.target_model = target_model
        self.generator = generator
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.num_updates = num_updates
        self.lr = lr
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.budget = budget
        self.type = type
        self.unlearning_method = unlearning_method
        self.alpha = alpha 
        self.gamma = gamma
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def evaluate_loss(self, z, labels, type=None, steps=None):
        """
        Evaluates the interpolation-based loss for the given latent vector z.
        """
        generated_image = self.generate_image(z, labels)
        # Create a copy
        recon = copy.deepcopy(self.original_model)

        if self.unlearning_method == 'neggrad':
            # Perform scrub update on recon
            recon = self.perform_sgd_updates(generated_image, labels, recon, alpha=self.alpha, gamma=self.gamma, min_epochs=self.min_epochs, max_epochs=self.max_epochs, lr=self.lr, loss_fn=self.loss_fn, steps=1)

        if self.unlearning_method == 'scrub':
            recon = self.perform_scrub_updates(generated_image, labels, recon, steps=1)

        if type == 'interpolated':
            recon_1 = copy.deepcopy(self.original_model)
            if self.unlearning_method == 'neggrad':
                recon_1 = self.perform_sgd_updates(generated_image, labels, recon_1, alpha=self.alpha, gamma=self.gamma, min_epochs=self.min_epochs, max_epochs=self.max_epochs, lr=self.lr, loss_fn=self.loss_fn, steps=steps)
            if self.unlearning_method == 'scrub':
                recon_1 = self.perform_sgd_updates(generated_image, labels, recon_1, steps=steps)

            # Interpolate between original and unlearned model
            interp_model = self.interpolate_models(self.original_model, self.target_model, alpha=0.5)

            # Compute two differences
            loss_1 = self.compute_model_difference_l2(recon_1, interp_model)
            loss_2 = self.compute_model_difference_l2(recon, self.target_model)

            return loss_1+loss_2


        elif type == 'weighted':
            loss = self.compute_model_difference_weighted(recon, self.target_model)
        elif type == 'l2':
            loss = self.compute_model_difference_l2(recon, self.target_model)
        elif type == 'l1':
            loss = self.compute_model_difference_l1(recon, self.target_model)
        
        return loss

    def perform_scrub_updates(self, generated_image, labels, original_model, alpha, gamma, min_epochs, max_epochs, lr, loss_fn, verbose=True, steps=None):
        scrub = SCRUB(device=self.device)

        # Convert label to tensor if necessary
        if isinstance(labels, int):
            labels = torch.tensor([labels]).long().to(self.device)
        else:
            labels = labels.to(self.device)

        # Package the generated image and label as a dictionary
        data_dict = {
            'forget': [(generated_image, labels)]  # Provide a list of tuples (image, label)
        }

        # Hardcode any additional arguments needed for unlearn method
        kwargs = {
            'alpha': alpha,
            'gamma': gamma,
            'loss_fn': loss_fn,
            'lr': lr,
            'min_epochs': min_epochs,
            'max_epochs' : max_epochs

        }

        # Run the unlearning process
        unlearned_model, losses = scrub.unlearn(
            model=original_model,
            data_dict=data_dict,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            verbose=verbose,
            **kwargs
        )

        return unlearned_model

    def interpolate_models(self, model1, model2, alpha=0.5):
        """
        Linearly interpolate between two models: model1 and model2.
        Returns a new model: alpha * model1 + (1 - alpha) * model2
        """
        interp_model = copy.deepcopy(model1)
        for (name1, param1), (name2, param2), (_, param_interp) in zip(model1.named_parameters(), model2.named_parameters(), interp_model.named_parameters()):
            assert name1 == name2
            param_interp.data = alpha * param1.data + (1 - alpha) * param2.data
        return interp_model

    def compute_model_difference_weighted(self, model1, model2):
        """
        Compute a layer-weighted difference between two models.
        Later layers get higher weights.
        """
        diff = 0.0
        total_layers = len(list(model1.parameters()))
        for i, (p1, p2) in enumerate(zip(model1.parameters(), model2.parameters())):
            weight = (i + 1) / total_layers  # gradually increases for later layers
            diff += weight * torch.sum((p1 - p2) ** 2)  # L2 norm works well here
        return diff.item()
    
    def compute_model_difference_l2(self, model1, model2):
        """
        Compute the difference between two models' parameters.
        """
        diff = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff += torch.sum(abs(p1 - p2)**2)  # L1 norm of the difference
        return diff.item()

    def compute_model_difference_l1(self, model1, model2):
        """
        Compute the difference between two models' parameters.
        """
        diff = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff += torch.sum(abs(p1 - p2))  # L1 norm of the difference
        return diff.item()


    
    def generate_image(self, z, labels):
        """
        Generate an image using the BigGAN generator from the latent vector z.
        """
        # BigGAN requires input noise z and labels
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)

        if z.dim() > 2:
            z = z.squeeze(0)

        if z.dim() == 1:
            z = z.unsqueeze(0)

        z = z.to(self.device)

        noise_vector = z  
        self.generator = self.generator.to(self.device)  # Ensure generator is on correct device
        #noise_vector = z.to(self.device)
        
        # Ensure labels is a tensor
        if isinstance(labels, int):
            label_tensor = torch.tensor([labels]).long().to(self.device)
        else:
            label_tensor = labels.long().to(self.device)
        
        # Convert labels to one-hot encoding for BigGAN
        c = torch.nn.functional.one_hot(label_tensor, num_classes=self.num_classes).float().to(self.device)

        # Use BigGAN to generate an image
        with torch.no_grad():
            generated_image = self.generator(noise_vector, c, 1)  # 1 is the truncation value

        # Rescale image to 224x224 
        generated_image = nn.functional.interpolate(generated_image, size=(224, 224), mode='area')
        
        return generated_image


    def perform_sgd_updates(self, generated_image, labels, updated_model_attacker, steps=None):
        if steps is None:
            steps = self.num_updates
        optimizer = optim.SGD(updated_model_attacker.parameters(), lr=0.001)
        for _ in range(steps):
            optimizer.zero_grad()
            pred = updated_model_attacker(generated_image)
            loss = self.loss_fn(pred, labels)
            (-loss).backward()
            optimizer.step()
        return updated_model_attacker


    def optimize_latent_vector(self, labels, initial_z=None, x_path=None, z_path=None):
        """
        Optimize the latent vector z using Turbo Bayesian Optimization.
        If `initial_z` is provided, it starts from there instead of a random initialization.
        """
        f = lambda z: self.evaluate_loss(z, labels, type=self.type)  # Define the objective function

        # Define search space
        z_lb = -2 * np.ones(self.search_dim)
        z_ub = 2 * np.ones(self.search_dim)

        # Initialize Turbo optimizer
        device_str = "cuda" if self.device.type == "cuda" else "cpu"

        self.optimizer = Turbo1(
            f=f,
            lb=z_lb,
            ub=z_ub,
            n_init=256,  # Default initial evaluations
            max_evals=self.budget,
            batch_size=10,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device=device_str,
            dtype="float32",
        )

        # If an initial z is provided, use it instead of a random start
        if initial_z is not None:
            initial_z = initial_z.cpu().numpy().reshape(1, -1)  # Ensure correct format
            

            # Evaluate the function at the initial point
            initial_loss = f(initial_z)

            # Override Turbo's initial data
            self.optimizer.X = initial_z  # Set the initial point
            self.optimizer.fX = np.array([initial_loss])  # Set the corresponding loss

            # Reduce number of initial random evaluations 
            self.optimizer.n_init = max(0, self.optimizer.n_init - 1)  

        # Run the optimization
        self.optimizer.optimize()

        # Extract results
        X = self.optimizer.X
        fX = self.optimizer.fX
        ind_best = np.argmin(fX)
        loss_res, z_res = fX[ind_best], X[ind_best, :]

        # Convert best z to tensor
        z_res = torch.from_numpy(z_res).unsqueeze(0).to(self.device)
        if self.use_tanh:
            z_res = torch.tanh(z_res)

        # Generate final image
        label_tensor = torch.tensor([labels]).long().to(self.device)
        c = torch.nn.functional.one_hot(label_tensor, num_classes=self.num_classes).float().to(self.device)

        with torch.no_grad():
            x_res = self.generator(z_res.float(), c, 1)

        x_res = nn.functional.interpolate(x_res, size=(224, 224), mode='area')
        if x_path and z_path:
            self.save_results(z_res, x_res, z_path, x_path)
        return z_res, x_res, loss_res
    
    def save_results(self, z, x, z_path, x_path):
        # Ensure x is on CPU and properly scaled from [-1, 1] to [0, 1]
        x = x.detach().cpu()
        x = (x.squeeze(0) + 1) / 2.0  # Normalize from [-1, 1] to [0, 1]

        # Convert tensor to PIL image
        transform = transforms.ToPILImage()
        img_pil = transform(x.clamp(0, 1))  # Clamp to [0, 1] range
        img_pil.save(x_path)  # Save the image to specified path
        print(f"Image saved to {z_path}")

        # Convert z to numpy and save
        if isinstance(z, torch.Tensor):
            z_np = z.detach().cpu().numpy()
        else:
            z_np = np.array(z)
        
        np.save(z_path, z_np)
        print(f"Latent vector z saved to {x_path}.npy")


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_label(label):
    labels = torch.tensor([label])  # Assign a label
    labels.to(device)
    if isinstance(labels, int):
        labels = torch.tensor([labels]).long().to(device)
    else:
        labels = labels.long().to(device)

    return labels


if __name__ == "__main__":
    # Load the configuration from YAML
    config = load_config("config_ggl.yaml")
    print('yaml file loaded')
    
    seed = config['Unlearner']['seed']
    set_seed(seed)
    device = config['Reconstructor']['device']
    data_root = config['Reconstructor']['data_root']
    dataset_name = config['Reconstructor']['dataset_name']
    transform = get_transform(dataset_name)
    forget_set = datasets.ImageFolder(root=data_root, transform=transform)
    batch_size = config['Reconstructor']['batch_size']
    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=True)
    #generated_image = "/vol/bitbucket/oap24/final_project/safe-unlearning/src/unlearning/test_samples3/n00002357/ILSVRC2012_val_00002357.JPEG"
    generated_image, labels = forget_set[0]
    generated_image.to(device)
    generated_image = torch.tensor(generated_image,device="cuda:0").unsqueeze(0)
    # Perform SGD updates on the original model
    label = config['Reconstructor']['label']


    model_name = config['Unlearner']['model_name']
    
    label = process_label(label)
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)
    # Optimizer
    model.eval()

    updated_model_unlearner = copy.deepcopy(model)

    unlearning_method = config['Unlearner']['name']
    unlearning_path = config['Unlearner']['unlearning_path']
    # Load the state dictionary from the file
    state_dict = torch.load(unlearning_path)
    # Load the state dictionary into the model
    updated_model_unlearner.load_state_dict(state_dict)
    #unlearned = perform_unlearning(updated_model_unlearner, generated_image, labels)
    generator = BigGAN.from_pretrained("biggan-deep-256")  
    generator.eval()  # Set to evaluation mode

        
    x_path = config['Reconstructor']['save_img']
    z_path = config['Reconstructor']['save_z']

    #loss_fn = config['loss']
    loss_fn = nn.CrossEntropyLoss()
    loss_models = config['Reconstructor']['loss_models']
    num_updates = config['Unlearner']['cfg']['epochs']
    lr = config['Unlearner']['cfg']["lr"]
    budget = config['Reconstructor']['budget']
    num_classes = config['Unlearner']['num_classes']
    search_dim = config['Reconstructor']['search_dim']
    use_tanh = config['Reconstructor']['use_tanh']

    print('yaml file read')

    # parameters specific to scrub
    alpha = config['Unlearner'].get('alpha', None)
    gamma = config['Unlearner'].get('gamma', None)
    min_epochs = config['Unlearner'].get('min_epochs', None)
    max_epochs = config['Unlearner'].get('max_epochs', None)



    rec = ModelDiffReconstructorBO(original_model=model, target_model=updated_model_unlearner, generator=generator, loss_fn=loss_fn, type=loss_models, num_classes=num_classes, num_updates=num_updates, lr=lr, search_dim=search_dim, use_tanh=use_tanh, budget=budget, unlearning_method=unlearning_method, alpha=alpha, gamma=gamma, min_epochs=min_epochs, max_epochs=max_epochs)
    z_res, x_res, loss_res = rec.optimize_latent_vector(labels=label, x_path=x_path, z_path=z_path) 