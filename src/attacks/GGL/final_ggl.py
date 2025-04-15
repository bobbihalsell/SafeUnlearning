import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from pytorch_pretrained_biggan import BigGAN
from torchvision import transforms
from attacks.GGL.turbo import Turbo1
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os
from unlearning.scrub import SCRUB



class GGLReconstructor():
    #def __init__(self, original_model, target_model, generator, loss_fn, unlearning_method, loss_models='l1', num_classes=1000, num_updates=None, lr=0.01, search_dim=128, use_tanh=False, budget=500, alpha=None, gamma=None, min_epochs=None, max_epochs=None, label=None, save_img=None, save_z=None, batch_size=1):
    def __init__(self, original_model, target_model, loss_fn, search_dim=128, use_tanh=False, budget=500, loss_models='l1', unlearning_method='neggrad', num_classes=1000, num_updates=None, lr=0.01,  alpha=None, gamma=None, min_epochs=None, max_epochs=None, labels=None, exp_name='ggl', initial_z=None, batch_size=1, gp_optim="AdamW", use_scheduler=False, initial_lr=1):
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
        generator = BigGAN.from_pretrained("biggan-deep-256")  
        generator.eval()  # Set to evaluation mode
        self.generator = generator
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.num_updates = num_updates
        self.lr = lr
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.budget = budget
        self.type = loss_models
        self.unlearning_method = unlearning_method
        self.alpha = alpha 
        self.gamma = gamma
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.label = labels
        self.exp_name = exp_name
        self.initial_z_path = initial_z
        self.gp_optim = gp_optim
        self.use_scheduler=use_scheduler
        self.initial_lr=initial_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def evaluate_loss(self, z, labels, loss_type=None, steps=None):
        """
        Evaluates the interpolation-based loss for the given latent vector z.
        """
        generated_image = self.generate_image(z, labels)
        

        # Create a copy
        recon = copy.deepcopy(self.original_model)
        recon.eval()

        if self.unlearning_method == 'neggrad':
            # Perform scrub update on recon
            recon = self.perform_sgd_updates(generated_image, labels, recon, steps=1)

        if self.unlearning_method == 'scrub':
            recon = self.perform_scrub_updates(generated_image, labels, recon, alpha=self.alpha, gamma=self.gamma, min_epochs=self.min_epochs, max_epochs=self.max_epochs, lr=self.lr, loss_fn=self.loss_fn, steps=1)

        if loss_type == 'interpolated':
            recon_1 = copy.deepcopy(self.original_model)
            if self.unlearning_method == 'neggrad':
                recon_1 = self.perform_sgd_updates(generated_image, labels, recon_1,  steps=steps)
            if self.unlearning_method == 'scrub':
                recon_1 = self.perform_sgd_updates(generated_image, labels, alpha=self.alpha, gamma=self.gamma, min_epochs=self.min_epochs, max_epochs=self.max_epochs, lr=self.lr, loss_fn=self.loss_fn, steps=steps)

            # Interpolate between original and unlearned model
            interp_model = self.interpolate_models(self.original_model, self.target_model, alpha=0.5)

            # Compute two differences
            loss_1 = self.compute_model_difference_l2(recon_1, interp_model)
            loss_2 = self.compute_model_difference_l2(recon, self.target_model)

            return loss_1+loss_2


        elif loss_type == 'weighted':
            loss = self.compute_model_difference_weighted(recon, self.target_model)
        elif loss_type == 'l2':
            loss = self.compute_model_difference_l2(recon, self.target_model)
        elif loss_type == 'l1':
            loss = self.compute_model_difference_l1(recon, self.target_model)
        
        return loss

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
    
    def perform_scrub_updates(self, generated_image, labels, original_model, alpha, gamma, min_epochs, max_epochs, lr, loss_fn, verbose=True, steps=None):
        scrub = SCRUB(device=self.device)

        # Convert label to tensor if necessary
        if isinstance(labels, int):
            labels = torch.tensor([labels]).long().to(self.device)
        else:
            labels = labels.to(self.device)

        # Package the generated image and label as a dictionary
        forget_dataset = torch.utils.data.TensorDataset(generated_image, labels)
        forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=len(forget_dataset), shuffle=False)

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
            data_dict=forget_loader,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            verbose=verbose,
            **kwargs
        )

        return unlearned_model

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


    def reconstruct(self, initial_z=None):
        """
        Optimize the latent vector z using Turbo Bayesian Optimization.
        If `initial_z` is provided, it starts from there instead of a random initialization.
        """
        label = self.label[0] #TODO: change for multiple instances

        # Set the directory for saving results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        # Build the path to the artifacts folder
        results_dir = os.path.join(project_root, "artifacts", "reconstructed", "GGL", "results_report")
        # Ensure the directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Build the name of the files
        file_name = f"{self.exp_name}_labels{label}_{self.unlearning_method}_lr{self.lr}_updates{self.num_updates}_budget{self.budget}_loss{self.type}_BO_seed42_gp{self.gp_optim}_{self.initial_lr}_scheduler{self.use_scheduler}"
        x_path = os.path.join(results_dir, file_name + ".png")
        z_path = os.path.join(results_dir, file_name)


        labels = torch.tensor([label])  # Assign a label
        f = lambda z: self.evaluate_loss(z, labels, loss_type=self.type)  # Define the objective function
        labels = labels.to(self.device)

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
            gp_optim=self.gp_optim,
            use_scheduler=self.use_scheduler,
            initial_lr=self.initial_lr
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

    