import copy
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_biggan import BigGAN
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets

from attacks.GGL.turbo import Turbo1
from attacks.GGL.projection import privacy_attack_without_forget_data

from unlearning.neggrad import NegGradPlus
from unlearning.scrub import SCRUB

class GGLReconstructor:
    """
    Implements a Generative Gradient Leackage attack as introduced in
    "Auditing Privacy Defenses in Federated Learning
    via Generative Gradient Leakage"(https://arxiv.org/pdf/2203.15696)

    This class reconstructs unlearned images using the unlearned model,
    original model and the label of unlearned image. It leverages
    BigGAN to use the prior information about the label in order
    to generate reconstructions.

    It utilises Turbo Bayesian Optimisation in order to find an image that
    produces the same changes to the original model as the ones in
    the unlearned model by mimicking unlearning process.
    """

    def __init__(
        self,
        original_model,
        target_model,
        loss_fn,
        loss_models="l1",
        unlearning_method="neggrad",
        num_classes=1000,
        num_updates=None,
        lr=0.001,
        labels=None,
        exp_name="ggl",
        initial_z_path=None,
        batch_size=1,
        gp_optim="AdamW",
        use_scheduler=False,
        initial_lr=1,
        search_dim=128,
        use_tanh=False,
        budget=500,
    ):
        """
        Initialize the GGLReconstructor object.

        Args:
            original_model: The original model (pre-update).
            target_model: The target model (unlearned model).
            loss_fn: The loss function to compute the loss on model output.
            loss_models: The loss for the objective function of optimization
                process.
            unlearning_method: Method used for unlearning (scrub or neggrad).
            num_classes: Number of classes in classification.
            num_updates: The number of SGD updates to performed during
                unlearning.
            lr: Learning rate for SGD updates during unlearning.
            labels: Label of the forgotten image.
            exp_name: Cusomisable exp_name for saving results.
            initial_z_path: Path where inital latent vector z should be loaded
                from (might be None).
            batch_size: Batch size.
            gp_optim: Optimiser for Gaussian Process in Turbo
            use_scheduler: Whether to use scheduler during Gaussian Process
            initial_lr: Initial lr for Gaussian Process optimiser updates
            search_dim: The dimension of the latent space (size of z).
            use_tanh: Whether to apply tanh activation to the latent vector z.
            budget: The maximum number of evaluations for Bayesian
                Optimization.
        """
        self.original_model = original_model
        self.target_model = target_model
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.num_updates = num_updates
        self.lr = lr
        self.batch_size = batch_size
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.budget = budget
        self.type = loss_models
        self.unlearning_method = unlearning_method
        self.label = labels
        self.exp_name = exp_name
        self.initial_z_path = initial_z_path
        self.gp_optim = gp_optim
        self.use_scheduler = use_scheduler
        self.initial_lr = initial_lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = BigGAN.from_pretrained("biggan-deep-256")
        generator.eval()  # Set to evaluation mode
        self.generator = generator
        # data_path = "/vol/bitbucket/vb524/imagenet_subset_val/Small-ImageNet-Validation-Dataset-1000-Classes/ILSVRC2012_img_val_subset"
        bdata = datasets.ImageFolder(
            data_path,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            ),
        )
        bdata_loader = DataLoader(bdata, batch_size=32, shuffle=False)
        results = privacy_attack_without_forget_data(
        self.original_model, self.target_model, bdata_loader, self.device, K=50)   # This projects the target_model

    def evaluate_loss(self, z, labels, loss_type=None, steps=None, **kwargs):
        """
        Evaluates thee loss for the given latent vector z.
        """
        generated_image = self.generate_image(z, labels)
        # Create a copy
        recon = copy.deepcopy(self.original_model)
        recon.eval()

        if self.unlearning_method == "neggrad":
            # Perform neggrad updates on recon
            recon = self.perform_sgd_updates(
                generated_image, labels, recon, steps=steps
            )

        if self.unlearning_method == "scrub":
            # Perform scrub updates on recon
            recon = self.perform_scrub_updates(generated_image, labels, recon, **kwargs)

        if self.unlearning_method == "neggradplus":
            # Perform neggradplus updates on recon
            recon = self.perform_neggradplus_updates(
                generated_image, labels, recon, **kwargs
            )

        if loss_type == "interpolated":
            # Calculate interpolated loss
            recon_1 = copy.deepcopy(self.original_model)
            recon_1.eval()
            if self.unlearning_method == "neggrad":
                recon_1 = self.perform_sgd_updates(
                    generated_image, labels, recon_1, steps=1
                )

            if self.unlearning_method == "scrub":
                steps = kwargs["max_epochs"]
                kwargs["max_epochs"] = 1
                recon_1 = self.perform_scrub_updates(
                    generated_image, labels, recon_1, **kwargs
                )
                kwargs["max_epochs"] = steps

            if self.unlearning_method == "neggradplus":
                steps = kwargs["epochs"]
                kwargs["epochs"] = 1
                recon_1 = self.perform_neggradplus_updates(
                    generated_image, labels, recon_1, **kwargs
                )
                kwargs["epochs"] = steps
            # Interpolate between original and unlearned model
            interp_model = self.interpolate_models(
                self.original_model, self.target_model, alpha=0.5
            )

            # Compute two differences
            loss_1 = self.compute_model_difference_l2(recon_1, interp_model)
            loss_2 = self.compute_model_difference_l2(recon, self.target_model)

            return loss_1 + loss_2

        # Compute difference between model from reconstruction and unlearned
        elif loss_type == "weighted":
            loss = self.compute_model_difference_weighted(recon, self.target_model)
        elif loss_type == "l2": 
            loss = self.compute_model_difference_l2(recon, self.target_model)
        elif loss_type == "l1":
            loss = self.compute_model_difference_l1(recon, self.target_model)
        return loss

    def interpolate_models(self, model1, model2, alpha=0.5):
        """
        Linearly interpolate between two models: model1 and model2.
        Returns a new model: alpha * model1 + (1 - alpha) * model2
        """
        interp_model = copy.deepcopy(model1)

        for (name1, param1), (name2, param2), (_, param_interp) in zip(
            model1.named_parameters(),
            model2.named_parameters(),
            interp_model.named_parameters(),
        ):
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
            weight = (i + 1) / total_layers  # Gradually increases
            diff += weight * torch.sum((p1 - p2) ** 2)  # Weighted L2 norm

        return diff.item()

    def compute_model_difference_l2(self, model1, model2):
        """
        Compute the difference between two models' parameters.
        """
        diff = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff += torch.sum(abs(p1 - p2) ** 2)  # L2 norm of the difference
        return diff.item()

    def compute_model_difference_l1(self, model1, model2):
        """
        Compute the difference between two models' parameters.
        """
        diff = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff += torch.sum(abs(p1 - p2))  # L1 norm of the difference
        return diff.item()

    def perform_scrub_updates(
        self, generated_image, labels, original_model, verbose=False, **kwargs
    ):
        """
        Perform scrub updates to mimic the unlearning process.
        """
        scrub = SCRUB(device=self.device)

        # Convert label to tensor if necessary
        if isinstance(labels, int):
            labels = torch.tensor([labels]).long().to(self.device)
        else:
            labels = labels.to(self.device)

        # Create forget loader
        forget_dataset = torch.utils.data.TensorDataset(generated_image, labels)

        forget_loader = torch.utils.data.DataLoader(
            forget_dataset, batch_size=len(forget_dataset), shuffle=False
        )
        # Create forget dictionary
        forget_dict = {"forget": forget_loader}

        empty_x = torch.empty((0, 3, 224, 224))
        empty_labels = torch.empty((0,), dtype=torch.long)
        empty_dataset = TensorDataset(empty_x, empty_labels)
        # Create retain dictionary; since we don't have access to retain set
        # we handle that with empty dataset
        forget_dict["retain"] = DataLoader(empty_dataset)

        # Mimic the unlearning process
        unlearned_model, _ = scrub.unlearn(
            model=original_model,
            data_dict=forget_dict,
            verbose=verbose,
            lr=self.lr,
            **kwargs,
        )

        return unlearned_model

    def perform_neggradplus_updates(
        self, generated_image, labels, original_model, verbose=False, **kwargs
    ):
        """
        Perform neggradplus updates to mimic the unlearning process.
        """
        neggradplus = NegGradPlus(device=self.device)

        # Convert label to tensor if necessary
        if isinstance(labels, int):
            labels = torch.tensor([labels]).long().to(self.device)
        else:
            labels = labels.to(self.device)

        # Create forget loader
        forget_dataset = torch.utils.data.TensorDataset(generated_image, labels)

        forget_loader = torch.utils.data.DataLoader(
            forget_dataset, batch_size=len(forget_dataset), shuffle=False
        )
        # Create forget dictionary
        forget_dict = {"forget": forget_loader}

        empty_x = torch.empty((0, 3, 224, 224))
        empty_labels = torch.empty((0,), dtype=torch.long)
        empty_dataset = TensorDataset(empty_x, empty_labels)
        # Create retain dictionary since we don't have access to retain set
        # we handle that with empty dataset
        forget_dict["retain"] = DataLoader(empty_dataset)

        # Mimic the unlearning process
        unlearned_model, _ = neggradplus.unlearn(
            model=original_model,
            data_dict=forget_dict,
            verbose=verbose,
            lr=self.lr,
            **kwargs,
        )

        return unlearned_model

    def perform_sgd_updates(
        self, generated_image, labels, updated_model_attacker, steps=None
    ):
        """
        Perform updates in neggrad to mimic the neggrad unlearning process.
        """
        if steps is None:
            steps = self.num_updates
        optimizer = optim.SGD(updated_model_attacker.parameters(), lr=self.lr)
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
        self.generator = self.generator.to(self.device)
        # Ensure labels is a tensor
        if isinstance(labels, int):
            label_tensor = torch.tensor([labels]).long().to(self.device)
        else:
            label_tensor = labels.long().to(self.device)
        # Convert labels to one-hot encoding for BigGAN
        c = (
            torch.nn.functional.one_hot(label_tensor, num_classes=self.num_classes)
            .float()
            .to(self.device)
        )

        # Use BigGAN to generate an image
        with torch.no_grad():
            noise_vector = noise_vector.to(dtype=torch.float32)
            c = c.to(dtype=torch.float32)
            generated_image = self.generator(noise_vector, c, 1)

        # Rescale image to 224x224
        generated_image = nn.functional.interpolate(
            generated_image, size=(224, 224), mode="area"
        )

        return generated_image

    def reconstruct(self, **kwargs):
        """
        Optimize the latent vector `z` using Turbo Bayesian Optimization.

        If `initial_z` is provided, the optimization will start from that
        initial latent vector instead of a random initialization.

        This method leverages Bayesian optimization to iteratively adjust
        the latent vector `z` to minimize the loss function.
        """

        if self.initial_z_path is not None:
            z = np.load(self.initial_z_path)
            initial_z = torch.from_numpy(z).float().to(self.device)
        else:
            initial_z = None

        # It is always assumed that reconstruction is for a single image
        label = self.label[0]

        # Set up the directory for saving results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        # Build the path to the artifacts folder
        results_dir = os.path.join(
            project_root, "artifacts", "reconstructed", "GGL", "results_report"
        )
        # Ensure the directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Build the name of the files
        file_name = (
            f"{self.exp_name}_labels{label}_{self.unlearning_method}_"
            f"lr{self.lr}_updates{self.num_updates}_budget{self.budget}_"
            f"loss{self.type}_BO_seed42_gp{self.gp_optim}_{self.initial_lr}_"
            f"scheduler{self.use_scheduler}"
        )
        x_path = os.path.join(results_dir, file_name + ".png")
        z_path = os.path.join(results_dir, file_name)

        # Assign a label
        labels = torch.tensor([label])
        labels = labels.to(self.device)

        # Define the objective function
        f = lambda z: self.evaluate_loss(z, labels, loss_type=self.type, **kwargs)

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
            initial_lr=self.initial_lr,
        )

        # If an initial z is provided, use it instead of a random start
        if initial_z is not None:
            # Ensure correct format
            initial_z = initial_z.cpu().numpy().reshape(1, -1)
            # Evaluate the function at the initial point
            initial_loss = f(initial_z)

            # Override Turbo's initial data
            self.optimizer.X = initial_z  # Set the initial point
            # Set the corresponding loss
            self.optimizer.fX = np.array([initial_loss])

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
        c = torch.nn.functional.one_hot(label_tensor, num_classes=self.num_classes)
        c = c.float().to(self.device)

        with torch.no_grad():
            x_res = self.generator(z_res.float(), c, 1)

        x_res = nn.functional.interpolate(x_res, size=(224, 224), mode="area")
        if x_path and z_path:
            self.save_results(z_res, x_res, z_path, x_path)
        return z_res, x_res, loss_res

    def save_results(self, z, x, z_path, x_path):
        """
        Saves final image and coresponding latent vector.
        Reads all latent vectors created during the run, converts them to
        images and saves.
        """
        # Ensure x is on CPU and properly scaled from [-1, 1] to [0, 1]
        x = x.detach().cpu()
        x = (x.squeeze(0) + 1) / 2.0  # Normalize from [-1, 1] to [0, 1]

        # Convert tensor to PIL image
        transform = transforms.ToPILImage()
        img_pil = transform(x.clamp(0, 1))  # Clamp to [0, 1] range
        # Save the image to specified path
        img_pil.save(x_path)
        print(f"Image saved to {x_path}")

        # Convert z to numpy and save
        if isinstance(z, torch.Tensor):
            z_np = z.detach().cpu().numpy()
        else:
            z_np = np.array(z)
        # Save latent vector for checkpoints
        np.save(z_path, z_np)
        print(f"Latent vector z saved to {z_path}.npy")

        # Convert all latent vectors from the run to images and save
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        # Build the path to the artifacts folder
        results_dir = os.path.join(project_root, "artifacts", "run", z_path)
        # Ensure the directory exists
        os.makedirs(results_dir, exist_ok=True)

        z_dir = os.path.join(project_root, "artifacts", "run")
        os.makedirs(z_dir, exist_ok=True)

        npy_data = {}

        # Assign a label
        label = self.label[0]
        labels = torch.tensor([label])

        for filename in os.listdir(z_dir):
            if filename.endswith(".npy"):
                full_path = os.path.join(z_dir, filename)
                npy_data[filename] = np.load(full_path)

                latent_tensor = torch.from_numpy(npy_data[filename])
                latent_tensor = latent_tensor.to(self.device)
                image = self.generate_image(latent_tensor, labels)
                image = image.detach().cpu()
                # Normalize from [-1, 1] to [0, 1]
                image = (image.squeeze(0) + 1) / 2.0
                transform = transforms.ToPILImage()
                img_pil = transform(image.clamp(0, 1))  # Clamp to [0, 1] range

                # Create output filename
                image_filename = filename.replace(".npy", ".png")
                output_path = os.path.join(results_dir, image_filename)
                print(output_path)
                img_pil.save(output_path)
                print(f"Saved generated image for {filename} to {output_path}")

        # Clearn z_dir for next experiment
        shutil.rmtree(z_dir)
        os.makedirs(z_dir, exist_ok=True)
