import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import snapshot_download
import legacy  # comes from stylegan2-ada-pytorch
import dnnlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from recover_gradients.recover_gradients import calculate_covar, calculate_svd, calculate_svd_difference

# Download pretrained StyleGAN2-ADA weights and config
model_path = snapshot_download("justinpinkney/stylegan2-ada")
network_pkl = f"{model_path}/ffhq.pkl"  # adjust if you download a different pkl

# Load both G and D
with dnnlib.util.open_url(network_pkl) as f:
    data = legacy.load_network_pkl(f)
    G = data['G_ema'].to('cuda')   # or data['G'] for the non-EMA generator
    D = data['D'].to('cuda')

class StyleGanAttack:
    def __init__(G, D, latent_dim, original_model, unlearned_model, device):
        self.G = G.eval()
        self.D = D.eval()
        self.original_model = original_model
        self.unlearned_model = unlearned_model
        self.latent_dim = latent_dim

    def initialise_z(self, n):
        """
        Sample `n` latent vectors from a standard Gaussian distribution.

        Args:
            n (int): Number of latent vectors to sample.
            latent_dim (int): Dimensionality of the latent space (default 512).

        Returns:
            torch.Tensor: Latent vectors of shape (n, latent_dim).
        """
        return torch.randn(n, self.latent_dim).to(self.device)


    def w_from_z(self, z):
        """
        Map latent vector z to intermediate latent w via the mapping network.

        Args:
            z (torch.Tensor): Input latent vector(s), shape (N, 512).

        Returns:
            torch.Tensor: Corresponding W vectors, shape (N, 18, 512) for StyleGAN2.
        """
        w = G.mapping(self.z, None)  # `None` is for label input
        return w


    def generate_from_z(self, z, noise=None):
        """
        Generate images directly from z, passing through mapping and synthesis.

        Args:
            z (torch.Tensor): Latent vector(s) from Z space.
            noise (optional): Custom noise input (list of tensors or None).

        Returns:
            torch.Tensor: Generated image tensor(s).
        """
        w = w_from_z(z)
        return generate_from_w(w, noise=noise)


    def generate_from_w(w, noise=None):
        """
        Generate images from W-space latent vectors.

        Args:
            w (torch.Tensor): W latent tensor of shape (N, 18, 512) or (N, 1, 512).
            noise (optional): Custom noise injection (list of tensors, or None).

        Returns:
            torch.Tensor: Generated images, shape (N, 3, H, W).
        """
        return self.G.synthesis(w, noise_mode='const', noise=noise)


    def discriminator_score(self, img, label=None):
        """
        Get the raw discriminator prediction for a single image (or batch).

        Args:
            img (torch.Tensor): Image tensor of shape (1, 3, H, W), normalized [-1, 1].
            label (torch.Tensor or None): Optional label tensor for conditional D.

        Returns:
            torch.Tensor: Raw discriminator output.
        """
        img = img.to(self.device)
        if label is not None:
            label = label.to(self.device)
            return self.D(img, label)
        else:
            return self.D(img)


    def diversity_loss(ws, metric='l2'):
        """
        Compute diversity loss from a batch of W-space latent vectors.

        Args:
            ws (torch.Tensor): Latent vectors in W space, shape (N, ..., D)
            metric (str): Distance metric, either 'l2' or 'cos'

        Returns:
            torch.Tensor: Scalar diversity loss (lower = less diverse)
        """
        N = ws.shape[0]
        ws_flat = ws.view(N, -1)  # flatten (for W+ or multi-layer W)
        loss = 0.0
        count = 0

        for i in range(N):
            for j in range(i + 1, N):
                if metric == 'l2':
                    dist = (ws_flat[i] - ws_flat[j]).pow(2).sum()
                elif metric == 'cos':
                    dist = 1 - F.cosine_similarity(ws_flat[i].unsqueeze(0), ws_flat[j].unsqueeze(0))
                    dist = dist.mean()
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                loss += dist
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=ws.device)


# transform = transforms.Compose([
#     transforms.ToTensor(),  # or Resize(), Normalize(), etc.
# ])

    def gradient_loss(self, imgs, true_svd, transform, k=None):
        img_loader = DataLoader(imgs, batch_size=len(dataset), shuffle=False)
        original_covar = calculate_covar(self.original_model, img_loader, self.device)
        unlearned_covar = calculate_covar(self.unlearned_model, img_loader, self.device)
        svd_difference = calculate_svd_difference(original_svd, unlearned_svd, self.device, k)
        # TODO
        return 

    def attack_loss(self, true_svd, ws, imgs, disc_w, div_w, grad_w, metric, transform, k=None):
        disc_L = 0
        for img in imgs
            disc_L += 1 - self.discriminator_score(img)
        div_L = self.diversity_loss(ws, metric)
        grad_L = self.gradient_loss(imgs, true_svd, transform, k)
        return disc_w*disc_L + div_w*div_L + grad_w*grad_L



