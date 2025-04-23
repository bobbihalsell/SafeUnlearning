import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2
import os
from utils import setup_device


class GLiR:
    def __init__(self, 
                 model_before, 
                 model_after, 
                 method='glir',
                 loss_fn=nn.CrossEntropyLoss(), 
                 num_params=None, 
                 small_var_lim=0.1,
                 device=None):
        """
        Initialize with models before and after unlearning
        
        Args:
            model_before: The model before unlearning
            model_after: The model after unlearning
            loss_fn: Loss function used for gradient computation 
                (default: nn.CrossEntropyLoss())
            num_params: Number of parameters to use (default: None)
            small_var_lim: Exclude parameters with variance below 
                this threshold for stability (default: 0.1)
            device: Device to run computations on (default: None)
        """
        self.model_before = model_before
        self.model_after = model_after
        self.method = method
        self.loss_fn = loss_fn
        if device is None:
            device = setup_device()
        self.device = device
        self.small_var_lim = small_var_lim
        
        # Attributes that will be set in baseline establishment
        self.mean_vector = None
        self.sigma = None
        self.sigma_inv = None
        
        # If num_params is provided, use only that many parameters
        self.num_params = int(num_params)

        num_model_params = sum(p.numel() for p in model_before.parameters())
        if num_params is not None:
            # Sample parameters uniformly across the parameter space
            self.indices = np.linspace(0, num_model_params - 1, 
                                       num_params, dtype=int)
        else:
            self.indices = range(num_model_params)
            
    def compute_gradient(self, model, x, y):
        """
        Compute gradient of loss w.r.t model parameters for a single data point
        """
        model.eval()
        model.zero_grad()
        if (x.dim() == 3 and hasattr(model, 'conv1') 
           or any('conv' in name for name, _ in model.named_modules())):
            # If single image with shape [channels, height, width], 
            # add batch dimension
            x = x.unsqueeze(0)  # Convert to [1, channels, height, width]
        elif x.dim() == 4:
            # Already has batch dimension
            pass
        elif x.dim() == 2:
            # Might be tabular data that needs a batch dimension
            x = x.unsqueeze(0)
        
        y = torch.tensor([y]) if isinstance(y, (int, float)) else y
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        outputs = model(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        
        # Collect gradients
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten().detach().cpu())

        grads = torch.cat(grads)
        return grads[self.indices]
    
    def establish_baseline(self, points):
        """
        Establish baseline distributions using points from a specific set
        """        
        print(f"Establishing baseline using {len(points)} points with method '{self.method}'")
        # Standard approach for non-separate methods
        features = []
        for (x, y) in tqdm(points):
            feature_vector = self.compute_feature_vector(x, y)
            features.append(feature_vector)
        features = torch.stack(features)

        # Compute the mean and variance of the features
        feature_means = features.mean(dim=0, keepdim=True)
        feature_vars_norm = features - feature_means
        feature_vars = feature_vars_norm.var(axis=0, keepdim=False)
        print(f"Total {self.method} feature dimensions:", feature_vars.numel())

        # Identify and remove small vars for numerical stability
        small_var = feature_vars < self.small_var_lim
        print("Small variance elements:", torch.sum(small_var).item())
        feature_means = feature_means[:, ~small_var]
        feature_vars_norm = feature_vars_norm[:, ~small_var]
        
        # Calculate covariance matrix
        self.sigma = (feature_vars_norm.t() @ feature_vars_norm
                      ) / len(feature_vars_norm)
        self.mean_vector = feature_means
        self.valid_indices = ~small_var
        
        print("Inverting Sigma matrix...")
        try:
            # Add regularization for numerical stability
            sigma_reg = self.sigma + torch.eye(self.sigma.shape[0], 
                                               device=self.sigma.device) * 1e-4
            L = torch.linalg.cholesky(sigma_reg)
            self.sigma_inv = torch.cholesky_inverse(L)
            print("Cholesky decomposition successful")
        except Exception as e:
            print(f"Cholesky failed: {str(e)}")
            print("Using standard inverse with regularization...")
            # Add more regularization for standard inverse
            sigma_reg = self.sigma + torch.eye(self.sigma.shape[0], 
                                               device=self.sigma.device) * 1e-3
            self.sigma_inv = torch.inverse(sigma_reg)
            print("Standard inverse successful")

    def compute_feature_vector(self, x, y):
        grad_before = self.compute_gradient(self.model_before, x, y)
        grad_after = self.compute_gradient(self.model_after, x, y)
        grad_diff = grad_before - grad_after
        grad_before_magnitude = torch.abs(grad_before)
        grad_diff_norm = torch.norm(grad_diff)
        grad_before_norm = torch.norm(grad_before_magnitude)
        epsilon = 1e-10
        if self.method == 'glir':
            return grad_diff
        
        elif self.method == 'ratio':
            # Return only the ratio features (compact representation)
            forget_score = grad_diff_norm / (grad_before_norm + epsilon)
            test_score = grad_before_norm / (grad_diff_norm + epsilon)
            return torch.tensor([forget_score, test_score])
            
    def compute_test_statistic(self, x, y):
        """
        Compute test statistic for a data point
        """
        feature_vector = self.compute_feature_vector(x, y)
        
        # Handle dimension mismatch between feature vector and valid_indices
        if hasattr(self, 'valid_indices'):
            feature_vector = feature_vector[self.valid_indices]
        
        # Center the feature vector
        if hasattr(self, 'mean_vector'):
            centered_vector = feature_vector - self.mean_vector.squeeze(0)
        # Compute the Mahalanobis distance
        if hasattr(self, 'sigma_inv'):
            mahalanobis_distance = torch.sum(
                centered_vector * (self.sigma_inv @ centered_vector)
            )
        # Compute the Likelihood Ratio Test Statistic
        lrt_statistic = 2 * mahalanobis_distance
    
        return lrt_statistic

    def compute_p_value(self, lrt_statistic):
        """
        Compute the p-value using the chi-squared distribution
        """
        lrt_value = lrt_statistic.item()
        self.df = self.sigma_inv.shape[0]
        # Compute the p-value using the chi-squared distribution CDF
        p_value = 1 - chi2.cdf(lrt_value, self.df)
    
        return p_value

    def classify_point(self, x, y, threshold, teststatistic=False):
        """
        Classify a point as in forget set or not using statistical deviation
        """
        if not hasattr(self, 'mean_vector'):
            raise ValueError("Must establish baseline before classification")

        lrt_statistic = self.compute_test_statistic(x, y)
        p_val = self.compute_p_value(lrt_statistic)
        is_forgotten = p_val < threshold
        if teststatistic:
            return is_forgotten, p_val, lrt_statistic
        else:
            return is_forgotten, p_val

    def classify_set(self, points, threshold, teststatistic=False):
        """
        Classify a set of points as forgotten or retained
        
        Args:
            points: List of (x, y) tuples to classify
            threshold: Optional custom threshold
            
        Returns:
            forget_set, retain_set with points and their p_vals values
        """
        classes = []
        p_vals = []
        if teststatistic:
            test_statistics = []
        
        for point in tqdm(points):
            x, y = point
            results = self.classify_point(x, y, threshold, teststatistic)
            is_forgotten = results[0]
            classes.append(1 if is_forgotten else 0)  
            p_val = results[1]
            p_vals.append(p_val)
            if teststatistic:
                test_statistic = results[2]
                test_statistics.append(test_statistic)
        if teststatistic:
            return classes, p_vals, test_statistics
        return classes, p_vals

    def plot_roc_curve(self, classifications, labels, savepath=None):
        """
        Plots the ROC curve by varying the classification threshold.
        """
        # Calculate the FPR and TPR for various thresholds
        fpr, tpr, thresholds = roc_curve(labels, classifications)

        # Calculate the Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Save the plot if a save path is provided
        if savepath is not None:
            try:
                os.makedirs(savepath, exist_ok=True)
                plot_path = os.path.join(savepath, "roc_curve.png")
                plt.savefig(plot_path, format='png', bbox_inches='tight')
                print(f"ROC curve saved at: {plot_path}")
            except Exception as e:
                print(f"Failed to save plot: {e}")

        plt.show(block=False)
        plt.close()

    def plot_glir_distribution(self, 
                               p_values=None, 
                               test_statistics=None, 
                               labels=None, 
                               savepath=None, 
                               alpha=0.05, 
                               bins=50, 
                               ):
        """
        Plot the GLiR test statistics distribution with different colors based 
        on labels:
        - Distribution curve: Baseline chi-squared distribution
        - Red dots/histogram: Forget points (label=1)
        - Blue dots/histogram: Test points (label=0)
        - Optional: Threshold line for significance at alpha level
        
        Args:
            p_values: List/array of p-values
            test_statistics: List/array of LRT statistics (before p-value 
            conversion)
            labels: Binary labels indicating forget (1) or test (0) points
            alpha: Significance level for threshold
            bins: Number of bins for histogram
            df: Degrees of freedom for chi-squared distribution
        """
        # Handle the case where only p-values are provided
        if p_values is not None:
            # Convert to numpy array if it's a list
            if isinstance(p_values, list):
                p_values = np.array(p_values, dtype=float)
            elif isinstance(p_values, torch.Tensor):
                p_values = p_values.cpu().numpy()
            if np.isscalar(p_values):
                p_values = np.array([p_values])
                
            # Calculate test statistics from p-values
            if test_statistics is None:
                test_statistics = chi2.ppf(1 - p_values, self.df)
        
        # Handle the case where test statistics are provided
        elif test_statistics is not None:
            if isinstance(test_statistics, torch.Tensor):
                test_statistics = test_statistics.cpu().numpy()
            if np.isscalar(test_statistics):
                test_statistics = np.array([test_statistics])
            
            # Calculate p-values if not provided
            if p_values is None:
                p_values = 1 - chi2.cdf(test_statistics, self.df)
        
        else:
            raise ValueError(
                "Either p_values or test_statistics must be provided."
                )
        
        # Convert labels to numpy array if provided
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            if np.isscalar(labels):
                labels = np.array([labels])
            
            # Ensure labels match the length of p_values/test_statistics
            if len(labels) != len(p_values):
                raise ValueError(
                    "Number of labels must match number of "
                    "p_values/test_statistics")
                
            # Initialize empty lists for forget and test points
            forget_stats = []
            test_stats = []
            forget_pvals = []
            test_pvals = []

            # Loop through labels and separate points into respective lists
            for i, val in enumerate(labels):
                if val == 1:  # Forget point
                    forget_stats.append(test_statistics[i])
                    forget_pvals.append(p_values[i])
                else:  # Test point (val == 0)
                    test_stats.append(test_statistics[i])
                    test_pvals.append(p_values[i])

            # Convert to numpy arrays for further processing
            forget_stats = np.array(forget_stats)
            test_stats = np.array(test_stats)
            forget_pvals = np.array(forget_pvals)
            test_pvals = np.array(test_pvals)
        else:
            # If no labels provided, treat all points the same (green)
            forget_stats = np.array([])
            test_stats = np.array([])
            forget_pvals = np.array([])
            test_pvals = np.array([])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Chi-squared distribution with test statistics
        x = np.linspace(0, max(20, np.max(test_statistics) * 1.2), 1000)
        chi2_pdf = chi2.pdf(x, self.df)
        ax1.plot(x, chi2_pdf, 'k-', lw=2, label=f'χ²({self.df}) Distribution')
        
        # Threshold alpha
        threshold = chi2.ppf(1 - alpha, self.df)
        ax1.axvline(x=threshold, color='g', linestyle='--', 
                    label=f'alpha={alpha} threshold')
        
        # Plot points with different colors based on labels
        if labels is not None:
            # Forget points (red)
            if len(forget_stats) > 0:
                ax1.scatter(forget_stats, chi2.pdf(forget_stats, self.df), 
                            color='r', alpha=0.7, 
                            label='Forget Points (label=1)')
                ax1.plot(forget_stats, np.zeros_like(forget_stats), '|', 
                         alpha=0.4, color='r', ms=20)
            
            # Test points (blue)
            if len(test_stats) > 0:
                ax1.scatter(test_stats, chi2.pdf(test_stats, self.df), 
                            color='b', alpha=0.7, 
                            label='Test Points (label=0)')
                ax1.plot(test_stats, np.zeros_like(test_stats), '|', 
                         alpha=0.4, color='b', ms=20)
        else:
            # No labels, plot all points in green
            ax1.scatter(test_statistics, chi2.pdf(test_statistics, self.df), 
                        color='g', alpha=0.7, 
                        label='Test Statistics')
            ax1.plot(test_statistics, np.zeros_like(test_statistics), '|', 
                     color='g', ms=20)
        
        ax1.set_xlabel('Test Statistic Values (χ²)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('GLiR Test Statistics Distribution')
        ax1.legend()
        
        # Plot 2: p-value distribution (histogram)
        if labels is not None:
            # Two histograms with different colors
            if len(forget_pvals) > 0:
                ax2.hist(forget_pvals, bins=bins, alpha=0.4, color='r', 
                         density=True, label='Forget Points (label=1)')
            
            if len(test_pvals) > 0:
                ax2.hist(test_pvals, bins=bins, alpha=0.4, color='b', 
                         density=True, label='Test Points (label=0)')
        else:
            # No labels, plot all p-values in green
            ax2.hist(p_values, bins=bins, alpha=0.6, color='g', 
                     density=True, label='All Points')
        
        # Null hypothesis line
        ax2.axhline(y=1.0, color='k', linestyle='-', 
                    label='Uniform Distribution')
        # Threshold line
        ax2.axvline(x=alpha, color='g', linestyle='--', 
                    label=f'alpha={alpha} threshold')
        
        ax2.set_xlabel('p-values')
        ax2.set_ylabel('Density')
        ax2.set_title('p-value Distribution')
        ax2.legend()
        
        plt.tight_layout()
        if savepath is not None:
            try:
                os.makedirs(savepath, exist_ok=True)
                plot_path = os.path.join(savepath, "glir_distribution.png")
                plt.savefig(plot_path, format='png', bbox_inches='tight')
                print(f"GLIR Distribution saved at: {plot_path}")
            except Exception as e:
                print(f"Failed to save plot: {e}")

        plt.show(block=False)
        plt.close()
        return fig