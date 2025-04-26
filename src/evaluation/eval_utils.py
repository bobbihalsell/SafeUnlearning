import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, Any
import os


def accuracy(model, dataloaders, device, model_name='', verbose=False):
    """Evaluate model accuracy on different data splits."""
    model.eval()
    model_results = {}
    
    for split_name, dataloader in dataloaders.items():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        model_results[split_name] = accuracy
        if verbose:
            print(f"{model_name} model accuracies:", end=" ")
            print(", ".join([f"{key}: {value:.2f}%" for key, value in model_results.items()]))  
    return model_results


def normed_distance(model_a, model_b):
    """Compute normalized distance between two models."""
    distance = 0
    normalization = 0
    
    for (p1, p2) in zip(model_a.parameters(), model_b.parameters()):
        current_dist = (p1.data - p2.data).pow(2).sum().item()
        current_norm = p1.data.pow(2).sum().item()
        distance += current_dist
        normalization += current_norm
    return 1.0 * np.sqrt(distance / normalization)


def l2_distance(model_a, model_b):
    """Calculate the L2 distance (Euclidean distance) between the weights of two models."""
    distance = 0.0
    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        if p1.data.nelement() == p2.data.nelement():
            diff = p1.data - p2.data
            distance += torch.norm(diff, p=2) ** 2
        else:
            raise ValueError("Models have different architectures or parameter configurations.")
    return torch.sqrt(distance).item()


def model_l2_norm(model, device):
    """Compute the L2 norm of the model's parameters."""
    distance = torch.zeros(1, device=device)
    for param in model.parameters():
        distance += torch.norm(param, p=2) ** 2
    return torch.sqrt(distance).item()


def normed_l2_distance(model_a, model_b, device):
    """Compute the L2 norm of the difference between two normalized models' parameters."""
    norm_a = model_l2_norm(model_a, device)
    norm_b = model_l2_norm(model_b, device)
    
    distance = torch.zeros(1, device=device)
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        normalized_param_a = param_a / norm_a
        normalized_param_b = param_b / norm_b
        distance += torch.norm(normalized_param_a - normalized_param_b, p=2) ** 2
    return torch.sqrt(distance).item()
    

def kld(model_a, model_b, dataloaders, device):
    """
    Compute KL divergence between two model outputs.
    KL(model_a || model_b)
    """
    results = {}
    
    for split_name, dataloader in dataloaders.items():
        total_kl = 0.0
        sample_count = 0
        
        model_a.eval()
        model_b.eval()
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                logits_a = model_a(inputs)
                logits_b = model_b(inputs)
                
                # Apply softmax to get probabilities
                probs_a = F.softmax(logits_a, dim=1)
                probs_b = F.softmax(logits_b, dim=1)
                
                # Compute KL divergence: KL(p||q)
                kl_div = F.kl_div(
                    probs_b.log(),  # q (target distribution) in log space
                    probs_a,         # p (source distribution)
                    reduction='batchmean'
                )
                
                total_kl += kl_div.item() * inputs.size(0)
                sample_count += inputs.size(0)
        
        avg_kl = total_kl / sample_count if sample_count > 0 else 0.0
        results[split_name] = avg_kl
    return results


def plot_model_accuracies(results, output_dir="plots"):
    """
    Plot accuracies for all models across different datasets.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if "model_accuracy" not in results:
        print("No accuracy results found")
        return
    
    model_accuracy = results["model_accuracy"]
    model_names = list(model_accuracy.keys())
    datasets = []
    
    # Get dataset names
    for model_data in model_accuracy.values():
        datasets = list(model_data.keys())
        break
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 6), squeeze=False)
    
    # Color map for models 
    colors = plt.cm.Set1(range(len(model_names))) 
    color_map = {model: colors[i] for i, model in enumerate(model_names)}
    
    # Plot data for each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[0, i]
        # x = np.arange(len(model_names))
        width = 0.7
        
        accuracies = []
        valid_models = []
        valid_colors = []
        
        for j, model in enumerate(model_names):
            if dataset in model_accuracy[model]:
                valid_models.append(model)
                accuracies.append(model_accuracy[model][dataset])
                valid_colors.append(color_map[model])
        bars = ax.bar(np.arange(len(valid_models)), accuracies, width, label=valid_models, color=valid_colors)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
        
        ax.set_title(f'Accuracy on {dataset} dataset')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 105)  
        ax.set_xticks(np.arange(len(valid_models)))
        ax.set_xticklabels(valid_models, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[model]) for model in model_names]
    fig.legend(handles, model_names, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(model_names), frameon=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle('Model Accuracy Comparison Across Datasets', fontsize=16, y=0.98)
    
    # Save figure
    plot_path = os.path.join(output_dir, "model_accuracies.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to {plot_path}")
    plt.close()


def plot_weight_distance_metrics(results, output_dir = "plots"):
    """
    Plot weight-based distance metrics between original and unlearned models side by side.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if "unl_distances" not in results:
        print("No distance results found")
        return
    
    unl_distances = results["unl_distances"]
    model_names = list(unl_distances.keys())
    weight_metrics = ["l2_distance", "normalized_distance", "normalized_l2_distance"]
    
    # Create figure with subplots side by side
    fig, axes = plt.subplots(1, len(weight_metrics), figsize=(5 * len(weight_metrics), 6), squeeze=False)
    
    # Color map for models
    colors = plt.cm.Set1(range(len(model_names))) 
    color_map = {model: colors[i] for i, model in enumerate(model_names)}
    
    for i, metric in enumerate(weight_metrics):
        ax = axes[0, i]
        
        # Extract values for this metric
        values = []
        valid_models = []
        valid_colors = []
        
        for model in model_names:
            if metric in unl_distances[model]:
                valid_models.append(model)
                values.append(unl_distances[model][metric])
                valid_colors.append(color_map[model])
        
        bars = ax.bar(np.arange(len(valid_models)), values, 0.7, color=valid_colors)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
        
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f'{metric_title}')
        ax.set_ylabel(metric_title)
        ax.set_xticks(np.arange(len(valid_models)))
        ax.set_xticklabels(valid_models, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[model]) for model in model_names]
    fig.legend(handles, model_names, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(model_names), frameon=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle('Weight-Based Distance Metrics Between Models', fontsize=16, y=0.98)
    
    # Save figure
    plot_path = os.path.join(output_dir, "weight_distance_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved weight distance metrics plot to {plot_path}")
    plt.close()


def plot_kl_divergence(results, output_dir="plots"):
    """
    Plot KL divergence between original and unlearned models across different datasets.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if "unl_distances" not in results:
        print("No distance results found")
        return
    
    unl_distances = results["unl_distances"]
    model_names = list(unl_distances.keys())
    
    first_model = list(unl_distances.keys())[0]
    datasets = list(unl_distances[first_model]["kl_divergence"].keys())
    
    # Create figure with subplots - one per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 6), squeeze=False)
    
    # Color map for models
    colors = plt.cm.Set1(range(len(model_names))) 
    color_map = {model: colors[i] for i, model in enumerate(model_names)}
    
    # Plot KL divergence for each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[0, i]
        
        # Extract KL values for this dataset
        values = []
        valid_models = []
        valid_colors = []
        
        for model in model_names:
            if "kl_divergence" in unl_distances[model] and dataset in unl_distances[model]["kl_divergence"]:
                valid_models.append(model)
                values.append(unl_distances[model]["kl_divergence"][dataset])
                valid_colors.append(color_map[model])
        
        # Plot bars
        bars = ax.bar(np.arange(len(valid_models)), values, 0.7, color=valid_colors)
        
        # Add values above bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
        
        # Customize plot
        ax.set_title(f'KL Divergence on {dataset} dataset')
        ax.set_ylabel('KL Divergence')
        ax.set_xticks(np.arange(len(valid_models)))
        ax.set_xticklabels(valid_models, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a common legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[model]) for model in model_names]
    fig.legend(handles, model_names, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(model_names), frameon=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.suptitle('KL Divergence Between Models Across Datasets', fontsize=16, y=0.98)
    
    # Save figure
    plot_path = os.path.join(output_dir, "kl_divergence.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved KL divergence plot to {plot_path}")
    plt.close()


def visualize_evaluation_results(results, output_dir="plots"):
    """
    Generate all visualization plots for the evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the plots
    """
    plot_model_accuracies(results, output_dir)
    plot_weight_distance_metrics(results, output_dir)
    plot_kl_divergence(results, output_dir)
    
    print(f"All plots saved to {output_dir}")

