import unittest
import os
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pytest

def plot_glir_distribution(p_values=None, 
                            test_statistics=None, 
                            labels=None, 
                            savepath=None, 
                            alpha=0.05, 
                            bins=50, 
                            df=1):
    """
    Plot the GLiR test statistics distribution with different colors based on labels:
    - Distribution curve: Baseline chi-squared distribution
    - Red dots/histogram: Forget points (label=1)
    - Blue dots/histogram: Test points (label=0)
    - Optional: Threshold line for significance at alpha level
    
    Args:
        p_values: List/array of p-values
        test_statistics: List/array of LRT statistics (before p-value conversion)
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
            test_statistics = chi2.ppf(1 - p_values, df)
    
    # Handle the case where test statistics are provided
    elif test_statistics is not None:
        if isinstance(test_statistics, torch.Tensor):
            test_statistics = test_statistics.cpu().numpy()
        if np.isscalar(test_statistics):
            test_statistics = np.array([test_statistics])
        
        # Calculate p-values if not provided
        if p_values is None:
            p_values = 1 - chi2.cdf(test_statistics, df)
    
    else:
        raise ValueError("Either p_values or test_statistics must be provided.")
    
    # Convert labels to numpy array if provided
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if np.isscalar(labels):
            labels = np.array([labels])
        
        # Ensure labels match the length of p_values/test_statistics
        if len(labels) != len(p_values):
            raise ValueError("Number of labels must match number of p_values/test_statistics")
            
        # Create masks for forget and test points
        forget_indices = [i for i, val in enumerate(labels) if val == 1]
        test_indices = [i for i, val in enumerate(labels) if val == 0]
        
        # Split data based on labels
        forget_stats = test_statistics[forget_indices]
        test_stats = test_statistics[test_indices]
        forget_pvals = p_values[forget_indices]
        test_pvals = p_values[test_indices]
    else:
        # If no labels provided, treat all points the same (green)
        forget_stats = np.array([])
        test_stats = np.array([])
        forget_pvals = np.array([])
        test_pvals = np.array([])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Chi-squared distribution with test statistics
    x = np.linspace(0, max(20, np.max(test_statistics) * 1.2), 1000)
    chi2_pdf = chi2.pdf(x, df)
    ax1.plot(x, chi2_pdf, 'k-', lw=2, label=f'χ²({df}) Distribution')
    
    # Threshold alpha
    threshold = chi2.ppf(1 - alpha, df)
    ax1.axvline(x=threshold, color='g', linestyle='--', 
                label=f'alpha={alpha} threshold')
    
    # Plot points with different colors based on labels
    if labels is not None:
        # Forget points (red)
        if len(forget_stats) > 0:
            ax1.scatter(forget_stats, chi2.pdf(forget_stats, df), 
                      color='r', alpha=0.7, label='Forget Points (label=1)')
            ax1.plot(forget_stats, np.zeros_like(forget_stats), '|', alpha=0.4, color='r', ms=20)
        
        # Test points (blue)
        if len(test_stats) > 0:
            ax1.scatter(test_stats, chi2.pdf(test_stats, df), 
                      color='b', alpha=0.7, label='Test Points (label=0)')
            ax1.plot(test_stats, np.zeros_like(test_stats), '|', alpha=0.4, color='b', ms=20)
    else:
        # No labels, plot all points in green
        ax1.scatter(test_statistics, chi2.pdf(test_statistics, df), 
                  color='g', alpha=0.7, label='Test Statistics')
        ax1.plot(test_statistics, np.zeros_like(test_statistics), '|', color='g', ms=20)
    
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
        ax2.hist(p_values, bins=bins, alpha=0.6, color='g', density=True, label='All Points')
    
    # Null hypothesis line
    ax2.axhline(y=1.0, color='k', linestyle='-', label='Uniform Distribution')
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

def make_random_pvals_and_test_statistics(n, df=1, astype='numpy'):
    """Generate random p-values and corresponding test statistics."""
    # Create some test statistics from chi-squared distribution
    test_stats = np.random.chisquare(df, n)
    # Convert to p-values
    pvals = 1 - chi2.cdf(test_stats, df)
    
    # Convert to requested type
    if astype == 'list':
        return pvals.tolist(), test_stats.tolist()
    elif astype == 'tensor':
        import torch
        return torch.tensor(pvals), torch.tensor(test_stats)
    return pvals, test_stats

def make_labels(pvals, threshold=0.05, astype='numpy'):
    """Create binary labels based on p-values (simulate forget vs test)."""
    # Create labels - values below threshold are "forget" (1), others are "test" (0)
    labels = np.zeros(len(pvals))
    labels[np.array(pvals) < threshold] = 1  # Some will be "forget" points
    
    # Make sure we have at least one of each class for testing
    if sum(labels) == 0:
        labels[0] = 1  # Ensure at least one forget point
    if sum(labels) == len(labels):
        labels[-1] = 0  # Ensure at least one test point
    
    # Convert to requested type
    if astype == 'list':
        return labels.tolist()
    elif astype == 'tensor':
        import torch
        return torch.tensor(labels)
    return labels

class TestGLiRDistribution(unittest.TestCase):
    """Test suite for the GLiR distribution plotting function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for saving plots
        self.test_dir = "test_plots"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Suppress plt.show() to avoid displaying plots during tests
        self.mock_show = patch('matplotlib.pyplot.show').start()
        self.mock_close = patch('matplotlib.pyplot.close').start()
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test directory and files
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
        # Stop all patches
        patch.stopall()
        
    def test_plot_glir_with_pvalues_only(self):
        """Test plotting using p-values only (numpy arrays)."""
        # Generate test data
        pvals, _ = make_random_pvals_and_test_statistics(100, df=1)
        labels = make_labels(pvals)
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=pvals, labels=labels)
            self.assertIsNotNone(fig)
            self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error: {e}")
            
    def test_plot_glir_with_test_statistics_only(self):
        """Test plotting using test statistics only (numpy arrays)."""
        # Generate test data
        _, test_stats = make_random_pvals_and_test_statistics(100, df=1)
        pvals = 1 - chi2.cdf(test_stats, df=1)
        labels = make_labels(pvals)
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(test_statistics=test_stats, labels=labels)
            self.assertIsNotNone(fig)
            self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error: {e}")
            
    def test_plot_glir_with_both_pvalues_and_test_statistics(self):
        """Test plotting using both p-values and test statistics."""
        # Generate test data
        pvals, test_stats = make_random_pvals_and_test_statistics(100, df=1)
        labels = make_labels(pvals)
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=pvals, test_statistics=test_stats, labels=labels)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error: {e}")
            
    def test_plot_glir_without_labels(self):
        """Test plotting without providing labels."""
        # Generate test data
        pvals, _ = make_random_pvals_and_test_statistics(100, df=1)
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=pvals)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error: {e}")
            
    def test_plot_glir_with_lists(self):
        """Test plotting using list inputs."""
        # Generate test data
        pvals, test_stats = make_random_pvals_and_test_statistics(100, df=1, astype='list')        
        labels = make_labels(pvals, astype='list')
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=pvals, test_statistics=None, labels=labels)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error with list input: {e}")
            
    def test_plot_glir_with_tensors(self):
        """Test plotting using tensor inputs."""
        # Generate test data
        pvals, test_stats = make_random_pvals_and_test_statistics(100, df=1, astype='tensor')        
        labels = make_labels(pvals, astype='tensor')
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=pvals, test_statistics=None, labels=labels)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error with tensor input: {e}")
            
    def test_plot_glir_with_savepath(self):
        """Test that the plot is saved to the specified path."""
        # Generate test data
        pvals, _ = make_random_pvals_and_test_statistics(100, df=1)
        labels = make_labels(pvals)
        
        # Use the test_dir as savepath
        fig = plot_glir_distribution(p_values=pvals, labels=labels, savepath=self.test_dir)
        
        # Check if the plot was saved
        expected_path = os.path.join(self.test_dir, "glir_distribution.png")
        self.assertTrue(os.path.exists(expected_path), f"Plot was not saved at {expected_path}")
        
    def test_plot_glir_with_scalar_inputs(self):
        """Test plotting with scalar inputs."""
        # Generate a single p-value and label
        p_value = 0.03  # A single p-value
        label = 1       # A single label
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=p_value, labels=label)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error with scalar input: {e}")
            
    def test_plot_glir_with_different_df(self):
        """Test plotting with different degrees of freedom."""
        # Generate test data
        pvals, _ = make_random_pvals_and_test_statistics(100, df=3)  # Use df=3
        labels = make_labels(pvals)
        
        # Should not raise errors
        try:
            fig = plot_glir_distribution(p_values=pvals, labels=labels, df=3)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution raised an unexpected error with df=3: {e}")
            
    def test_plot_glir_with_empty_inputs(self):
        """Test how the function handles empty arrays."""
        # Create empty arrays
        pvals = np.array([])
        labels = np.array([])
        
        # Should raise a ValueError because we need at least one data point
        with self.assertRaises(Exception):
            plot_glir_distribution(p_values=pvals, labels=labels)
            
    def test_plot_glir_with_mismatched_labels(self):
        """Test error handling when labels don't match p-values."""
        # Generate test data with mismatched lengths
        pvals, _ = make_random_pvals_and_test_statistics(100, df=1)
        labels = make_labels(pvals[:50])  # Only half the number of labels
        
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            plot_glir_distribution(p_values=pvals, labels=labels)
            
    @patch('matplotlib.pyplot.savefig')
    def test_plot_glir_with_savefig_error(self, mock_savefig):
        """Test error handling when saving the plot fails."""
        # Make savefig raise an exception
        mock_savefig.side_effect = Exception("Mock save error")
        
        # Generate test data
        pvals, _ = make_random_pvals_and_test_statistics(100, df=1)
        labels = make_labels(pvals)
        
        # Function should handle the save error gracefully
        try:
            fig = plot_glir_distribution(p_values=pvals, labels=labels, savepath=self.test_dir)
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"plot_glir_distribution did not handle savefig error gracefully: {e}")
            
    def test_plot_glir_without_pvalues_or_test_statistics(self):
        """Test error handling when neither p-values nor test statistics are provided."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            plot_glir_distribution()

# Main execution
if __name__ == '__main__':
    unittest.main()