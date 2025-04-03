import numpy as np
import random
import torch

# save image in png or pdf
# multiple or single
# with annotations/no annotations
# size options etc


# save image to tensor

# load tensor image

# unsuported image format

def setup_device():
    """ Setup a torch device.

    Returns:
        str
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior
