import torch


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
