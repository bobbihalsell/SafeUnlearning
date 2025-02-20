import torch
from src.unlearning.utils import _setup_device


class BaseUnlearner:
    def __init__(self):
        self.device = _setup_device()

    def get_model_accuracy(self, model, dataloader):
        """ Get model accuracy over a torch Dataloader.

        Args:
            model (torch.nn.Module): A Pytorch model.
            dataloader (torch.utils.data.Dataloader): A Dataloader for the dataset you want to compute accuracy on.

        Returns:
            accuracy (float): The model accuracy over the provided dataset.
        """
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predictions = outputs.max(1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        accuracy = 100 * correct / total if total > 0 else 0.0

        return accuracy