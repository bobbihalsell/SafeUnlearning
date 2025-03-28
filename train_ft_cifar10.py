# Test to check saved model artifacts
import torch

original = torch.load('artifacts/unlearn/neggrad/mobilenet_v2_42_original.pt', weights_only=False, map_location='cpu')
unlearned = torch.load('artifacts/unlearn/neggrad/mobilenet_v2_42_unlearned.pt', weights_only=False, map_location='cpu')


print(unlearned["train_losses"])
print('==' * 20)
