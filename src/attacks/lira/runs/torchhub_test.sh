python src/attacks/lira/main.py dataset=cifar10 model=resnet18 unlearner=neggrad attack=lira

python src/attacks/lira/main.py dataset=cifar10 model=torchhub\
 model.repo_path=chenyaofo/pytorch-cifar-models model.model_name=cifar10_resnet20\
 attack=lira trainer=trainer unlearner=neggradplus