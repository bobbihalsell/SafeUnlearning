export PYTHONPATH=./src
python src/attacks/GLiR/main.py dataset=cifar10 model=mobilenet_v2 dataset.save_path=./data model.original_model_ckpt_path=artifacts/unlearn/neggrad/mobilenet_v2_42_original.pt model.unlearned_model_ckpt_path=artifacts/unlearn/neggrad/mobilenet_v2_42_unlearned.pt output_dir=artifacts/attack model.num_classes=10 attack=attack

# python src/attacks/gaussian/main.py dataset=cifar10 model=torchhub attack=attack
# python src/attacks/GLiR/main.py dataset=cifar10 model=torchhub attack=attack
python src/attacks/GLiR/main.py dataset=cifar10 dataset.background_ratio=0.1 model=torchhub\
 model.repo_path=chenyaofo/pytorch-cifar-models model.model_name=cifar10_resnet20\
 model.original_model_ckpt_path=artifacts/modelstest/unlearn/neggradplus/cifar10_resnet20_42_original_001.pt\
 model.unlearned_model_ckpt_path=artifacts/modelstest/unlearn/neggradplus/cifar10_resnet20_42_unlearned_001.pt \
 attack=attack