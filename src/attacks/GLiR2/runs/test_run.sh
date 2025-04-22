export PYTHONPATH=./src
python src/attacks/GLiR/main.py dataset=cifar10 model=mobilenet_v2 dataset.save_path=./data model.original_model_ckpt_path=artifacts/unlearn/neggrad/mobilenet_v2_42_original.pt model.unlearned_model_ckpt_path=artifacts/unlearn/neggrad/mobilenet_v2_42_unlearned.pt output_dir=artifacts/attack model.num_classes=10 attack=attack

# python src/attacks/gaussian/main.py dataset=cifar10 model=torchhub attack=attack
# python src/attacks/GLiR/main.py dataset=cifar10 model=torchhub attack=attack