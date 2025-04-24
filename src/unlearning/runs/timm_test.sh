python src/unlearning/main.py dataset=cifar10 model=timm unlearner=neggrad \
    experiment_name=neggrad_test \
    model.model_name=resnet18 \
    model.original_model_ckpt_path=./artifacts/models/cifar10/resnet18_42_original.pt \
    dataset.save_path=./datatest