python src/attacks/GLiR/main.py dataset=cifar10 model=torchhub attack=glir \
    experiment_name=glir_test \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    model.original_model_ckpt_path=artifacts/models/cifar10_resnet20_42_original_001.pt \
    model.unlearned_model_ckpt_path=artifacts/models/cifar10_resnet20_42_unlearned_001.pt \
    +combined_roc_dir=artifacts/attacks/ +plot_combined_roc=true