python src/attacks/main_reconstructor.py dataset=cifar10 attack=invertgrad model=torchvision +wandb=reconstruction \
    experiment_name=invert_grad_test \
    dataset.save_path=./data \
    attack.unlearned_labels=[0] attack.cfg.recon_iterations=500 \
    model.original_model_ckpt_path=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/neggrad_1s_1e/unlearn/neggrad/resnet18_42_unlearned.pt \
    model.unlearned_model_ckpt_path=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/neggrad_1s_1e/unlearn/neggrad/resnet18_42_original.pt \
    model.model_name=resnet18 \
    wandb.project_name=testrun \