python src/attacks/main_reconstructor.py dataset=cifar10 reconstructor=invertgrad model=torchvision \
    dataset.save_path=./data \
    reconstructor.unlearned_labels=[0] \
    model.original_model_ckpt_path=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/neggrad_1s_1e/unlearn/neggrad/resnet18_42_unlearned.pt \
    model.unlearned_model_ckpt_path=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/neggrad_1s_1e/unlearn/neggrad/resnet18_42_original.pt \
    model.model_name=resnet18 \

    