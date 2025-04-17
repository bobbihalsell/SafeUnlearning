python src/attacks/main_reconstructor.py data=cifar10 reconstructor=invertgrad \
    unlearned_weights=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/resnet18_2s_1e/unlearn/neggrad/resnet18_42_unlearned.pt \
    original_weights=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/resnet18_2s_1e/unlearn/neggrad/resnet18_42_original.pt \
    data.labels=[1,0] data.data_root=./data data.model_name=resnet18 