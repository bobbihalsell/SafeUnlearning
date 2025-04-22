python src/attacks/main_reconstructor.py data=cifar10 reconstructor=invertgrad \
    unlearned_weights=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/neggrad_1s_1e/unlearn/neggrad/resnet18_42_unlearned.pt \
    original_weights=/vol/bitbucket/vb524/v2_safe/safe-unlearning/model/neggrad_1s_1e/unlearn/neggrad/resnet18_42_original.pt \
    data.save_path=./data 

    