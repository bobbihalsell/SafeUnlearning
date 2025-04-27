python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=neggradplus \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data \
    unlearner.cfg.beta=0.97 \
    unlearner.cfg.momentum=0.9 \
    id='001' \
    experiment_name=torchhub_test \
    verbose=false