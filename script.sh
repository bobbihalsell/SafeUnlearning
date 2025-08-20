
export PYTHONPATH="/vol/bitbucket/blh124/unlpaper/safe-unlearning/src:$PYTHONPATH"
source venv/bin/activate

# data split used in pgu
python src/datasets/main.py forget=classnum dataset=cifar10 \
    dataset.load_method=torchvision \
    forget.forget_idx="{0:500}" \
    dataset.binaries_download_dir=./bin \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test

### TO RUN PGU
python src/unlearning/main.py dataset=cifar10 model=class unlearner=pgu \
    dataset.save_path=./data1 \
    id='001' \
    experiment_name=allcnn_test \
    verbose=true
###

# neggradplus
python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=neggradplus \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data1 \
    unlearner.cfg.beta=0.97 \
    unlearner.cfg.momentum=0.9 \
    id='001' \
    experiment_name=torchhub_test \
    verbose=true

Initial Forget Loss: 0.0068. Acc: 99.80%. || Initial Val Loss: 0.2815. Acc: 92.59%. || Initial Retain Loss: 0.0053. Acc: 99.95%. || 
Epoch 1 Forget Loss: 0.7590 Acc: 80.80%.  || Val Loss: 0.4467 Acc: 88.43%.  || Retain Loss: 0.0900 Acc: 96.91%.  || 
Epoch 2 Forget Loss: 0.7707 Acc: 84.00%.  || Val Loss: 0.4100 Acc: 89.41%.  || Retain Loss: 0.0728 Acc: 97.45%.  || 
Epoch 3 Forget Loss: 4.3670 Acc: 57.80%.  || Val Loss: 0.5986 Acc: 87.07%.  || Retain Loss: 0.1432 Acc: 95.51%.  || 
Epoch 4 Forget Loss: 7.5090 Acc: 60.60%.  || Val Loss: 0.7858 Acc: 85.05%.  || Retain Loss: 0.2667 Acc: 92.86%.  || 
Model state_dict saved to artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_unlearned_001.pt


# scrub
python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=scrub \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data1 \
    unlearner.cfg.min_epochs=4 \
    unlearner.cfg.max_epochs=1 \
    unlearner.cfg.alpha=0.5 \
    unlearner.cfg.gamma=0.5 \
    unlearner.cfg.sep_epochs=true \
    id='001' \
    experiment_name=torchhub_test \
    verbose=true

Initial Forget Loss: 0.0068. Acc: 99.80%. || Initial Val Loss: 0.2815. Acc: 92.59%. || Initial Retain Loss: 0.0053. Acc: 99.95%. || 
Epoch 1 Forget Loss: 0.1792 Acc: 94.00%.  || Val Loss: 0.4518 Acc: 86.19%.  || Retain Loss: 0.2351 Acc: 91.60%.  || 
Epoch 2 Forget Loss: 0.2062 Acc: 94.80%.  || Val Loss: 0.4500 Acc: 85.50%.  || Retain Loss: 0.3016 Acc: 89.41%.  || 
Epoch 3 Forget Loss: 0.2288 Acc: 92.40%.  || Val Loss: 0.3141 Acc: 89.49%.  || Retain Loss: 0.1736 Acc: 94.22%.  || 
Epoch 4 Forget Loss: 0.2194 Acc: 93.80%.  || Val Loss: 0.3197 Acc: 89.23%.  || Retain Loss: 0.1790 Acc: 94.11%.  || 

# pgu 
python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=pgu \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data1 \
    id='001' \
    experiment_name=torchhub_test \
    verbose=true

# sgru
python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=sgru \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data1 \
    id='001' \
    experiment_name=torchhub_test \
    verbose=true



python3 src/attacks/GGL/test.py
