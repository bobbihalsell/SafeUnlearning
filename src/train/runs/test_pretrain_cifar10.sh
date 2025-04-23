python src/train/main.py dataset=cifar10 model=resnet18 wandb=default trainer=default dataset.save_path=./data model.pretrained=true model.save_dir=./artifacts/models/cifar10 trainer.epochs=1
python src/train/main.py dataset=cifar10 model=resnet18 trainer=default dataset.save_path=./data model.pretrained=true model.save_dir=./artifacts/models/cifar10 trainer.epochs=1
