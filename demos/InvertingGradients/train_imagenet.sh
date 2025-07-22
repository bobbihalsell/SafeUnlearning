python3 src/datasets/main.py \
    dataset=imagenet \
    forget=filename \
    dataset.init_path=demos/GGL/imagenet_example_data \
    dataset.save_path=demos/GGL/imagenet_example_split \
    dataset.val_ratio=0.1 \
    forget.forget_filenames="[n01532829_1.JPEG]" \
    experiment_name=ggl

python3 src/train/main.py \
    dataset=imagenet \
    model=torchvision \
    model.model_name=resnet18 \
    trainer=trainer \
    dataset.save_path=demos/GGL/imagenet_example_split \
    model.pretrained=true \
    trainer.model_save_dir=demos/InvertingGradients \
    trainer.cfg.epochs=0 \
    +wandb=default \
    experiment_name=pretrained_resnet