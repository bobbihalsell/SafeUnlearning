import random
import os



def main():
    random.seed(42)

    # generate 32 numbers for the unlearning
    idx = random.sample(range(36000), 32)
    idx_str = "[" + ",".join(str(i) for i in idx) + "]"

    dataset_dir = "./data"
    print(idx_str)

    os.system(
        f"python src/datasets/main.py "
        f"dataset=cifar10 forget=instance "
        f"dataset.init_dir=./raw dataset.save_dir={dataset_dir} "
        f"forget.forget_idx={str(idx_str)} forget.retain_size=20"
    )

    print("Data splitting completed.")
    print("Beginning training...")
    os.system(
        f"python src/train/main.py "
        f"model=resnet18 dataset=cifar10 "
        f"dataset.load_dir={dataset_dir} "
        f"model.pretrained=False model.num_classes=10 model.save_dir=./model "
        f"model.train_cfg.epochs=15 model.train_cfg.weight_decay=0 model.train_cfg.lr=0.01 " 
    )
    
if __name__ == "__main__":
    main()