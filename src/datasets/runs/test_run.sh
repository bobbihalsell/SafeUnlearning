# instance
# python src/datasets/main.py forget=instance dataset=cifar10 forget.forget_idx="[216, 32245, 27206, 10863, 2190, 31849, 25408, 33504]" forget.retain_size=15 dataset.init_dir=./raw dataset.save_dir=./data
# class
# python src/datasets/main.py forget=class dataset=cifar10 forget.forget_idx="[5]" dataset.init_dir=./raw dataset.save_dir=./data
# class num
python src/datasets/main.py forget=classnum dataset=cifar10 forget.forget_idx="{1:500}" dataset.init_dir=./raw dataset.save_dir=./data