from datasets.load_datasets import load_train_val_test_datasets
import numpy as np
import os
import shutil
from torchvision.datasets import ImageFolder
from unlearning.utils import ConfigError
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue


class DatasetInitializer:
    def __init__(self, config: DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        dataset_cfg = config['dataset']
        self.dataset_name = dataset_cfg['name']
        self.init_dir = dataset_cfg['init_dir']
        self.save_dir = dataset_cfg['save_dir']
        self.proportion = dataset_cfg['proportion']
        self.val_ratio = dataset_cfg['val_ratio']

        forget_cfg = config['forget']
        self.forget_method = forget_cfg['method']
        self.forget_idx = forget_cfg['forget_idx']
        self.retain_size = forget_cfg.get('retain_size', None)
        self.num = forget_cfg.get('num', None) 

        seed = config.get('seed', 42)
        np.random.seed(seed)

    def load_datasets(self):
        """ Download and save benchmark datasets with name support.

        This will download dataset splits in the save directory.
        """
        # Check if the save path is filled from a previous run and clear it
        if os.path.exists(self.save_dir):
            print("Clearing existing dataset "
                  f"split directory: {self.save_dir}")
            shutil.rmtree(self.save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

        load_train_val_test_datasets(
            dataset_name=self.dataset_name,
            proportion=self.proportion,
            val_ratio=self.val_ratio,
            dataset_load_dir=self.init_dir,
            dataset_save_dir=self.save_dir,
        )

        # Create symlinks for desired retain and forget set images
        if self.forget_method == 'instance':
            self.create_symlink_subsets_by_indices(
                train_dir=self.save_dir + '/train',
                output_dir=self.save_dir,
                forget_indices=self.forget_idx,
                retain_size=self.retain_size
            )
        elif self.forget_method == 'class':
            self.create_symlink_subsets_by_classes(
                train_dir=self.save_dir + '/train',
                output_dir=self.save_dir,
                forget_classes=self.forget_idx
            )
        elif self.forget_method == 'classnum':
            self.create_symlink_subsets_by_classes_number(
                train_dir=self.save_dir + '/train',
                output_dir=self.save_dir,
                forget_classes=self.forget_idx
            )
        else:
            raise ConfigError('Unsupported forget_method, '
                              f'received {self.forget_method}.')

    def create_symlink_subsets_by_indices(self,
                                          train_dir: str,
                                          output_dir: str,
                                          forget_indices: list,
                                          retain_size: int = None,
                                          ):
        """ Create symlink forget and retain subsets"""
        train_dataset = ImageFolder(root=train_dir)

        # If retain_size is None, use all indices except the forget_indices
        all_indices = set(range(len(train_dataset)))
        retain_indices = list(all_indices - set(forget_indices))

        if retain_size is not None:
            retain_indices = list(np.random.choice(retain_indices, min(retain_size, len(retain_indices))))

        def symlink_subset(subset_name, subset_indices):
            subset_dir = os.path.join(output_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            for idx in subset_indices:
                img_path, label = train_dataset.samples[idx]
                class_name = train_dataset.classes[label]
                target_dir = os.path.join(subset_dir, class_name)
                os.makedirs(target_dir, exist_ok=True)

                filename = os.path.basename(img_path)
                link_path = os.path.join(target_dir, filename)

                # Create symlink
                if not os.path.exists(link_path):
                    os.symlink(os.path.abspath(img_path), link_path)

        symlink_subset('forget', forget_indices)
        symlink_subset('retain', retain_indices)

    def create_symlink_subsets_by_classes(self,
                                          train_dir: str,
                                          output_dir: str,
                                          forget_classes: list
                                          ):
        """ Create symlinks if the user wanted to forget an entire class."""
        subdirs = [name for name in os.listdir(train_dir) if
                   os.path.isdir(os.path.join(train_dir, name))]
        forget_classes = [str(label) for label in forget_classes]

        def create_class_symlink(subset_name, label):
            if isinstance(label, int):
                label = str(label)
            subset_dir = os.path.join(output_dir, subset_name, label)
            origin_dir = os.path.join(train_dir, label)
            # Remove existing directory/symlink if it exists
            if os.path.exists(subset_dir):
                if os.path.islink(subset_dir):
                    os.unlink(subset_dir)  # Remove existing symlink
                else:
                    os.rmdir(subset_dir)  # Remove empty directory
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(subset_dir), exist_ok=True)
            abs_origin_dir = os.path.abspath(origin_dir)
            # Create the symlink
            os.symlink(abs_origin_dir, subset_dir, target_is_directory=True)

        for label in forget_classes:
            create_class_symlink('forget', label)

        retain_classes = list(set(subdirs) - set(forget_classes))
        for label in retain_classes:
            create_class_symlink('retain', label)


@hydra.main(version_base=None,
            config_path="config",
            config_name="config")
def main(cfg: DictConfig):
    # Print the config for the user first
    print('============ Run Configuration ============')
    print(OmegaConf.to_yaml(cfg))
    print('============================================')
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise MissingMandatoryValue(
            'Missing the following required arguments in the configuration: '
            f'{missing_keys}. \n'
            'Hint: python file.py key=value sets the appropriate value.')
    app = DatasetInitializer(cfg)
    app.load_datasets()


if __name__ == '__main__':
    main()
