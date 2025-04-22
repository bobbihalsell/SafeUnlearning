from datasets.load_datasets import load_train_val_test_datasets
import numpy as np
import os
import shutil
from train.image_loading import RobustImageFolder
from utils import ConfigError
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
import json
from config_validator import DatasetValidator


class DatasetInitializer(DatasetValidator):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        config = OmegaConf.to_container(config, resolve=True)
        np.random.seed(self.seed)

    def rename_imagenet_folders(self):
        """Create a copy of the ImageNet subset with folder names
        mapped to label indices, preserving the original dataset.
        """
        base_dir = os.path.dirname(__file__)
        mapping_path = os.path.join(base_dir,
                                    'imagenet',
                                    'imagenet_1k_mappings.json')

        with open(mapping_path, 'r') as f:
            default_imagenet_mapping = json.load(f)

        source_root = self.init_path
        target_root = self.init_path.rstrip('/') + '_renamed'

        os.makedirs(target_root, exist_ok=True)

        split_names = ['train', 'val']
        for split in split_names:
            src_split = os.path.join(source_root, split)
            tgt_split = os.path.join(target_root, split)

            if not os.path.exists(src_split):
                continue

            os.makedirs(tgt_split, exist_ok=True)

            folder_names = [i for i in
                            sorted(os.listdir(src_split)) if
                            not i.startswith('.')]
            for folder in folder_names:
                try:
                    label_index = default_imagenet_mapping[folder]
                except KeyError:
                    raise KeyError(
                        "Folder name not found in "
                        f"ImageNet-1K Class IDs: {folder}"
                    )

                src_dir = os.path.join(src_split, folder)
                tgt_dir = os.path.join(tgt_split, str(label_index))

                if os.path.exists(tgt_dir):
                    raise FileExistsError(
                        f"Target directory already exists: {tgt_dir}")

                shutil.copytree(src_dir, tgt_dir)

        print(f"Renamed dataset created at: {target_root}")

    def load_datasets(self):
        """ Download and save benchmark datasets with name support.

        This will download dataset splits in the save directory.
        """
        # Check if the save path is filled from a previous run and clear it
        if os.path.exists(self.save_path):
            print("Clearing existing dataset "
                  f"split directory: {self.save_path}")
            shutil.rmtree(self.save_path)

        os.makedirs(self.save_path, exist_ok=True)

        init_path = self.init_path
        if self.dataset_name == 'imagenet':
            # Rename the folders to match label indices
            self.rename_imagenet_folders()
            init_path = self.init_path.rstrip('/') + '_renamed'

        load_train_val_test_datasets(
            dataset_name=self.dataset_name,
            proportion=self.proportion,
            val_ratio=self.val_ratio,
            dataset_load_dir=init_path,
            dataset_save_dir=self.save_path,
        )

        # Create symlinks for desired retain and forget set images
        if self.forget_method == 'instance':
            self.create_symlink_subsets_by_indices(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path,
                forget_indices=self.forget_idx,
                retain_size=self.retain_size
            )
        elif self.forget_method == 'class':
            self.create_symlink_subsets_by_classes(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path,
                forget_classes=self.forget_idx
            )
        elif self.forget_method == 'classnum':
            self.create_symlink_subsets_by_class_number(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path,
                forget_classes=self.forget_idx
            )
        else:
            raise ConfigError('Unsupported forget_method, '
                              f'received {self.forget_method}.')

        if self.dataset_name == 'imagenet':
            # Remove the interim directory created to avoid mutating original
            renamed_dir = self.init_path.rstrip('/') + '_renamed'
            if os.path.exists(renamed_dir):
                shutil.rmtree(renamed_dir)

    def create_symlink_subsets_by_indices(self,
                                          train_dir: str,
                                          output_dir: str,
                                          forget_indices: list,
                                          retain_size: int = None,
                                          ):
        """ Create symlink forget and retain subsets"""
        train_dataset = RobustImageFolder(root=train_dir)

        # If retain_size is None, use all indices except the forget_indices
        all_indices = set(range(len(train_dataset)))
        retain_indices = list(all_indices - set(forget_indices))

        if retain_size is not None:
            retain_indices = list(
                np.random.choice(
                    retain_indices,
                    min(retain_size, len(retain_indices))
                    )
                )

        def symlink_subset(subset_name, subset_indices):
            subset_dir = os.path.join(output_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            for idx in subset_indices:
                img_path, label = train_dataset.samples[idx]
                # class_name = train_dataset.classes[label]
                class_name = str(label)
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

    def create_symlink_subsets_by_class_number(
            self,
            train_dir: str,
            output_dir: str,
            forget_classes: dict
    ):
        """ Create symlinks for n samples from each class to forget."""
        subdirs = [name for name in os.listdir(train_dir) if
                   os.path.isdir(os.path.join(train_dir, name))]
        forget_classes = {
            str(label): str(count) for
            label, count in forget_classes.items()}

        def create_file_symlinks(subset_name, class_label, files):
            # Create the target directory (forget classes)
            target_dir = os.path.join(output_dir, subset_name, class_label)
            os.makedirs(target_dir, exist_ok=True)

            for file in files:
                src_file = os.path.abspath(os.path.join(train_dir,
                                                        class_label,
                                                        file))
                dst_file = os.path.join(target_dir, file)

                # Remove existing symlink if it exists
                if os.path.exists(dst_file):
                    if os.path.islink(dst_file):
                        os.unlink(dst_file)  # Remove existing symlink
                    else:
                        os.remove(dst_file)  # Remove existing file
                # Create the symlink
                os.symlink(src_file, dst_file)

        def create_dir_symlink(subset_name, class_label):
            # Create the target directory (retain classes)
            subset_dir = os.path.join(output_dir, subset_name, class_label)
            origin_dir = os.path.join(train_dir, class_label)

            # Remove existing directory/symlink if it exists
            if os.path.exists(subset_dir):
                if os.path.islink(subset_dir):
                    os.unlink(subset_dir)  # Remove existing symlink
                else:
                    os.rmdir(subset_dir)  # Remove empty directory

            os.makedirs(os.path.dirname(subset_dir), exist_ok=True)
            abs_origin_dir = os.path.abspath(origin_dir)
            # Create the symlink
            os.symlink(abs_origin_dir, subset_dir, target_is_directory=True)

        for class_label in subdirs:
            if class_label in forget_classes:
                num_samples_to_forget = int(forget_classes[class_label])
                class_dir = os.path.join(train_dir, class_label)

                all_files = [f for f in os.listdir(class_dir) if
                             os.path.isfile(os.path.join(class_dir, f))]

                num_samples_to_forget = min(num_samples_to_forget,
                                            len(all_files))

                if num_samples_to_forget > 0:
                    # Randomly select files to forget
                    forget_files = list(np.random.choice(
                        all_files,
                        size=num_samples_to_forget,
                        replace=False
                    ))

                    create_file_symlinks('forget', class_label, forget_files)
                    retain_files = list(set(all_files) - set(forget_files))
                    create_file_symlinks('retain', class_label, retain_files)
                else:
                    # If no samples to forget, link entire class to retain
                    create_dir_symlink('retain', class_label)
            else:
                # For classes not in forget, link the whole directory to retain
                create_dir_symlink('retain', class_label)


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
