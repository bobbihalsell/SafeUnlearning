from datasets.load_datasets import (download_cifar_datasets,
                                    get_filenames_and_labels,
                                    stratified_split_filenames,
                                    create_symlinks)
from train.image_loading import RobustImageFolder
from datasets.config_validator import DatasetValidator
from utils import ConfigError
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from PIL import Image
from utils import set_seed
import os
import numpy as np
import shutil
import hydra


class DatasetInitializer(DatasetValidator):
    """
    Initializes and prepares datasets for experiments, including downloading,
    preprocessing, splitting into train/val sets, and generating forget/retain subsets.
    This class extends DatasetValidator to support structured dataset initialization.

    Attributes:
        config (dict): Configuration dictionary loaded from OmegaConf.
    """
    def __init__(self, config: DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        super().__init__(config)
        seed = config['seed']
        set_seed(seed)

    def load_datasets(self):
        """ 
        Download and save benchmark datasets with name support.
        This will download dataset splits in the save directory.

        Raises:
          ConfigError: If an unsupported dataset or forget method is specified.
        """
        if 'cifar' in self.dataset_name:
            if self.dataset_load_method == 'torchvision':
                download_cifar_datasets(
                    dataset_name=self.dataset_name,
                    download_root=self.binaries_download_dir,
                    save_dir=self.init_path,
                )

        else:
            if self.dataset_load_method == 'torchvision':
                raise ConfigError(
                    "Sorry, torchvision download is not enabled yet for "
                    "non-CIFAR datasets."
                )

        # Filter a proportion of the train dataset filenames
        filenames, labels = get_filenames_and_labels(self.init_path +
                                                     '/train')
        if os.path.exists(self.init_path + '/test'):
            test_filenames, test_labels = get_filenames_and_labels(
                                    self.init_path + '/test')

        remaining_filenames, _ = stratified_split_filenames(
            filenames,
            labels,
            proportion=self.proportion)

        # Retrieve the labels from the remaining portion of the train dataset
        remaining_labels = [self._retrieve_label_from_filepath(fp) for
                            fp in remaining_filenames]

        # Perform a train/val split from the remaining filenames
        train_filenames, val_filenames = stratified_split_filenames(
            remaining_filenames,
            remaining_labels,
            proportion=1-self.val_ratio
        )
        # Check if the splits path is filled from a previous run and clear it
        self._reinitialize_splits_dir()

        # Create train/val symlinks from self.save_path
        create_symlinks(train_filenames, self.save_path + '/train')
        create_symlinks(val_filenames, self.save_path + '/val')
        if os.path.exists(self.init_path + '/test'):
            create_symlinks(test_filenames, self.save_path + '/test')

        # Create symlinks for desired retain and forget set images
        if self.forget_method == 'random_n':
            self.create_forget_retain_symlinks_by_random_n(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path,
                forget_size=self.forget_size,
                retain_size=self.retain_size
            )
        elif self.forget_method == 'class':
            self.create_forget_retain_symlinks_by_classes(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path,
                forget_classes=self.forget_idx
            )
        elif self.forget_method == 'classnum':
            self.create_forget_retain_symlinks_by_class_number(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path,
                forget_classes=self.forget_idx
            )
        elif self.forget_method == 'filename':
            self.create_forget_retain_symlinks_by_filename(
                train_dir=self.save_path + '/train',
                output_dir=self.save_path
            )
        else:
            raise ConfigError('Unsupported forget_method, '
                              f'received {self.forget_method}.')

    def _reinitialize_splits_dir(self):
        """
        Remove and recreate an empty split directory.

        Prevents unintended errors across different experiment runs,
        if different splits are intended.

        Returns:
            None
        """
        if os.path.exists(self.save_path):
            print("Clearing existing dataset "
                  f"split directory: {self.save_path}")
            shutil.rmtree(self.save_path)

        os.makedirs(self.save_path, exist_ok=True)

    def _retrieve_label_from_filepath(self, fp: str):
        """
        Retrieve the label from a filepath from ImageFolder format.
        e.g. retrieve '1' from ./cifar10/train/1/xyz.png

        Args:
           fp (str): Full path to the image file.

        Returns:
            str: The class label as a string.
        """
        return fp.split('/train/')[1][0]

    def create_forget_retain_symlinks_by_random_n(
            self,
            train_dir: str,
            output_dir: str,
            forget_size: int,
            retain_size: int = None
    ):
        """
        Create symlink forget and retain subsets

        Args:
        train_dir (str): Path to the training data directory.
        output_dir (str): Directory where symlinks will be created.
        forget_size (int): Number of samples to forget.
        retain_size (int, optional): Number of samples to retain. If None, uses the rest.

        Raises:
        ConfigError: If forget_size is larger than the dataset size.
        """
        train_dataset = RobustImageFolder(root=train_dir)
        if forget_size > len(train_dataset):
            raise ConfigError("forget_size is larger than dataset size")
        forget_indices = np.random.choice(range(len(train_dataset)),
                                          size=forget_size,
                                          replace=False)
        all_indices = set(range(len(train_dataset)))
        # If retain_size is None, use all indices except the forget_indices
        retain_indices = list(all_indices - set(forget_indices))
        if retain_size is not None:
            retain_indices = list(
                np.random.choice(
                    retain_indices,
                    min(retain_size, len(retain_indices)),
                    replace=False
                    )
                )

        def symlink_subset(subset_name, subset_indices):
            subset_dir = os.path.join(output_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            for idx in subset_indices:
                img_path, label = train_dataset.samples[idx]
                class_name = str(label)
                target_dir = os.path.join(subset_dir, class_name)
                os.makedirs(target_dir, exist_ok=True)

                filename = os.path.basename(img_path)
                link_path = os.path.join(target_dir, filename)

                os.symlink(os.path.abspath(img_path), link_path)

        symlink_subset('forget', forget_indices)
        symlink_subset('retain', retain_indices)

    def create_forget_retain_symlinks_by_classes(
            self,
            train_dir: str,
            output_dir: str,
            forget_classes: list
    ):
        """
        Create symlinks if the user wanted to forget an entire class.
        Args:
            train_dir (str): Path to the training data directory.
            output_dir (str): Directory to store symlinks.
            forget_classes (list): List of class labels to forget.

        """
        subdirs = [name for name in os.listdir(train_dir) if
                   os.path.isdir(os.path.join(train_dir, name))]
        forget_classes = [str(label) for label in forget_classes]

        def create_class_symlink(subset_name, label):
            if isinstance(label, int):
                label = str(label)
            subset_dir = os.path.join(output_dir, subset_name, label)
            origin_dir = os.path.join(train_dir, label)
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

    def create_forget_retain_symlinks_by_class_number(
            self,
            train_dir: str,
            output_dir: str,
            forget_classes: dict
    ):
        """
        Create symlinks for n samples from each class to forget.

        Args:
            train_dir (str): Path to the training data directory.
            output_dir (str): Directory to store symlinks.
            forget_classes (dict): Dictionary mapping class labels to number of
                                   samples to forget.

        """
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
                # If src_file is itself a symlink, get the real path
                if os.path.islink(src_file):
                    real_src = os.path.realpath(src_file)
                    os.symlink(real_src, dst_file)
                else:
                    os.symlink(src_file, dst_file)

        def create_dir_symlink(subset_name, class_label):
            # Create the target directory (retain classes)
            subset_dir = os.path.join(output_dir, subset_name, class_label)
            origin_dir = os.path.join(train_dir, class_label)

            os.makedirs(os.path.dirname(subset_dir), exist_ok=True)
            abs_origin_dir = os.path.abspath(origin_dir)
            # Create the symlink
            os.symlink(abs_origin_dir, subset_dir, target_is_directory=True)

        for class_label in subdirs:
            if class_label in forget_classes:
                num_samples_to_forget = int(forget_classes[class_label])
                class_dir = os.path.join(train_dir, class_label)

                all_files = []
                for file in os.listdir(class_dir):
                    try:
                        with Image.open(os.path.join(class_dir, file)) as img:
                            img.verify()  # Verify that it's an image
                        all_files.append(file)
                    except Exception:
                        print(f'Encountered invalid image file {file}. '
                              'Skipping...')
                        continue

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

    def create_forget_retain_symlinks_by_filename(
            self,
            train_dir: str,
            output_dir: str,
    ):
        """
        Create forget/retain symlinks by user-defined filenames.

        Uses self.forget_filenames and self.retain_filenames and creates
        symlinks based on the user-provided values in the config.

        Args:
            train_dir (str): Path to training directory.
            output_dir (str): Directory to store symlinks.
        """
        found = []
        for label in os.listdir(train_dir):
            forget_dir = os.path.join(output_dir, 'forget', label)
            retain_dir = os.path.join(output_dir, 'retain', label)

            label_dir = os.path.join(train_dir, label)
            if not os.path.isdir(label_dir):
                # Ignore unexpected files at class folder level
                continue
            for fp in os.listdir(label_dir):
                try:
                    with Image.open(os.path.join(label_dir, fp)) as img:
                        img.verify()  # Verify that it's an image
                except Exception:
                    print(f'Encountered invalid image file {fp}. Skipping...')
                    continue

                if self.retain_filenames is None:
                    # All files not in forget_filenames are in retain
                    src_file = os.path.abspath(os.path.join(train_dir,
                                                            label,
                                                            fp))
                    if os.path.basename(fp) in self.forget_filenames:
                        found.append(fp)
                        os.makedirs(forget_dir, exist_ok=True)
                        # Create symlink to forget set
                        dst_file = os.path.join(forget_dir, fp)
                        os.symlink(src_file, dst_file)
                    else:
                        os.makedirs(retain_dir, exist_ok=True)
                        # Create symlink to retain set
                        dst_file = os.path.join(retain_dir, fp)
                        os.symlink(src_file, dst_file)

                else:
                    # Only specified files to be in retain_filenames
                    if os.path.basename(fp) in self.forget_filenames:
                        found.append(fp)
                        # Create symlink to forget set
                        os.makedirs(forget_dir, exist_ok=True)
                        src_file = os.path.abspath(os.path.join(train_dir,
                                                                label,
                                                                fp))
                        dst_file = os.path.join(forget_dir, fp)
                        os.symlink(src_file, dst_file)
                    if os.path.basename(fp) in self.retain_filenames:
                        found.append(fp)
                        os.makedirs(retain_dir, exist_ok=True)
                        src_file = os.path.abspath(os.path.join(train_dir,
                                                                label,
                                                                fp))
                        dst_file = os.path.join(retain_dir, fp)
                        os.symlink(src_file, dst_file)

        # Check to see the set difference between found and retain + forget
        if self.retain_filenames is None:
            forget_retain_filenames = self.forget_filenames
        else:
            forget_retain_filenames = (self.forget_filenames +
                                       self.retain_filenames)

        missing = set(forget_retain_filenames) - set(found)
        if missing:
            print('Warning: Failed to find the following filenames you have '
                  f'specified as forget/retain in the dataset: {missing}')

    def del_parent_dir(self):
        if self.init_path == 'tmp_dir':
            # Remove the tmp_dir directory
            shutil.rmtree(self.init_path, ignore_errors=True)
            print("Deleted the temporary directory holding the "
                  "original training and test data.")


@hydra.main(version_base=None,
            config_path="../../configs",
            config_name="datasets")
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
    app.del_parent_dir()


if __name__ == '__main__':
    main()
