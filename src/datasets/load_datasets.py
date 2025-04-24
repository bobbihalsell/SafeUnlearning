from torch.utils.data import Subset
import torchvision.datasets as datasets
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import shutil
import os
import torch
import torchvision


def get_filenames_and_labels(dataset_dir: str):
    """
    Traverse ImageFolder-style directory and collect image paths and labels.

    The class folders must correspond to the label index of the class.
    E.g.
        train/ <-- dataset_dir
        ├── 0/
        ├── 1/
        ...

    Args:
        dataset_dir (str): The parent directory of the class folders.

    """
    image_paths = []
    labels = []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            # Skip over accidental files left at the class folder level
            continue
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()  # Verify that it's an image
                image_paths.append(fpath)
                labels.append(int(class_name))
            except Exception:
                print(f'Detected invalid image at {fpath}. Skipping...')
                continue

    return image_paths, labels


def stratified_split_filenames(filenames, labels, proportion):
    """
    Perform stratified sampling on image filenames.
    """
    filenames = np.array(filenames)
    labels = np.array(labels)

    selected, remainder = [], []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        split = int(len(cls_indices) * proportion)
        selected.extend(cls_indices[:split])
        remainder.extend(cls_indices[split:])
    return filenames[selected], filenames[remainder]


def create_symlinks(file_list, split_dir):
    for src in file_list:
        label = os.path.basename(os.path.dirname(src))
        dst_dir = os.path.join(split_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)


def download_cifar_datasets(dataset_name: str,
                            download_root: str,
                            save_dir: str,
                            transform: torchvision.transforms = None):
    """ Load and save CIFAR datasets in ImageFolder format. """
    # For CIFAR datasets, download if necessary and filter
    if dataset_name == 'cifar10':
        raw_train = datasets.CIFAR10(root=download_root,
                                     train=True,
                                     download=True,
                                     transform=transform)
        raw_test = datasets.CIFAR10(root=download_root,
                                    train=False,
                                    download=True,
                                    transform=transform)

        save_cifar_as_imagefolder(raw_train, raw_test, download_root, save_dir)

    elif dataset_name == 'cifar100':
        raw_train = datasets.CIFAR100(root=download_root,
                                      train=True,
                                      download=True,
                                      transform=transform)
        raw_test = datasets.CIFAR100(root=download_root,
                                     train=False,
                                     download=True,
                                     transform=transform)

        save_cifar_as_imagefolder(raw_train, raw_test, download_root, save_dir)

    elif dataset_name == 'cifar5':
        raw_train = datasets.CIFAR10(root=download_root,
                                     train=True,
                                     download=True,
                                     transform=transform)
        raw_test = datasets.CIFAR10(root=download_root,
                                    train=False,
                                    download=True,
                                    transform=transform)

        # Filter for classes 0–4
        train_indices = [i for i, (_, label) in
                         enumerate(raw_train) if label < 5]
        test_indices = [i for i, (_, label) in
                        enumerate(raw_test) if label < 5]
        raw_train = Subset(raw_train, train_indices)
        raw_test = Subset(raw_test, test_indices)

        save_cifar_as_imagefolder(raw_train, raw_test, download_root, save_dir)

    else:
        raise Exception(f'{dataset_name} is an unsupported dataset.')


def save_cifar_as_imagefolder(raw_train,
                              raw_test,
                              download_root,
                              dataset_save_dir):
    """ Save a CIFAR dataset as ImageFolder.

    Create unique file names for each image file.
    """
    if os.path.exists(dataset_save_dir):
        print("Clearing existing dataset "
              f"save directory: {dataset_save_dir}")
        shutil.rmtree(dataset_save_dir)
    to_pil = ToPILImage()
    os.makedirs(download_root, exist_ok=True)
    os.makedirs(dataset_save_dir, exist_ok=True)

    counter = 0
    for split_name, dataset in [("train", raw_train), ("test", raw_test)]:
        for img, label in dataset:
            if isinstance(img, torch.Tensor):
                img = to_pil(img)
            # Path: <dataset_save_dir>/<split_name>/<class>/
            class_dir = os.path.join(dataset_save_dir, split_name, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{counter:05}.png"))
            counter += 1
    print(f"Saved {counter} images to {dataset_save_dir} "
          "in ImageFolder format.")
