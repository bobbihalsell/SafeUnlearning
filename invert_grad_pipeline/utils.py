import random
import os
import shutil
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from settings import FORGET_ROOT, FORGET_POOL, RETAIN_ROOT, RETAIN_POOL, SEED


# Redo symlinks based on number of unlearning samples
def reset_forget_folder():
    if os.path.exists(FORGET_ROOT):
        shutil.rmtree(FORGET_ROOT)
    os.makedirs(FORGET_ROOT)


def reset_retain_folder():
    if os.path.exists(RETAIN_ROOT):
        shutil.rmtree(RETAIN_ROOT)
    os.makedirs(RETAIN_ROOT)


def create_symlinks(pool,
                    root,
                    n=1,
                    label_output_path="reconstruction_pipeline/labels.txt",
                    create_labels=False):
    """Create symbolic links to the files in the pool directory.
    Args:
        pool (str): Path to the pool directory containing samples.
        root (str): Path to the root directory where symbolic links will
          be created.
        n (int): Number of samples to select.
        label_output_path (str): Path to the output file for labels.
        create_labels (bool): Whether to create a labels file.
    """
    # Sample across all classes
    random.seed(SEED)
    all_samples = []
    for class_name in sorted(os.listdir(pool)):
        class_path = os.path.join(pool, class_name)
        if not os.path.isdir(class_path):
            continue
        for sample_name in sorted(os.listdir(class_path)):
            full_sample_path = os.path.join(class_path, sample_name)
            all_samples.append((class_name, sample_name, full_sample_path))

    chosen = random.sample(all_samples, min(n, len(all_samples)))

    labels = []
    for class_name, sample_name, src in chosen:
        dst_class_path = os.path.join(root, class_name)
        os.makedirs(dst_class_path, exist_ok=True)
        dst = os.path.join(dst_class_path, sample_name)
        os.symlink(os.path.abspath(src), dst)
        labels.append(class_name)

    # Write labels to a file
    if create_labels:
        with open(label_output_path, "w") as f:
            for label in labels:
                f.write(f"{label}\n")


def save_forget_images(output_dir="forget_previews", n=1, max_cols=8, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    for folder_name in sorted(os.listdir(FORGET_ROOT)):
        folder_path = os.path.join(FORGET_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for img_name in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")

        n_images = len(images)
        n_cols = min(max_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        plt.close('all')

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2, n_rows * 2),
                                 dpi=dpi)
        axes = axes.flatten() if isinstance(
            axes, (list, np.ndarray)
        ) else [axes]

        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(str(i), fontsize=8)

        # Hide any unused axes
        for j in range(n_images, len(axes)):
            axes[j].axis('off')

        save_path = os.path.join(output_dir, f"forget_{n}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


def main(n_forget, n_retain):

    label_output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"labels/labels_{n_forget}.txt")

    reset_forget_folder()
    create_symlinks(FORGET_POOL, FORGET_ROOT, n_forget,
                    label_output_path, create_labels=True)

    reset_retain_folder()
    if n_retain > 0:
        create_symlinks(RETAIN_POOL, RETAIN_ROOT, n_retain)
    else:
        pass

    if not os.path.exists(f"forget_previews/forget_{n_forget}.png"):
        save_forget_images(n=n_forget)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_forget", type=int, required=True)
    parser.add_argument("--n_retain", type=int, required=True)
    args = parser.parse_args()

    main(n_forget=args.n_forget,
         n_retain=args.n_retain)
