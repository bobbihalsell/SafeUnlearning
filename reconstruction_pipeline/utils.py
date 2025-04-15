import random
import os
import shutil
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

FORGET_ROOT = "./data/forget"
FORGET_POOL = "./forget_pool"  # where all possible forget samples are
SEED = 42

# Redo symlinks based on number of unlearning samples
def reset_forget_folder():
    if os.path.exists(FORGET_ROOT):
        shutil.rmtree(FORGET_ROOT)
    os.makedirs(FORGET_ROOT)

def create_symlinks(n=1, label_output_path="reconstruction_pipeline/labels.txt"):
    reset_forget_folder()
    random.seed(SEED)
    all_samples = []
    for class_name in os.listdir(FORGET_POOL):
        class_path = os.path.join(FORGET_POOL, class_name)
        if not os.path.isdir(class_path):
            continue
        for sample_name in os.listdir(class_path):
            full_sample_path = os.path.join(class_path, sample_name)
            all_samples.append((class_name, sample_name, full_sample_path))

    # Sample across all classes
    chosen = random.sample(all_samples, min(n, len(all_samples)))

    labels = []
    for class_name, sample_name, src in chosen:
        dst_class_path = os.path.join(FORGET_ROOT, class_name)
        os.makedirs(dst_class_path, exist_ok=True)
        dst = os.path.join(dst_class_path, sample_name)
        os.symlink(os.path.abspath(src), dst)
        labels.append(class_name)

    # Write labels to a file
    with open(label_output_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")



def save_forget_images(output_dir="forget_previews", n=1, max_cols=8, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    for folder_name in os.listdir(FORGET_ROOT):
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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), dpi=dpi)
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

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



def main(n, label_output_path):
    create_symlinks(n, label_output_path)

    if not os.path.exists(f"forget_previews/forget_{n}.png"):
        save_forget_images(n=n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--label_output_path", type=str, default="labels.txt")
    args = parser.parse_args()

    main(n=args.n, label_output_path=args.label_output_path)
