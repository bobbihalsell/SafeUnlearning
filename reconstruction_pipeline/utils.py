import random
import os
import shutil
import argparse

FORGET_ROOT = "./data/forget"
FORGET_POOL = "./data/forget_pool"  # where all possible forget samples are
SEED = 42

# Redo symlinks based on number of unlearning samples
def reset_forget_folder():
    random.seed(SEED)
    if os.path.exists(FORGET_ROOT):
        shutil.rmtree(FORGET_ROOT)
    os.makedirs(FORGET_ROOT)

def create_symlinks(n=1, label_output_path="reconstruction_pipeline/labels.txt"):
    reset_forget_folder()

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


def main(n, label_output_path):
    # Create move data to pool 
    if not os.path.exists("./data/forget_pool"):
        os.rename("./data/forget", "./data/forget_pool")

    os.makedirs("./data/forget", exist_ok=True)

    create_symlinks(n, label_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--label_output_path", type=str, default="labels.txt")
    args = parser.parse_args()

    main(n=args.n, label_output_path=args.label_output_path)
