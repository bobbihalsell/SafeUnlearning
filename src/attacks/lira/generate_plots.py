import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots(exp_dir):
    memberships = [file for file in os.listdir(exp_dir) if file.endswith("membership.npy")]
    if not memberships:
        print("Couldn't find any membership scores to plot. Skipping..")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    for data_path in memberships:
        data = np.load(exp_dir / data_path, allow_pickle=True)
        label = data_path.split("/")[-1][:-4].replace("_membership", "")

        scores = []
        for i in range(len(data)):
            if data[i]:
                scores.append(np.mean(data[i]))

        sns.kdeplot(scores, ax=axs[0], label=label.title())
        axs[1].bar(label.title(), np.mean(scores), yerr=np.std(scores))

    xmin, xmax, ymin, ymax = axs[0].axis()
    axs[0].vlines(0.5, ymin, ymax, color="black", linestyle="--")
    axs[0].axis([xmin, xmax, ymin, ymax])
    axs[0].set_xlabel("Per-sample Membership Inference Accuracy")

    xmin, xmax, ymin, ymax = axs[1].axis()
    axs[1].axis([xmin, xmax, ymin, ymax])
    axs[1].set_ylabel("Average Membership Inference Accuracy")

    axs[0].legend()
    plt.savefig(exp_dir / "plots.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    exp_dir = Path(f"artifacts/attacks/{args.experiment_name}")
    generate_plots(exp_dir)
