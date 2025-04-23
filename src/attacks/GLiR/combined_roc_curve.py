import os
import json
import matplotlib.pyplot as plt

def load_roc_data(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "roc_data.json":
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    results.append(data)
    return results

def plot_all_rocs(roc_data_list, savepath=None):
    plt.figure(figsize=(10, 6))
    for d in roc_data_list:
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f'{d["label"]} (AUC = {d["auc"]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)

    if savepath:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, "combined_roc.png"))
        print(f"Saved combined plot to: {savepath}")
    plt.show()

# if __name__ == "__main__":
#     base_dir = "outputs/" 
#     roc_data_list = load_roc_data(base_dir)
#     plot_all_rocs(roc_data_list, savepath="outputs/combined")
