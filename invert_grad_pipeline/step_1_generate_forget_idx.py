import random
import json
import os

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))

    random.seed(42)

    idx = random.sample(range(36000), 32)
    print(idx)

    file_path = os.path.join(dir, "forget_indices.json")
    with open(file_path, "w") as f:
        json.dump(idx, f)
    print("Forget indices saved to forget_indices.json.")
