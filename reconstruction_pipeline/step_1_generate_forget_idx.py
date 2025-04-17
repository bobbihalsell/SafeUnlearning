import random
import json

    
if __name__ == "__main__":
    random.seed(42)
    idx = random.sample(range(36000), 32)

    print(idx)
    with open("reconstruction_pipeline/forget_indices.json", "w") as f:
        json.dump(idx, f)
    
    print("Forget indices saved to forget_indices.json.")