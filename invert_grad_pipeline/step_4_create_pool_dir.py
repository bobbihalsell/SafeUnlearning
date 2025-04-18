# This script creates the pool directories for forget and retain samples

import os
import shutil

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f"{current_dir}/labels"):
        os.makedirs(f"{current_dir}/labels")

    if not os.path.exists("./forget_pool"):
        os.rename("./data/forget", "./data/forget_pool")
        shutil.move("./data/forget_pool", "./")

    if not os.path.exists("./retain_pool"):
        os.rename("./data/retain", "./data/retain_pool")
        shutil.move("./data/retain_pool", "./")
