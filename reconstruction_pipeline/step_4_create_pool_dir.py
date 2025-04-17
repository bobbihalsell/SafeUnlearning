## This script creates the pool directories for forget and retain samples

import os
import shutil

if __name__ == "__main__":
        if not os.path.exists("reconstruction_pipeline/labels"):
                os.makedirs("reconstruction_pipeline/labels")

        if not os.path.exists("./forget_pool"):
                os.rename("./data/forget", "./data/forget_pool")
                shutil.move("./data/forget_pool", "./")
        
        if not os.path.exists("./retain_pool"):
                os.rename("./data/retain", "./data/retain_pool")
                shutil.move("./data/retain_pool", "./")
