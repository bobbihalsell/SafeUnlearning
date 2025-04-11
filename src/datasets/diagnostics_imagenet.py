import os
import json


def create_class_to_index_mapping(dataset_dir: str):
    """ Map original dataset folder names to the label index for training."""
    class_mapping = {}
    # Assuming the folder structure is like: dataset_dir/train/class_x/
    class_names = [cls for cls in
                   sorted(os.listdir(os.path.join(dataset_dir, "train"))) if
                   not cls.startswith('.')]
    for idx, class_name in enumerate(class_names):
        class_mapping[idx] = class_name

    with open('./imagenet_class_to_id.json', 'w') as f:
        json.dump(class_mapping, f)
