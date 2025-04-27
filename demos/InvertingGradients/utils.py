import os


def get_forget_labels(forget_path):
    """
    Iterate through all images in subfolders of
    forget_labels_path and return a list of labels
    """
    image_label_pairs = []

    for label in os.listdir(forget_path):
        label_path = os.path.join(forget_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith((".png", ".jpg", ".jpeg")):
                    image_label_pairs.append(int(label))

    return image_label_pairs


def get_retain_size(retain_path):
    count = 0
    for root, subdirs, files in os.walk(retain_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                count += 1
    return count
