import os


def get_dataset_name(local_dataset_path: str) -> str:
    dataset_filename = os.path.splitext(os.path.basename(local_dataset_path))[0]
    return dataset_filename.replace('_dataset', '')
