def get_label_map(data_name):

    label_maps = {
        "PathMNIST":   {6: 0, 7: 1, 8: 2},
        "OCTMNIST":    {0: 0, 1: 1, 2: 2, 3: 3},
        "TissueMNIST": {4: 0, 5: 1, 6: 2}
    }

    if data_name not in label_maps:
        raise ValueError(f"Unsupported dataset: {data_name}")

    return label_maps[data_name]

def get_n_labels(data_name):

    n_labels = {
        "PathMNIST":   3,
        "OCTMNIST":    4,
        "TissueMNIST": 3
    }

    if data_name not in n_labels:
        raise ValueError(f"Unsupported dataset: {data_name}")

    return n_labels[data_name]