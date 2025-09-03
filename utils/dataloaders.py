import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
import numpy as np
import random
import medmnist
from medmnist import PathMNIST, OCTMNIST, TissueMNIST

# Dataloaders

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

def get_data(data_name,
             raw_train_dataset,
             raw_val_dataset,
             raw_test_dataset,
             batch_size, 
             test_batch_size=64,
             n_train_data=3000,
             noise=0,
             seed=42):

    # Set global random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Training transformations
    train_transform = T.Compose([
        T.Resize((28, 28)),
        T.Grayscale(num_output_channels=1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        AddGaussianNoise(mean=0., std=noise),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    # Test/validation transformations
    test_transform = T.Compose([
        T.Resize((28, 28)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    class TransformedDataset(torch.utils.data.Dataset):
        """ Custom wrapper to apply transforms to MedMNIST datasets """
        def __init__(self, medmnist_dataset, transform=None):
            self.data = medmnist_dataset
            self.transform = transform

        def __getitem__(self, idx):
            image, label = self.data[idx]
            if isinstance(image, torch.Tensor):
                image = T.ToPILImage()(image)  # Ensure transform works properly
            if self.transform:
                image = self.transform(image)
            return image, int(label.item())

        def __len__(self):
            return len(self.data)

    def get_class_indices(dataset, class_list):
        """ Extract indices of samples belonging to the specified class list """
        class_to_indices = {cls: [] for cls in class_list}
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in class_to_indices:
                class_to_indices[label].append(i)
        return class_to_indices

    def get_split_indices(indices_list, n_train):
        """ Select first N elements per class """
        return indices_list[:n_train]

    # Define selected classes per dataset
    if data_name == "PathMNIST": classes = [6, 7, 8]
    if data_name == "OCTMNIST": classes = [0, 1, 2, 3]
    if data_name == "TissueMNIST": classes = [4, 5, 6]

    # Datasets with respective transforms
    train_dataset = TransformedDataset(raw_train_dataset, transform=train_transform)
    val_dataset = TransformedDataset(raw_val_dataset, transform=test_transform)
    test_dataset = TransformedDataset(raw_test_dataset, transform=test_transform)

    # Build class-balanced training subset
    base_dataset = TransformedDataset(raw_train_dataset, transform=test_transform)
    class_to_indices = get_class_indices(base_dataset, classes)
    train_indices = []
    for i in classes:
        train_indices += get_split_indices(class_to_indices[i], n_train_data)
    random.shuffle(train_indices)

    train_subset = Subset(train_dataset, train_indices)

    # Validation subset
    class_to_indices = get_class_indices(val_dataset, classes)
    val_indices = []
    for i in classes:
        val_indices += class_to_indices[i]
    random.shuffle(val_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Test subset
    class_to_indices = get_class_indices(test_dataset, classes)
    test_indices = []
    for i in classes:
        test_indices += class_to_indices[i]
    random.shuffle(test_indices)
    test_subset = Subset(test_dataset, test_indices)

    # DataLoaders with fixed seed generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        generator=g
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2
    )

    return train_loader, val_loader, test_loader

def get_data_testing(raw_dataset, 
                     selected_classes,
                     batch_size=64):

    # Test/validation transformations
    test_transform = T.Compose([
        T.Resize((28, 28)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    class TransformedDataset(torch.utils.data.Dataset):
        """ Custom wrapper to apply transforms to MedMNIST datasets """
        def __init__(self, medmnist_dataset, transform=None):
            self.data = medmnist_dataset
            self.transform = transform

        def __getitem__(self, idx):
            image, label = self.data[idx]
            if isinstance(image, torch.Tensor):
                image = T.ToPILImage()(image)
            if self.transform:
                image = self.transform(image)
            return image, int(label.item())

        def __len__(self):
            return len(self.data)

    def get_class_indices(dataset, class_list):
        """ Extract indices of samples belonging to the specified class list """
        class_to_indices = {cls: [] for cls in class_list}
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in class_to_indices:
                class_to_indices[label].append(i)
        return class_to_indices

    # Check that exactly 2 classes are provided
    assert isinstance(selected_classes, (list, tuple)) and len(selected_classes) == 2, \
        "selected_classes must be a list or tuple of exactly 2 class labels."

    # Dataset with respective transforms
    dataset = TransformedDataset(raw_dataset, transform=test_transform)

    # Get only the samples from the selected classes
    class_to_indices = get_class_indices(dataset, selected_classes)
    indices = []
    for cls in selected_classes:
        indices += class_to_indices[cls]

    subset = Subset(dataset, indices)

    # DataLoader
    data_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2
    )

    return data_loader