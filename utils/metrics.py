import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_multiclass_auc(targets, probabilities):
    """
    Compute the multiclass AUC (macro-average) given target labels and predicted class probabilities.

    Args:
        targets (list or np.ndarray): Ground-truth class labels, shape (N,)
        probabilities (np.ndarray): Predicted probabilities for each class, shape (N, n_labels)

    Returns:
        float: Multiclass AUC score with macro-average (One-vs-Rest strategy)
    """
    # Convert targets to numpy array in case they are a list
    targets = np.array(targets)

    # Get the unique classes present in the target
    classes = np.unique(targets)

    # Convert targets to one-hot encoding for AUC computation
    targets_bin = label_binarize(targets, classes=classes)

    # Compute and return macro-average multiclass AUC
    return roc_auc_score(targets_bin, probabilities, multi_class='ovr', average='macro')