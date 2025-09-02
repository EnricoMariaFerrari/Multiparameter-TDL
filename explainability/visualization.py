import torch
import matplotlib.pyplot as plt 

def plot_complex(image, filter, image_index, filter_index):

    complex = torch.where((filter < filter_index) & (image < image_index), torch.tensor(1.0), torch.tensor(-1.0))

    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(complex, cmap='bwr', vmin=-1, vmax=1)
    axs[1].set_title("Selected Complex")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    return(fig)

def get_critical_complex_plots(
    val_loader,
    filters,
    class_0,
    class_1,
    critical_index_image,
    critical_index_filter,
    n_plots=5
):
    figs_class_0 = []
    figs_class_1 = []

    count = 0
    i = 0
    for inputs, targets in val_loader:
        for j in range(len(targets)):
            if count == n_plots:
                break
            if targets[j] == class_0:
                filter_ = filters[i][j][0]
                image = inputs[j][0]
                fig = plot_complex(image, filter_, critical_index_image, critical_index_filter)
                figs_class_0.append(fig)
                count += 1
        if count == n_plots:
            break
        i += 1

    count = 0
    i = 0
    for inputs, targets in val_loader:
        for j in range(len(targets)):
            if count == n_plots:
                break
            if targets[j] == class_1:
                filter_ = filters[i][j][0]
                image = inputs[j][0]
                fig = plot_complex(image, filter_, critical_index_image, critical_index_filter)
                figs_class_1.append(fig)
                count += 1
        if count == n_plots:
            break
        i += 1

    return figs_class_0, figs_class_1