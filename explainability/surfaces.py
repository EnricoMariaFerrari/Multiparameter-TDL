import numpy as np
import matplotlib.pyplot as plt 
import torch          
from tqdm import tqdm
import multipers as mp

def compute_net_output(net, data_loader, batch_size):
    signed_measures = [] # list of all the signed measures
    filters = [] # list of all the learned filters
    n_inputs = 0

    print("signed measures computation")

    for inputs, targets in tqdm(data_loader):
        with torch.no_grad():

            x = inputs.squeeze(dim=1).to(net.dtype)

            x_orig = x.unsqueeze(-1)
            x_conv = x.unsqueeze(1)

            x_conv = net.convolutional(x_conv)
            x_conv = torch.tanh(x_conv + net.pre_attention*x.unsqueeze(1))

            filters.append(x_conv)

            x_conv = x_conv.squeeze(1).unsqueeze(-1)  # (B, H, W, 1)

            x_cat = torch.cat([x_conv, x_orig], dim=-1)
            x_cat_cpu = x_cat.cpu()
            x = net.topological_layer(x_cat_cpu)
        
            signed_measures.append(x)

            n_inputs += len(targets)

    points = [None]*n_inputs # list where for each data there're the points of its signed measure
    weights = [None]*n_inputs # list where for each data there're the weights of its signed measure
    
    # change measures format
    i = 0
    for j in range(len(data_loader)):
        for i in range(batch_size):
        
            points[j*batch_size + i] = signed_measures[j][i][:, :2]
            points[j*batch_size + i] = np.asarray(points[j*batch_size + i], dtype=np.float64)

            weights[j*batch_size + i] = signed_measures[j][i][:, 2]
            weights[j*batch_size + i] = np.asarray(weights[j*batch_size + i], dtype=np.int64)

    return(points, weights, filters)

# function to compute the index in [-1,1]x[-1,1], relative to the index of the associated grid
def index_conversion(image_index, filter_index, resolution):

    max_filter = 1
    min_filter = -1

    filter_index = (max_filter-min_filter)*(filter_index/resolution)+min_filter
    image_index = 2*(image_index/resolution)-1

    return(image_index, filter_index)

def euler_surfaces_on_grid(data_loader, class_0, class_1, points, weights, resolution):

    all_targets = [] # list of all the labels
    for _, targets in data_loader:
        all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)

    surfaces_class_0 = [] # list of Euler surfaces of class 0, evaluated on the same grid
    surfaces_class_1 = [] # list of Euler surfaces of class 1, evaluated on the same grid
    grid = [np.linspace(-1,1,resolution), np.linspace(-1,1,resolution)] # grid where to evaluate the Euler surfaces

    for i in range(len(all_targets)):
        if all_targets[i] == class_0:
        
            surfaces_class_0.append(mp.point_measure.integrate_measure(pts = points[i], weights=weights[i], filtration_grid=grid))

        if all_targets[i] == class_1:

            surfaces_class_1.append(mp.point_measure.integrate_measure(pts = points[i], weights=weights[i], filtration_grid=grid))

    # compute mean surfaces for each class, and their Euler terrain

    surfaces_class_0 = np.stack(surfaces_class_0, axis=0)
    mean_surface_class_0 = np.mean(surfaces_class_0, axis=0)

    surfaces_class_1 = np.stack(surfaces_class_1, axis=0)
    mean_surface_class_1 = np.mean(surfaces_class_1, axis=0)

    terrain = np.abs(mean_surface_class_0 - mean_surface_class_1)

    # Ticks and extent
    ticks = [-1, -0.5, 0, 0.5, 1]
    extent = [-1, 1, -1, 1]

    fig0, ax0 = plt.subplots()
    im0 = ax0.imshow(mean_surface_class_0, origin='lower', extent=extent)
    fig0.colorbar(im0, ax=ax0)
    ax0.set_title(f"Mean Euler Surface of class {class_0}")
    ax0.set_xticks(ticks)
    ax0.set_yticks(ticks)
    ax0.set_xlabel("Pixel intensity")
    ax0.set_ylabel("Filter value")

    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(mean_surface_class_1, origin='lower', extent=extent)
    fig1.colorbar(im1, ax=ax1)
    ax1.set_title(f"Mean Euler Surface of class {class_1}")
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xlabel("Pixel intensity")
    ax1.set_ylabel("Filter value")

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(terrain, origin='lower', extent=extent)
    fig2.colorbar(im2, ax=ax2)
    ax2.set_title("Euler Terrain Magnitude")
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xlabel("Pixel intensity")
    ax2.set_ylabel("Filter value")

    critical_index_filter, critical_index_image = np.unravel_index(np.argmax(terrain), (resolution, resolution))
    critical_index_image, critical_index_filter = index_conversion(critical_index_image, critical_index_filter, resolution)
    print(f"Argmax index at: ({critical_index_image}, {critical_index_filter})")

    return critical_index_image, critical_index_filter, fig0, fig1, fig2