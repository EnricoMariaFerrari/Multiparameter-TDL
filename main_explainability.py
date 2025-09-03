import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from medmnist import PathMNIST, OCTMNIST, TissueMNIST

from models.toponet import TopoNet
from utils.dataloaders import get_data_testing
from explainability.surfaces import compute_net_output, euler_surfaces_on_grid
from explainability.visualization import get_critical_complex_plots

if __name__ == "__main__":

    # --- Argument parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PathMNIST")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--class_0", type=int, default=6)
    parser.add_argument("--class_1", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--n_plots", type=int, default=5)
    parser.add_argument("--net_path", type=str, required=True)
    parser.add_argument("--pre_attention", type=float, default=1)
    parser.add_argument("--n_cpu_test", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join("results", f"{args.data_name}_class_{args.class_0}_vs_{args.class_1}")

    # --- Dataset loading ---
    dataset_map = {
        "PathMNIST": PathMNIST,
        "OCTMNIST": OCTMNIST,
        "TissueMNIST": TissueMNIST
    }

    if args.data_name not in dataset_map:
        raise ValueError(f"'{args.data_name}' not supported.")

    DatasetClass = dataset_map[args.data_name]
    raw_val_dataset = DatasetClass(split="val", download=True, size=28)

    # --- Dataloader ---
    val_loader = get_data_testing(
        raw_dataset=raw_val_dataset,
        batch_size=args.batch_size,
        selected_classes=[args.class_0, args.class_1]
    )

    # --- Load trained model ---
    checkpoint = torch.load(args.net_path, map_location='cpu', weights_only=False)
    net = TopoNet(
        n_parameters=2,
        n_cpu_test=args.n_cpu_test,
        pre_attention=args.pre_attention
    )
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # --- Compute outputs ---
    points, weights, filters = compute_net_output(net, val_loader, args.batch_size)

    # --- Compute Euler surfaces ---
    critical_index_image, critical_index_filter, mean_surface_class_0, mean_surface_class_1, terrain = \
        euler_surfaces_on_grid(val_loader, args.class_0, args.class_1, points, weights, args.resolution)

    # --- Plot critical complexes ---
    figs_class_0, figs_class_1 = get_critical_complex_plots(
        val_loader,
        filters,
        args.class_0,
        args.class_1,
        critical_index_image,
        critical_index_filter,
        args.n_plots
    )

    # --- Save results ---
    os.makedirs(args.save_dir, exist_ok=True)

    # Save surfaces
    np.save(os.path.join(args.save_dir, "mean_surface_class_0.npy"), mean_surface_class_0)
    np.save(os.path.join(args.save_dir, "mean_surface_class_1.npy"), mean_surface_class_1)
    np.save(os.path.join(args.save_dir, "terrain.npy"), terrain)

    # Save figures
    for i, fig in enumerate(figs_class_0):
        fig.savefig(os.path.join(args.save_dir, f"class_{args.class_0}_plot_{i}.png"))
        plt.close(fig)

    for i, fig in enumerate(figs_class_1):
        fig.savefig(os.path.join(args.save_dir, f"class_{args.class_1}_plot_{i}.png"))
        plt.close(fig)
    print(f"\n✅ Surfaces and plots saved in '{args.save_dir}'")