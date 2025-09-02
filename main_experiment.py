import argparse
import torch

from medmnist import PathMNIST, OCTMNIST, TissueMNIST

from data.dataloaders import get_data
from utils.label_map import get_label_map
from training.loss import get_cost_function
from training.train import train_model, test

if __name__ == "__main__":

    # --- Argument parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="PathMNIST")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--n_train_data", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_conv", type=float, default=0.0005)
    parser.add_argument("--lr_lin", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--bandwidth", type=float, default=0.2)
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--n_parameters", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--const", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cpu_test", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pre_attention", type=float, default=1)
    parser.add_argument("--eta_min", type=float, default=0.0005)
    parser.add_argument("--n_saved_net", type=int, default=0)
    args = parser.parse_args()

    # --- Dataset loading ---
    dataset_map = {
        "PathMNIST": PathMNIST,
        "OCTMNIST": OCTMNIST,
        "TissueMNIST": TissueMNIST
    }

    if args.data_name not in dataset_map:
        raise ValueError(f"'{args.data_name}' not supported.")

    DatasetClass = dataset_map[args.data_name]
    raw_train_dataset = DatasetClass(split="train", download=True, size=28)
    raw_val_dataset   = DatasetClass(split="val", download=True, size=28)
    raw_test_dataset  = DatasetClass(split="test", download=True, size=28)

    # --- Dataloader ---
    train_loader, val_loader, test_loader = get_data(
        data_name=args.data_name,
        raw_train_dataset=raw_train_dataset,
        raw_val_dataset=raw_val_dataset,
        raw_test_dataset=raw_test_dataset,
        n_train_data=args.n_train_data,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        noise=args.noise,
        seed=args.seed
    )

    print(f"Number training data = {len(train_loader.dataset)}")
    print(f"Number val data = {len(val_loader.dataset)}")
    print(f"Number test data = {len(test_loader.dataset)}")

    # --- Train ---
    net, history_loss, history_acc, history_auc = train_model(
        data_name=args.data_name,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_conv=args.lr_conv,
        lr_lin=args.lr_lin,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        bandwidth=args.bandwidth,
        resolution=args.resolution,
        n_parameters=args.n_parameters,
        const=args.const,
        n_cpu_test=args.n_cpu_test,
        eta_min=args.eta_min,
        pre_attention=args.pre_attention,
        device=args.device
    )

    # --- Test ---
    label_map = get_label_map(args.data_name)
    cost_function_class = get_cost_function("classification")
    cost_function_con   = get_cost_function("contrastive")

    _, _, _, test_acc, test_auc = test(
        label_map, net, test_loader,
        cost_function_con=cost_function_con,
        cost_function_class=cost_function_class,
        const=args.const,
        device=args.device
    )

    # --- Results ---
    print("\n📊 Test Performance Summary")
    print("=" * 32)
    print(f"Accuracy:           {test_acc:.2f}%")
    print(f"AUC:                 {test_auc:.4f}")
    print("=" * 32)

    # --- Save training result ---
    torch.save({
        'model_state_dict': net.state_dict(),
        'history_loss': history_loss,
        'history_acc': history_acc,
        'history_auc': history_auc
    }, f"results/saved_training{args.n_saved_net}.pth")

    print(f"\n✅ Training history saved in 'saved_training{args.n_saved_net}.pth'")