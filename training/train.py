import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.metrics import compute_multiclass_auc
from utils.label_map import get_label_map, get_n_labels
from training.loss import get_cost_function
from training.optimizer import get_optimizer
from models.toponet import TopoNet

def train(label_map, net, data_loader, optimizer, cost_function_con, cost_function_class, const=0, device="cuda"):
    
    n_batches = 0
    cumulative_loss = 0.0
    samples = 0.0
    cumulative_loss_con = 0.0
    cumulative_loss_class = 0.0
    cumulative_accuracy = 0.0
    all_targets = []
    all_outputs = []

    # Set the network to training mode
    net.train()

    # Iterate over the training set
    for _, (inputs, targets) in enumerate(tqdm(data_loader, desc="Training")):

        # Load data into GPU
        inputs = inputs.squeeze(1).to(device)
        targets = targets.to(device)
        targets = torch.tensor([label_map[int(t.item())] for t in targets]).to(torch.long).to(device)

        # Forward pass
        outputs, vect_output = net(inputs)
        loss_con = cost_function_con(vect_output, targets)
        loss_class = cost_function_class(outputs,targets)
        loss = loss_class + const * loss_con

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        _, predicted = outputs.max(dim=1)
        cumulative_accuracy += predicted.eq(targets).sum().item()
        cumulative_loss_class += loss_class.item()
        cumulative_loss_con += loss_con.item()
        cumulative_loss += loss.item()
        all_targets.extend(targets.detach().cpu().numpy())
        all_outputs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

        n_batches +=1
        samples += inputs.shape[0]

    # Compute macro-AUC
    auc_multi = compute_multiclass_auc(all_targets, np.array(all_outputs))

    return (
        cumulative_loss / n_batches,
        cumulative_loss_class / n_batches,
        cumulative_loss_con / n_batches,
        cumulative_accuracy / samples * 100,
        auc_multi
    )

def test(label_map, net, data_loader, cost_function_con, cost_function_class, const=0, device="cuda"):
    n_batches = 0
    cumulative_loss = 0.0
    cumulative_loss_con = 0.0
    cumulative_loss_class = 0.0
    cumulative_accuracy = 0.0
    samples = 0
    all_targets = []
    all_outputs = []

    # Set the network to evaluation mode
    net.eval()
    # Disable gradient computation
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(data_loader, desc="Testing")):

            # Load data into GPU
            inputs = inputs.squeeze(1).to(device)
            targets = targets.to(device)
            targets = torch.tensor([label_map[int(t.item())] for t in targets]).to(torch.long).to(device)

            # Forward pass
            outputs, vect_output = net(inputs)
            loss_con = cost_function_con(vect_output, targets)
            loss_class = cost_function_class(outputs, targets)
            loss = loss_class + const * loss_con

            _, predicted = outputs.max(dim=1)
            cumulative_accuracy += predicted.eq(targets).sum().item()
            cumulative_loss_class += loss_class.item()
            cumulative_loss_con += loss_con.item()
            cumulative_loss += loss.item()
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            n_batches += 1
            samples += inputs.shape[0]

    # Calcolo AUC multiclasse (macro)
    auc_multi = compute_multiclass_auc(all_targets, np.array(all_outputs))

    return (
        cumulative_loss / n_batches,
        cumulative_loss_class / n_batches,
        cumulative_loss_con / n_batches,
        cumulative_accuracy / samples * 100,
        auc_multi
    )

def train_model(data_name,
         train_loader,
         val_loader,
         device="cpu",
         lr_conv=0.01,
         lr_lin=0.01,
         const=0,
         weight_decay=0.0005,
         epochs=10,
         bandwidth=0.2,
         resolution=5,
         n_parameters=2,
         n_cpu_test=1,
         pre_attention=1,
         eta_min=0.000001
         ):

    label_map = get_label_map(data_name)
    n_labels = get_n_labels(data_name)

    # Instantiate model and send it to device
    net = TopoNet(
        n_parameters=n_parameters,
        resolution=resolution,
        bandwidth=bandwidth,
        n_cpu_test=n_cpu_test,
        n_labels=n_labels,
        pre_attention=pre_attention
    ).to(device)
    
    # Loss functions
    cost_function_class = get_cost_function("classification")
    cost_function_con = get_cost_function("contrastive")

    # Optimizer & scheduler
    optimizer = get_optimizer(net, lr_conv=lr_conv, lr_lin=lr_lin, wd=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    history_loss = {'train_loss': [], 'val_loss': []}
    history_acc  = {'train_acc': [],  'val_acc': []}
    history_auc  = {'train_auc': [],  'val_auc': []}

    # Track best model
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    # Training loop
    for e in range(epochs):

        print(f"---------------------- Epoch {e+1} ----------------------")
        train_loss, train_loss_class, train_loss_con, train_acc, train_auc = train(
            label_map, net, train_loader, optimizer,
            cost_function_con=cost_function_con,
            cost_function_class=cost_function_class,
            const=const, device=device
        )
        print(f"train/acc: {train_acc:.2f}%")
        print(f"train/auc:, {train_auc:.4f}")
        print(f"train/loss:, {train_loss:.4f}")
        print(f"train/loss_class:, {train_loss_class:.4f}")
        print(f"train/loss_dist:, {train_loss_con:.4f}")

        print("---------------------------------------------------------")
        val_loss, val_loss_class, val_loss_con, val_acc, val_auc = test(
            label_map, net, val_loader,
            cost_function_con=cost_function_con,
            cost_function_class=cost_function_class,
            const=const, device=device
        )
        print(f"val/acc: {val_acc:.2f}%")
        print(f"val/auc:, {val_auc:.4f}")
        print(f"val/loss:, {val_loss:.4f}")
        print(f"val/loss_class:, {val_loss_class:.4f}")
        print(f"val/loss_dist:, {val_loss_con:.4f}")

        # Save history
        history_acc['train_acc'].append(train_acc)
        history_acc['val_acc'].append(val_acc)
        history_auc['train_auc'].append(train_auc)
        history_auc['val_auc'].append(val_auc)
        history_loss['train_loss'].append(train_loss)
        history_loss['val_loss'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = e
            best_state = net.state_dict()

        scheduler.step()

    # Restore best model
    if best_state is not None:
        net.load_state_dict(best_state)

    print(f"best epoch is {best_epoch+1}")

    return net, history_loss, history_acc, history_auc