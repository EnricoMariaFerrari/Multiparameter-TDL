import torch

def get_optimizer(net, lr_conv, lr_lin, wd):

    conv_params = []
    lin_params = []

    for name, param in net.named_parameters():
        if 'convolutional' in name.lower():
            conv_params.append(param)
        else:
            lin_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': conv_params, 'lr': lr_conv},
        {'params': lin_params, 'lr': lr_lin}
        ], weight_decay=wd)

    return optimizer