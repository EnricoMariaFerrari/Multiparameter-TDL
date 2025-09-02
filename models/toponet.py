import torch
import torch.nn as nn
import torch.nn.functional as F

from multipers.filtrations.density import batch_signed_measure_convolutions
import multipers as mp

from joblib import Parallel, delayed

from utils.formatter import signed_measures_formatter

class TopoLayer(torch.nn.Module):
    """
    TopoLayer: Computes Euler profiles from n-parameter cubical filtrations for a batch
    of N n-dimensional images (each n-image is a n-dimensional filtering function).

    This layer computes the Euler profiles as signed measures, returned at the end.

    Args:
        n_cpu_test (int): Number of CPU cores to use during evaluation (not training) 
                          for parallel computation of topological signatures.

    Returns:
        A batch of formatted signed measures, one per input, representing topological
        descriptors derived from the multi-parameter cubical filtrations.
    """

    def __init__(
        self,
        n_cpu_test: int = 1,
    ):
        super().__init__()
        self.n_cpu_test = n_cpu_test

    def _signed_measure(
        self,
        input, # n-dimensional image
        filtration_grid # filtration grid of the input's filtration
    ):
        # Compute the n-parameter cubical filtration (detach input from autograd)
        # Note: Differentiability is preserved at the invariant level via the 'grid' argument
        multi_filtration = mp.filtrations.Cubical(input.detach())

        # Compute the signed measure of the Euler profile using the given grid
        # The grid ensures differentiability of the output even though 'input' is detached
        sm = mp.signed_measure(multi_filtration, grid=filtration_grid, invariant="euler")

        return sm

    def _batch_signed_measure(
        self, 
        inputs,
        grids
    ):
        if self.training:
            # Sequential computation during training
            sms = (self._signed_measure(input, grid)
                   for input, grid in zip(inputs, grids))
        else:
            # Parallel computation during evaluation
            sms = Parallel(n_jobs=self.n_cpu_test)(
                delayed(self._signed_measure)(input, grid)
                for input, grid in zip(inputs, grids)
            )

        return tuple(sms)

    def forward(
        self,
        inputs  # tensor (N, h, w, n_parameters): batch of N n-images, each seen as n-filtering function
    ):
        # Compute an n-filtration grid for each input
        filtration_grids = tuple(
            mp.grids.compute_grid([input[..., i].ravel() for i in range(input.shape[-1])])
            for input in inputs
        )

        # Compute Euler profiles as signed measures for the batch
        sms = self._batch_signed_measure(inputs, filtration_grids)

        # Change sms format for vect_layer use
        sms = signed_measures_formatter(sms)

        return sms
    
class VectLayer(torch.nn.Module):
    """
    VectLayer: Converts a batch of signed measures into vector representations
    by convolving them with a Gaussian kernel over a fixed grid of evaluation points.

    Args:
        n_parameters (int): Dimensionality of the points of the measure.
        dtype (torch.dtype): Data type for tensors (default: torch.float64).
        resolution (int): Number of evaluation points per dimension in the grid.
        bandwidth (float): Bandwidth of the Gaussian kernel used for convolution.

    Returns:
        torch.Tensor: A batch of vectorized representations, obtained by evaluating
                      the convolved measures on the points of the fixed grid.
    """
    def __init__(
        self,
        n_parameters: int = 2,
        dtype = torch.float64,
        resolution: int = 5,
        bandwidth = 0.2,
    ):
        super().__init__()
        self.dtype = dtype
        self.n_parameters = n_parameters
        self.resolution = resolution
        self.bandwidth = bandwidth

        self.pts_to_evaluate = torch.cartesian_prod(
            *(torch.linspace(-1, 1, self.resolution) for _ in range(self.n_parameters))
        ).type(dtype)

        if self.n_parameters == 1:
            self.pts_to_evaluate = self.pts_to_evaluate.unsqueeze(-1)

    def forward(self, sms):  # sms is a batch of signed measures
        conv_sms = batch_signed_measure_convolutions(
            sms,
            self.pts_to_evaluate,
            bandwidth=self.bandwidth,
            kernel="gaussian",
        )
        return conv_sms

# NETWORK

class TopoNet(nn.Module):
    
    def __init__(self, n_parameters=2, resolution=5, bandwidth=0.3, n_cpu_test=1, pre_attention = 0, n_labels=3):
        super().__init__()

        self.dtype = torch.float64
        self.resolution = resolution
        self.n_parameters = n_parameters
        self.bandwidth = bandwidth
        self.n_cpu_test = n_cpu_test
        self.pre_attention = pre_attention
        self.n_labels = n_labels

        # Convolutions
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=24, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=24, out_channels=1, kernel_size=3, padding=1),
        ).to(self.dtype)

        # Topological and Vectorization layers
        self.topological_layer = TopoLayer(n_cpu_test=n_cpu_test)
        self.vect_layer = VectLayer(resolution=self.resolution, dtype=self.dtype, n_parameters=self.n_parameters)

        # MLP
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.resolution ** self.n_parameters, 128).to(self.dtype),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64).to(self.dtype),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.n_labels).to(self.dtype)
        )

    def forward(self, x):
        device = next(self.parameters()).device # same device of network
        x = x.to(device=device, dtype=self.dtype)

        x_orig = x.unsqueeze(-1)
        x_conv = x.unsqueeze(1)  # (B, 1, H, W)

        x_conv = self.convolutional(x_conv)
        x_conv = torch.tanh(x_conv + self.pre_attention*x.unsqueeze(1))

        x_conv = x_conv.squeeze(1).unsqueeze(-1)  # (B, H, W, 1)

        if self.n_parameters == 2:
            x_cat = torch.cat([x_conv, x_orig], dim=-1)
        else:
            x_cat = x_conv

        x_cat = x_cat.cpu()

        x = self.topological_layer(x_cat)
        x = self.vect_layer(x)

        x = x.to(device=device)
        x = F.normalize(x, p=2, dim=1)

        # For classification
        y = self.mlp(x)

        return y, x # return logits and topological vectors