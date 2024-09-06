import torch
import torch.nn as nn
from typing import Callable


class RBFLayer(nn.Module):
    def __init__(self,
                 in_features_dim: int,
                 num_kernels: int,
                 out_features_dim: int,
                 radial_function: Callable[[torch.Tensor], torch.Tensor],
                 norm_function: Callable[[torch.Tensor], torch.Tensor],
                 normalization: bool = True,
                 initial_shape_parameter: torch.Tensor = None,
                 initial_centers_parameter: torch.Tensor = None,
                 initial_weights_parameters: torch.Tensor = None,
                 constant_shape_parameter: bool = False,
                 constant_centers_parameter: bool = False,
                 constant_weights_parameters: bool = False):
        super(RBFLayer, self).__init__()

        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization

        self.initial_shape_parameter = initial_shape_parameter
        self.constant_shape_parameter = constant_shape_parameter

        self.initial_centers_parameter = initial_centers_parameter
        self.constant_centers_parameter = constant_centers_parameter

        self.initial_weights_parameters = initial_weights_parameters
        self.constant_weights_parameters = constant_weights_parameters

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialize linear combination weights
        if self.constant_weights_parameters:
            self.weights = nn.Parameter(self.initial_weights_parameters, requires_grad=False)
        else:
            self.weights = nn.Parameter(torch.zeros(self.out_features_dim, self.num_kernels, dtype=torch.float32))

        # Initialize kernels' centers
        if self.constant_centers_parameter:
            self.kernels_centers = nn.Parameter(self.initial_centers_parameter, requires_grad=False)
        else:
            self.kernels_centers = nn.Parameter(torch.zeros(self.num_kernels, self.in_features_dim, dtype=torch.float32))

        # Initialize shape parameter
        if self.constant_shape_parameter:
            self.log_shapes = nn.Parameter(self.initial_shape_parameter, requires_grad=False)
        else:
            self.log_shapes = nn.Parameter(torch.zeros(self.num_kernels, dtype=torch.float32))

        self.reset()

    def reset(self, upper_bound_kernels: float = 1.0, std_shapes: float = 0.1, gain_weights: float = 1.0) -> None:
        if self.initial_centers_parameter is None:
            nn.init.uniform_(self.kernels_centers, a=-upper_bound_kernels, b=upper_bound_kernels)

        if self.initial_shape_parameter is None:
            nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)

        if self.initial_weights_parameters is None:
            nn.init.xavier_uniform_(self.weights, gain=gain_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the RBF layer given an input tensor.
        Input has size [batch_size, sequence_length, in_features].
        """

        batch_size = input.size(0)
        sequence_length = input.size(1)

        # Expand centers to match the batch and sequence length
        c = self.kernels_centers.expand(batch_size, sequence_length, self.num_kernels, self.in_features_dim)

        # Compute differences between input and centers
        diff = input.unsqueeze(2) - c  # Shape: [batch_size, sequence_length, num_kernels, in_features_dim]

        # Apply norm function to get distances
        r = self.norm_function(diff)  # Shape: [batch_size, sequence_length, num_kernels]

        # Apply shape parameters (log_shapes) to the distances
        eps_r = self.log_shapes.exp().unsqueeze(0).unsqueeze(0) * r

        # Apply radial basis function (e.g., Gaussian)
        rbfs = self.radial_function(eps_r)

        if self.normalization:
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1, keepdim=True))

        # Combine RBF outputs using the weights
        out = (self.weights.unsqueeze(0).unsqueeze(0) * rbfs.unsqueeze(2)).sum(dim=-1)

        return out
