import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from RBFLayer import RBFLayer

def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

class CustomRBFFeedForward(nn.Module):
    def __init__(self, in_features, out_features, num_kernels):
        super(CustomRBFFeedForward, self).__init__()
        # RBFLayer from the given implementation
        self.rbf_layer = RBFLayer(
            in_features_dim=in_features,  # Input size (e.g., 896)
            num_kernels=num_kernels,  # Number of kernels in the RBF layer (can be tuned)
            out_features_dim=out_features,  # Output size (e.g., 4864)
            radial_function=rbf_gaussian,  # Use the Gaussian RBF
            norm_function=l_norm  # Use Euclidean norm
        )

    def forward(self, x):
        # Apply the RBF layer to the input x
        return self.rbf_layer(x)
