# custom_modeling_qwen.py
from transformers import Qwen2ForCausalLM
import torch.nn as nn
from RBFLayer import RBFLayer  # Import your RBF implementation
import torch

class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        # Replace the feedforward MLP layers with RBF layers
        for i, layer in enumerate(self.model.layers):
            in_features = 896
            out_features = 4864
            
            # Replace gate_proj and up_proj (from 896 to 4864)
            layer.mlp.gate_proj = RBFLayer(in_features_dim=in_features, num_kernels=2, out_features_dim=out_features, radial_function=gaussian_rbf, norm_function=euclidean_norm)
            layer.mlp.up_proj = RBFLayer(in_features_dim=in_features, num_kernels=2, out_features_dim=out_features, radial_function=gaussian_rbf, norm_function=euclidean_norm)
            
            # Replace down_proj (from 4864 to 896)
            layer.mlp.down_proj = RBFLayer(in_features_dim=out_features, num_kernels=2, out_features_dim=in_features, radial_function=gaussian_rbf, norm_function=euclidean_norm)

# Define radial basis and norm functions (assuming they're part of your custom RBF implementation)
def gaussian_rbf(x):
    return torch.exp(-x**2)

def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)
