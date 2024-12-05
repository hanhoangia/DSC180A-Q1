from message_passing import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
import scatter
import torch.nn as nn

class HyperConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HyperConvLayer, self).__init__()
        self.mlp = Seq(
            Linear(in_channels, out_channels),
            ReLU()
        )

    def forward(self, x, edge_index):
        # Aggregate information from neighbors
        row, col = edge_index
        x_aggregated = scatter.scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        return self.mlp(x_aggregated)

class DEHNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, len_instances):
        super(DEHNN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HyperConvLayer(in_channels, hidden_channels))
            in_channels = hidden_channels
        self.output_layer = Linear(hidden_channels, out_channels)
        self.len_instances = len_instances

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_layer(x[:self.len_instances])  # Predict only for nodes