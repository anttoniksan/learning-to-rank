import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super(MLP, self).__init__()
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
