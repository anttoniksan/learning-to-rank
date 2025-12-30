from typing import Literal

import torch
import torch.nn as nn

from src.models.common import MLP


class CrossNet(nn.Module):
    def __init__(self, input_dim: int, n_layers: int):
        super(CrossNet, self).__init__()
        self.n_layers = n_layers

        self.w = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=False) for _ in range(n_layers)]
        )

        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros((input_dim,))) for _ in range(n_layers)]
        )

    def forward(self, x):
        xl = x

        for layer, bias in zip(self.w, self.b):
            xl = x * layer(xl) + bias + xl

        return xl


class DCNV2(nn.Module):
    def __init__(
        self,
        n_features: int,
        field_dims: list[int],
        embedding_dim: int,
        mlp_dims: list[int] = [128, 64],
        structure: Literal["stacked", "parallel"] = "parallel",
        output_dim: int = 1,
        dropout: float = 0.1,
        n_layers: int = 2,
    ):
        super(DCNV2, self).__init__()

        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.embedding_out_dim = n_features * embedding_dim
        self.structure = structure

        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        mlp_output_dims = [self.embedding_out_dim] + mlp_dims
        if structure == "stacked":
            mlp_output_dims.append(output_dim)

        self.mlp = MLP(
            input_dim=embedding_dim * len(field_dims),
            output_dims=mlp_output_dims,
            dropout=dropout,
            bias=False,
        )
        self.act = nn.ReLU()
        self.projection = nn.Linear(
            self.embedding_out_dim + mlp_output_dims[-1], output_dim
        )

        self.cross_net = CrossNet(input_dim=self.embedding_out_dim, n_layers=n_layers)
        self.output = nn.Sigmoid()

    def forward(self, x):
        embed_x = self.embedding(x)
        cross_net_out = self.cross_net(embed_x.view(-1, self.embedding_out_dim))
        if self.structure == "parallel":
            mlp_out = self.mlp(embed_x.view(-1, self.embedding_out_dim))
            out = self.projection(torch.cat([mlp_out, cross_net_out], dim=-1))
        else:
            mlp_out = self.mlp(self.act(cross_net_out))
            out = mlp_out
        return self.output(out)
