import torch
import torch.nn as nn

from src.models.common import MLP


class FM(nn.Module):
    def __init__(self, output_dim: int):
        super(FM, self).__init__()

        self.output_dim = output_dim

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        interaction = square_of_sum - sum_of_square
        return 0.5 * interaction.sum(1, keepdim=True)


class DeepFM(nn.Module):
    def __init__(
        self,
        n_features: int,
        field_dims: list[int],
        embedding_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super(DeepFM, self).__init__()

        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.embedding_out_dim = n_features * embedding_dim

        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        self.offsets = torch.tensor(
            (0, *torch.cumsum(torch.tensor(field_dims), 0)[:-1])
        )

        self.fm = FM(output_dim=output_dim)

        self.linear = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)), requires_grad=True)

        self.mlp = MLP(
            input_dim=self.embedding_out_dim,
            # Last connection condenses to 1 output node
            output_dims=[32, 32, output_dim],
            dropout=dropout,
        )
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = x + self.offsets.to(x.device)
        linears = torch.sum(self.linear(x), dim=1) + self.bias

        embed_x = self.embedding(x)
        fm_x = self.fm(embed_x)
        mlp_x = self.mlp(embed_x.view(-1, self.embedding_out_dim))
        x = linears + fm_x + mlp_x

        return self.output(x.squeeze(1))
