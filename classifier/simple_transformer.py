from torch import nn
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)

class ProminenceBreakTransformer(nn.Module):
    def __init__(
        self,
        in_channels=256,
        filter_size=256,
        kernel_size=1,
        dropout=0.1,
        nlayers=2,
        num_outputs=2,
    ):
        super().__init__()
        in_channels = in_channels
        num_outputs = 2

        self.in_layer = nn.Sequential(
            nn.Linear(in_channels, filter_size),
            nn.BatchNorm1d(filter_size),
            nn.GELU(),
            nn.Linear(filter_size, filter_size),
            nn.BatchNorm1d(filter_size),
            nn.GELU(),
        )

        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                filter_size,
                2,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=nlayers,
            norm=nn.LayerNorm(filter_size),
        )

        self.output_layer = nn.Linear(filter_size, num_outputs)

    def forward(self, x):
        x = self.in_layer(x)
        # x = self.positional_encoding(x)
        # x = self.transformer(x)
        x = self.output_layer(x)
        return x