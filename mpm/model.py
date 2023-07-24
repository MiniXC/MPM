import torch
from torch import nn
from tqdm.auto import tqdm
from mpm.transformer import TransformerEncoder, PositionalEncoding
from mpm.conformer_layer import ConformerLayer
from mpm.pooling import AttentiveStatsPooling

class MPM(nn.Module):
    """
    Masked Prosody Model
    """

    def __init__(
        self,
        bins=128,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=False,
        nlayers=8,
        length=513,
    ):
        super().__init__()

        self.pitch_embedding = nn.Embedding(bins+1, filter_size)
        self.energy_embedding = nn.Embedding(bins+1, filter_size)
        self.vad_embedding = nn.Embedding(bins+1, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.transformer = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
            ),
            num_layers=nlayers,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.LayerNorm((length, filter_size)),
            nn.GELU(),
        )

        self.output_pitch = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.LayerNorm((length, filter_size)),
            nn.GELU(),
            nn.Linear(filter_size, bins),
        )

        self.output_energy = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.LayerNorm((length, filter_size)),
            nn.GELU(),
            nn.Linear(filter_size, bins),
        )

        self.output_vad = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.LayerNorm((length, filter_size)),
            nn.GELU(),
            nn.Linear(filter_size, bins),
        )


        self.apply(self._init_weights)

        # save hparams
        self.hparams = {
            "bins": bins,
            "filter_size": filter_size,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "depthwise": depthwise,
            "nlayers": nlayers,
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, pitch, energy, vad, return_reprs=False):
        pitch = self.pitch_embedding(pitch)
        energy = self.energy_embedding(energy)
        vad = self.vad_embedding(vad)
        x = pitch + energy + vad
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        pitch = self.output_pitch(x)
        energy = self.output_energy(x)
        vad = self.output_vad(x)
        if not return_reprs:
            return {
                "pitch": pitch,
                "energy": energy,
                "vad": vad,
            }
        else:
            return {
                "pitch": pitch,
                "energy": energy,
                "vad": vad,
                "reprs": x,
            }