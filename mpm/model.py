import torch
from torch import nn
from tqdm.auto import tqdm
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .pooling import AttentiveStatsPooling

class MPM(nn.Module):
    """
    Masked Prosody Model
    """

    def __init__(
        self,
        in_channels=80,
        pitch_bins=256,
        energy_bins=256,
        vad_bins=256,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        nlayers=4,
        pitch_mask_len=10,
        pitch_mask_percent=0.5,
        energy_mask_len=10,
        energy_mask_percent=0.5,
        vad_mask_len=10,
        vad_mask_percent=0.5,
    ):
        super().__init__()
        in_channels = in_channels
        num_outputs = 3

        self.loss_compounds = ["pitch", "energy", "vad"]

        self.mel_in_layer = nn.Sequential(
            nn.Linear(in_channels, filter_size),
            nn.BatchNorm1d(filter_size),
            nn.GELU(),
            nn.Linear(filter_size, filter_size),
            nn.BatchNorm1d(filter_size),
            nn.GELU(),
        )

        # bins
        self.pitch_bins = torch.linspace(-3, 3, pitch_bins)
        self.energy_bins = torch.linspace(-3, 3, energy_bins)
        self.vad_bins = torch.linspace(-3, 3, vad_bins)

        self.pitch_bins = nn.Parameter(self.pitch_bins, requires_grad=False)
        self.energy_bins = nn.Parameter(self.energy_bins, requires_grad=False)
        self.vad_bins = nn.Parameter(self.vad_bins, requires_grad=False)

        self.pitch_embedding = nn.Embedding(pitch_bins, filter_size)
        self.energy_embedding = nn.Embedding(energy_bins, filter_size)
        self.vad_embedding = nn.Embedding(vad_bins, filter_size)

        self.in_layer = nn.Linear(in_channels, filter_size)

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
            nn.BatchNorm1d(filter_size),
            nn.GELU(),
            nn.Linear(filter_size, num_outputs),
        )

        self.pitch_mask_len = pitch_mask_len
        self.pitch_mask_percent = pitch_mask_percent
        self.energy_mask_len = energy_mask_len
        self.energy_mask_percent = energy_mask_percent
        self.vad_mask_len = vad_mask_len
        self.vad_mask_percent = vad_mask_percent


        self.apply(self._init_weights)

        # save hparams
        self.hparams = {
            "in_channels": in_channels,
            "pitch_bins": pitch_bins,
            "energy_bins": energy_bins,
            "vad_bins": vad_bins,
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

    def forward(self, mel, pitch, energy, vad):
        mel_padding_mask = mel.sum(dim=-1) != 0
        mel_padding_mask = mel_padding_mask.to(mel.dtype)
        pitch *= mel_padding_mask
        energy *= mel_padding_mask
        vad *= mel_padding_mask
        x = self.mel_in_layer(mel)
        pitch = torch.bucketize(pitch, self.pitch_bins)
        energy = torch.bucketize(energy, self.energy_bins)
        vad = torch.bucketize(vad, self.vad_bins)
        pitch = self.pitch_embedding(pitch)
        energy = self.energy_embedding(energy)
        vad = self.vad_embedding(vad)
        x = x + pitch + energy + vad
        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mel_padding_mask)
        results = self.output_layer(x)
        return {
            "pitch": results[:, :, 0],
            "energy": results[:, :, 1],
            "vad": results[:, :, 2],
            "padding_mask": mel_padding_mask,
        }