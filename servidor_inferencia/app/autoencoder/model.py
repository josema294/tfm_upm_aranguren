from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DAutoencoder(nn.Module):
    """1D convolutional autoencoder for tri-axial vibration windows.

    Expected input shape: [batch, channels=3, window_size=100].
    """

    def __init__(self, in_channels: int = 3, filters: list[int] | None = None, kernel_size: int = 5):
        super().__init__()
        filters = filters or [16, 32, 64]

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, filters[0], kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(filters[0], filters[1], kernel_size=kernel_size, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(filters[1], filters[2], kernel_size=kernel_size, stride=2, padding=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(filters[2], filters[1], kernel_size=kernel_size, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(filters[1], filters[0], kernel_size=kernel_size, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
        )

        self.final_conv = nn.Sequential(
            nn.ConvTranspose1d(filters[0], in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = self.final_conv(decoded)
        if decoded.size(2) != x.size(2):
            decoded = F.interpolate(decoded, size=x.size(2))
        return decoded
