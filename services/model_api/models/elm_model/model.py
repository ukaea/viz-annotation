import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet1D, self).__init__()

        # Contracting Path (Encoder)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expanding Path (Decoder)
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.upconv0 = self.upconv_block(64, 32)

        # Final 1x1 convolution for output
        self.final_conv = nn.Conv1d(32, out_channels, kernel_size=1)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.9))

    def conv_block(self, in_channels, out_channels):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ]

        block = nn.Sequential(*layers)
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        return block

    def forward(self, x, label=None):
        # Encoder Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder Path (Up-sampling + concatenation)
        up4 = self.upconv4(bottleneck)
        up3 = self.upconv3(up4 + enc4)  # skip connection
        up2 = self.upconv2(up3 + enc3)  # skip connection
        up1 = self.upconv1(up2 + enc2)  # skip connection
        up0 = self.upconv0(up1)

        # Final output
        out = self.final_conv(up0)
        probs = F.sigmoid(out)

        loss_dict = {}
        if label is not None:
            loss_dict["bce_loss"] = self.bce_loss(out, label)

        return loss_dict, probs


class SingleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        drop=0,
        conv_layer=nn.Conv1d,
        norm_layer=nn.Identity,
    ):
        super().__init__()
        self.single_conv = nn.Sequential(
            conv_layer(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        conv_layer=nn.Conv1d,
        activation=nn.ReLU,
        norm_layer=nn.Identity,
    ):
        super().__init__()

        self.double_conv = nn.Sequential(
            conv_layer(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            norm_layer(out_channels),
            activation(),
            conv_layer(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            norm_layer(out_channels),
            activation(),
            conv_layer(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            norm_layer(out_channels),
            activation(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Network(nn.Module):
    def __init__(
        self, input_dim=1, hidden_dim=256, code_dim=64, output_dim=2, **kwargs
    ):
        super().__init__()

        self.in_channels = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            SingleConv(
                input_dim,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm1d,
            ),
            nn.MaxPool1d(2),
            DoubleConv(
                16, 32, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm1d
            ),
            nn.MaxPool1d(2),
            SingleConv(
                32,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm1d,
            ),
            nn.AdaptiveAvgPool1d(1),
        )

        # self.pool1d = nn.AdaptiveAvgPool1d(1)

        self.linear_head = nn.Linear(hidden_dim, output_dim)

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.9]))

    def instance_norm(self, x, epsilon=1e-5):
        # Compute mean and variance along the L dimension (axis=2)
        mean = x.mean(dim=2, keepdim=True)  # Shape: (N, D, 1)
        var = x.var(dim=2, keepdim=True, unbiased=False)  # Shape: (N, D, 1)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + epsilon)  # Shape: (N, D, L)
        return x_normalized

    def forward(self, x, label=None):
        x = self.instance_norm(x)
        x = self.mlp(x).squeeze(-1)
        x = self.linear_head(x)
        x = x.unsqueeze(1)
        probs = F.sigmoid(x)

        loss_dict = {}
        if label is not None:
            loss_dict["bce_loss"] = self.bce_loss(x, label)

        return loss_dict, probs
