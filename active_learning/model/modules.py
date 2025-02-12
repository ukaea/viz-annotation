"""
@authors: 
    Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk)
    Samuel Jackson UKAEA (samuel.jackson@ukaea.uk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def print_verbose(text, verbose=0):
    if verbose>0:
        print(text)
        
def MLP(channels: List[int], norm_layer=nn.Identity) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            layers.append(norm_layer(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
    

class ClassHead(nn.Module):
    """
    Simple MLP network with dropout.
    """
    def __init__(self, in_channels=512, out_channels=2, act_layer=nn.ReLU, dropout=0.0, bias=True):
        super().__init__()

        self.act = act_layer()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_channels, in_channels//2, bias=bias)
        self.fc2 = nn.Linear(in_channels//2, in_channels//4, bias=bias)
        self.fc3 = nn.Linear(in_channels//4, out_channels, bias=bias)

    def forward(self, x):

        x = self.act(self.fc1(self.drop(x)))
        x = self.act(self.fc2(self.drop(x)))
        x = self.fc3(x)
        
        return x

class DetectionHead(nn.Module):
    """
    Simple MLP network with dropout.
    """
    def __init__(self, in_channels=512, out_channels=2, act_layer=nn.ReLU, dropout=0.0, bias=True):
        super().__init__()

        self.act = act_layer()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Conv1d(in_channels, in_channels//2, kernel_size=1, bias=bias)
        self.fc2 = nn.Conv1d(in_channels//2, in_channels//4, kernel_size=1, bias=bias)
        self.fc3 = nn.Conv1d(in_channels//4, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):

        x = self.act(self.fc1(self.drop(x)))
        x = self.act(self.fc2(self.drop(x)))
        x = self.fc3(x)
        
        return x
        
class ClusterHead(nn.Module):
    """
    Cluster head for unsupervised classification.
    """
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, in_channels))

    def forward(self, x, alpha=None, log_probs=False):
        
        normed_features = F.normalize(x, dim=1)
        normed_clusters = F.normalize(self.clusters, dim=1)
        
        inner_products = torch.einsum("bcl,nc->bn", normed_features, normed_clusters)

        if alpha is None:
            # cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.n_classes).permute(0, 2, 1).to(torch.float32)
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.n_classes).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        # cluster_loss = -torch.log(torch.exp(cluster_probs * inner_products)).sum(1).mean()
        
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
    
    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, n_clusters={self.n_classes})"


class UNet1D(nn.Module):
    n_features = 64
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

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
        
        self.final = self.upconv_block(64, self.n_features)

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

    def forward(self, x, verbose=0):
        
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
        out = self.final(up1)

        return out
        

class SingleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        drop=0,
        activation=nn.ReLU,
        conv_layer=nn.Conv1d,
        norm_layer=nn.Identity,
    ):
        super().__init__()
        self.single_conv = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels),
            activation(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """
        (convolution => [BN] => ReLU) * 2
    """
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
        drop=0,
    ):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.double_conv = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_channels),
            activation(),
            conv_layer(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            norm_layer(out_channels),
            activation(),
            conv_layer(out_channels,out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            norm_layer(out_channels),
            activation(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.double_conv(self.drop(x))

class CNN1D(nn.Module):
    """
        Simple CNN to extract features from 1D signal.
    """
    FEATURES = [32, 64, 128, 256, 512]
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, self.FEATURES[0], kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(self.FEATURES[0], self.FEATURES[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(self.FEATURES[1], self.FEATURES[2], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(self.FEATURES[2], out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(self.FEATURES[2], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv5(x)) + F.relu(self.conv4(x))

        return x
        
class PointCNN1D(nn.Module):
    """
        Simple PointCNN like network to extract point features from 1D signal.
    """
    FEATURES = [64, 128, 256, 512, 1024]
    def __init__(self, in_channels=1, out_channels=512, norm_layer=nn.Identity):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layer1 = SingleConv(in_channels, self.FEATURES[0], kernel_size=5, padding=2, drop=0.1, norm_layer=norm_layer)
        self.layer2 = SingleConv(self.FEATURES[0], self.FEATURES[1], kernel_size=3, drop=0.1, norm_layer=norm_layer)
        self.layer3 = DoubleConv(self.FEATURES[1], self.FEATURES[2], kernel_size=3, drop=0.1, norm_layer=norm_layer)
        self.layer4 = DoubleConv(self.FEATURES[2], out_channels, kernel_size=3, drop=0.2, norm_layer=norm_layer)
        self.layer4_skip = SingleConv(self.FEATURES[2], out_channels, kernel_size=3, norm_layer=norm_layer)
        self.layer5 = DoubleConv(self.FEATURES[3], out_channels, kernel_size=3, drop=0.2, norm_layer=norm_layer)
        self.layer5_skip = SingleConv(self.FEATURES[3], out_channels, kernel_size=3, norm_layer=norm_layer)
        
    def forward(self, x, verbose=0):
        
        x = self.layer1(x)
        print_verbose(f"layer1: {x.shape}", verbose)
            
        x = self.layer2(x)
        print_verbose(f"layer2: {x.shape}", verbose)
        
        x = self.layer3(x)
        print_verbose(f"layer3: {x.shape}", verbose)
            
        x = self.layer4(x) + self.layer4_skip(x)
        print_verbose(f"layer4: {x.shape}", verbose)

        # x = self.layer5(x) + self.layer5_skip(x)
        
        return x 