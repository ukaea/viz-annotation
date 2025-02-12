"""
@authors: 
    Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk)
    Samuel Jackson UKAEA (samuel.jackson@ukaea.uk)
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *

class Network(nn.Module):
    
    def __init__(self, cfg, device='cuda', **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.device = device
        
        n_features = cfg.net.hidden_dim
        if cfg.net.type.lower()=='mlp':
            self.net = MLP([cfg.net.input_dim, cfg.net.hidden_dim*2, cfg.net.hidden_dim], norm_layer=nn.LayerNorm)
            
        elif cfg.net.type.lower()=='cnn':
            self.net = CNN1D(cfg.net.input_dim, cfg.net.hidden_dim)
        
        elif cfg.net.type.lower()=='pointcnn':
            self.net = PointCNN1D(cfg.net.input_dim, cfg.net.hidden_dim)
            
        elif cfg.net.type.lower()=='unet':
            self.net = UNet1D(cfg.net.input_dim, cfg.net.hidden_dim)
            n_features = self.net.n_features
            
        elif cfg.net.type.lower()=='chronos':
            from chronos import ChronosPipeline
            self.net = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-base",
                device_map=device,
                torch_dtype=torch.float32,
            )
            self.net.tokenizer.config.context_length = cfg.data.context_len
            self.net.tokenizer.config.use_eos_token = False
            n_features = 768
        else:
            raise RuntimeError(f"Network type {cfg.net.type} is not implemented")

        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        self.class_head = ClassHead(n_features, cfg.data.n_classes, dropout=0.1)

        self.detection_head = DetectionHead(n_features, 2, dropout=0.1)

    def instance_norm(self, x, epsilon=1e-5):
        # Compute mean and variance along the L dimension (axis=2)
        mean = x.mean(dim=2, keepdim=True)  # Shape: (N, D, 1)
        var = x.var(dim=2, keepdim=True, unbiased=False)  # Shape: (N, D, 1)
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + epsilon)  # Shape: (N, D, L)
        return x_normalized
    
    def forward(self, x, **kwargs):
        # x = self.instance_norm(x)
        
        if self.cfg.net.type.lower()=='chronos':
            embeddings, _ = self.net.embed(x.squeeze(1).cpu()) # (B, context_len, n_features)
            x = embeddings.permute(0, 2, 1).to(x.device)
        else:
            x = self.net(x) # (B, h, L)
            
        print_verbose(f"net {x.shape}=", kwargs.get('verbose', 0))
        
        x_cls = self.class_head(self.pool1d(x).squeeze(-1))

        x_det = self.detection_head(x)
            
        return x_cls, x_det
