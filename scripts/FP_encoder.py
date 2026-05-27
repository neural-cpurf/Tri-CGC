import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.4):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)  
            )
        else:
            self.shortcut = nn.Identity() 

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class FingerprintMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], output_dim=128, dropout=0.4):
        super(FingerprintMLP, self).__init__()

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(ResidualBlock(curr_dim, h_dim, dropout))
            curr_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.output_layer = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.output_layer(features)

        return out


