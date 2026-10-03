from torch import nn
import torch


class SpatialHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embedding_dim"]
        self.projection = nn.Conv2d(256, d, 1)
        self.position = nn.Parameter(torch.empty(1, 36, d))
        # Construct distinct layers, rather than cloning one initialized layer.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, cfg["attention_heads"], cfg["feedforward_dim"],
                                       dropout=cfg["encoder_dropout"], activation="gelu",
                                       batch_first=True, norm_first=True, layer_norm_eps=1e-5)
            for _ in range(cfg["encoder_layers"] if cfg["head"] == "encoder" else 0)])
        self.norm = nn.LayerNorm(d, eps=1e-5)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2) + self.position
        for layer in self.layers:
            x = layer(x)
        return self.dropout(self.norm(x).mean(1))
