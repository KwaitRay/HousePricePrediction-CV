"""Scratch-only AlexNet with explicitly scoped VGG/Inception/residual changes."""
import torch
from torch import nn
from common.config import stable_seed


class DeterministicAdaptivePool(nn.AdaptiveAvgPool2d):
    """Same adaptive-average bins, without CUDA's nondeterministic backward kernel."""
    def forward(self, x):
        oh, ow = self.output_size
        h, w = x.shape[-2:]
        if (h, w) == (oh, ow):
            return x
        if h % oh == 0 and w % ow == 0:
            return nn.functional.avg_pool2d(x, (h // oh, w // ow))
        rows = []
        for i in range(oh):
            start_h, end_h = i * h // oh, ((i + 1) * h + oh - 1) // oh
            rows.append(torch.stack([
                x[..., start_h:end_h, j * w // ow:((j + 1) * w + ow - 1) // ow].mean(dim=(-2, -1))
                for j in range(ow)], dim=-1))
        return torch.stack(rows, dim=-2)


class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        def conv(a, b, k):
            return nn.Sequential(nn.Conv2d(a, b, k, padding=k // 2), nn.ReLU())
        self.branches = nn.ModuleList([
            conv(192, 96, 1), nn.Sequential(conv(192, 64, 1), conv(64, 128, 3)),
            nn.Sequential(conv(192, 32, 1), conv(32, 96, 5)),
            nn.Sequential(nn.MaxPool2d(3, 1, 1), conv(192, 64, 1))])

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class AlexNetRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.conv2 = (nn.Sequential(nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(), nn.Conv2d(96, 192, 3, padding=1))
                      if cfg["small_kernels"] else nn.Conv2d(64, 192, 5, padding=2))
        self.conv3 = MultiScale() if cfg["inception"] else nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.shortcut = nn.Conv2d(384, 256, 1) if cfg["residual"] else None
        self.pool = nn.MaxPool2d(3, 2)
        self.spatial_pool = DeterministicAdaptivePool((6, 6))
        self.activation = nn.ReLU()
        if cfg["head"] == "alexnet":
            self.regressor = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(9216, 4096), nn.ReLU(),
                                           nn.Dropout(cfg["dropout"]), nn.Linear(4096, 4096), nn.ReLU())
            self.output = nn.Linear(4096, 1)
        else:
            from models.encoder import SpatialHead
            self.regressor = SpatialHead(cfg)
            self.output = nn.Linear(cfg["embedding_dim"], 1)

    def features(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        y = self.conv5(self.activation(self.conv4(x)))
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        return self.pool(self.activation(y))

    def forward(self, x):
        x = self.spatial_pool(self.features(x))
        if self.cfg["head"] == "alexnet":
            x = x.flatten(1)
        return self.output(self.regressor(x)).squeeze(-1)


def initialize(model, seed, target_mean, variant="default"):
    """Stable module-specific seeds keep shared layers paired across variants."""
    for name, module in model.named_modules():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed(seed, "initialization", name))
            if isinstance(module, nn.Conv2d):
                if variant == "xavier_convolution":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stable_seed(seed, "output"))
        nn.init.normal_(model.output.weight, std=.001)
        nn.init.constant_(model.output.bias, target_mean)
        for name, parameter in model.named_parameters():
            if name.endswith("position"):
                torch.manual_seed(stable_seed(seed, name))
                nn.init.normal_(parameter, std=.02)


def penalty_parameters(model):
    # Matrix/tensor weights only: excludes bias, normalization and position encoding.
    return [(name, p) for name, p in model.named_parameters() if p.ndim >= 2 and not name.endswith("position")]
