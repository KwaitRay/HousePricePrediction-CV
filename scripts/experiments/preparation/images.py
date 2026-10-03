"""Deterministic preprocessing; isolated per-image, per-operation augmentation RNG."""
from pathlib import Path
import math
import json
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as F, InterpolationMode
from common.config import stable_seed

ORDER = ("crop", "translation", "rotation", "flip", "brightness", "contrast", "saturation", "blur", "noise", "erase")


def read_rgb(path):
    with Image.open(path) as im:
        return F.to_tensor(ImageOps.exif_transpose(im).convert("RGB"))


def filtering(x, mode):
    if mode == "spatial":
        return F.gaussian_blur(x, [5, 5], [.5, .5])
    if mode == "frequency":
        arr = x.numpy()
        ph, pw = math.ceil(arr.shape[1] * .1), math.ceil(arr.shape[2] * .1)
        arr = np.pad(arr, ((0, 0), (ph, ph), (pw, pw)), mode="reflect")
        fy, fx = np.fft.fftfreq(arr.shape[1]), np.fft.fftfreq(arr.shape[2])
        weight = np.exp(-(fy[:, None] ** 2 + fx[None, :] ** 2) / (2 * .25 ** 2))
        result = np.fft.ifft2(np.fft.fft2(arr) * weight).real[:, ph:-ph, pw:-pw]
        return torch.from_numpy(result.copy()).float().clamp(0, 1)
    return x


def augment(x, settings, seed, epoch, imageid):
    events = {}
    for name in ORDER:
        op = settings.get(name)
        if not op:
            continue
        rng = np.random.default_rng(stable_seed(seed, epoch, imageid, name))
        active = rng.random() < op["probability"]
        events[name] = {"triggered": bool(active), "skipped": False}
        if not active:
            continue
        h, w = x.shape[-2:]
        low, high = op.get("range", [0, 0])
        if name == "crop":
            area = rng.uniform(low, high)
            nh, nw = max(1, min(h, round(h * math.sqrt(area)))), max(1, min(w, round(w * math.sqrt(area))))
            top, left = int(rng.integers(h - nh + 1)), int(rng.integers(w - nw + 1))
            x = x[:, top:top + nh, left:left + nw]
            events[name]["area"] = nh * nw / (h * w)
        elif name == "translation":
            dx, dy = round(rng.uniform(low, high) * w), round(rng.uniform(low, high) * h)
            x = F.affine(x, 0, [dx, dy], 1, [0., 0.], interpolation=InterpolationMode.BILINEAR, fill=[.5] * 3)
            events[name]["pixels"] = [dx, dy]
        elif name == "rotation":
            angle = float(rng.uniform(low, high))
            x = F.rotate(x, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=[.5] * 3)
            events[name]["angle"] = angle
        elif name == "flip":
            x = F.hflip(x)
        elif name in {"brightness", "contrast", "saturation"}:
            factor = float(rng.uniform(low, high))
            x = getattr(F, f"adjust_{name}")(x, factor).clamp(0, 1)
            events[name]["factor"] = factor
        elif name == "blur":
            sigma = float(rng.uniform(low, high))
            x = F.gaussian_blur(x, [5, 5], [sigma, sigma])
            events[name]["sigma"] = sigma
        elif name == "noise":
            noise = rng.normal(0, op["std"], tuple(x.shape)).astype(np.float32)
            x = (x + torch.from_numpy(noise)).clamp(0, 1)
        elif name == "erase":
            placed = False
            for _ in range(10):
                area = rng.uniform(low, high) * h * w
                aspect = math.exp(rng.uniform(math.log(op["aspect"][0]), math.log(op["aspect"][1])))
                eh, ew = round(math.sqrt(area / aspect)), round(math.sqrt(area * aspect))
                if 0 < eh <= h and 0 < ew <= w:
                    top, left = int(rng.integers(h - eh + 1)), int(rng.integers(w - ew + 1))
                    x = x.clone()
                    x[:, top:top + eh, left:left + ew] = .5
                    events[name]["area"] = eh * ew / (h * w)
                    placed = True
                    break
            events[name]["skipped"] = not placed
    return x, events


def geometry(x, cfg):
    size = cfg["size"]
    h, w = x.shape[-2:]
    if cfg["geometry"] == "stretch":
        return F.resize(x, [size, size], antialias=True)
    if cfg["geometry"] == "center_crop":
        short = round(size * 256 / 224)
        return F.center_crop(F.resize(x, short, antialias=True), [size, size])
    nh, nw = max(1, round(h * size / max(h, w))), max(1, round(w * size / max(h, w)))
    resized = F.resize(x, [nh, nw], antialias=True)
    result = torch.full((3, size, size), .5, dtype=x.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    result[:, top:top + nh, left:left + nw] = resized
    return result


def channel_stats(root, ids):
    sums, squares, n = np.zeros(3), np.zeros(3), 0
    for name in ids:
        x = read_rgb(Path(root) / "train" / name).numpy().astype(np.float64).reshape(3, -1)
        sums += x.sum(1)
        squares += (x * x).sum(1)
        n += x.shape[1]
    mean = sums / n
    std = np.sqrt(np.maximum(squares / n - mean ** 2, 0))
    std[std == 0] = 1
    return {"mean": mean.tolist(), "std": std.tolist(), "pixels": n}


class HouseImages(Dataset):
    def __init__(self, root, frame, config, stats=None, training=False, partition="train"):
        self.root, self.frame, self.config = Path(root), frame.reset_index(drop=True), config
        self.stats = stats or {"mean": [.5] * 3, "std": [.5] * 3}
        self.training, self.partition, self.epoch = training, partition, 0

    def __len__(self):
        return len(self.frame)

    def view(self, index):
        row = self.frame.iloc[index]
        x = filtering(read_rgb(self.root / self.partition / row.imageid), self.config["preprocess"]["filter"])
        events = {}
        if self.training:
            x, events = augment(x, self.config["augmentation"], self.config["seed"], self.epoch, row.imageid)
        x = geometry(x, self.config["preprocess"])
        return x, events

    def __getitem__(self, index):
        x, events = self.view(index)
        row = self.frame.iloc[index]
        x = F.normalize(x, self.stats["mean"], self.stats["std"])
        return x, float(row.get("price", 0)), row.imageid, json.dumps(events)


def preview(root, frame, c, output, count=32, views=5):
    chosen = frame.sample(min(count, len(frame)), random_state=2026).sort_values("price")
    ds = HouseImages(root, chosen, c, training=True)
    size = c["preprocess"]["size"]
    sheet = Image.new("RGB", ((views + 1) * size, len(ds) * (size + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    logs = []
    for i in range(len(ds)):
        row = ds.frame.iloc[i]
        raw = geometry(read_rgb(Path(root) / "train" / row.imageid), c["preprocess"])
        sheet.paste(F.to_pil_image(raw), (0, i * (size + 20)))
        draw.text((0, i * (size + 20) + size), row.imageid, fill="black")
        for j in range(views):
            ds.epoch = j
            x, events = ds.view(i)
            sheet.paste(F.to_pil_image(x), ((j + 1) * size, i * (size + 20)))
            logs.append({"imageid": row.imageid, "view": j, "events": events})
    sheet.save(output)
    return logs
