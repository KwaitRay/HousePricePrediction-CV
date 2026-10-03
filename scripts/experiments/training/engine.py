"""Shared scratch training loop for all neural experiments."""
from pathlib import Path
import hashlib
import random
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from common.config import stable_seed, write_json
from common.metrics import evaluate, transform_target, inverse_target, save_predictions
from models.alexnet import AlexNetRegressor, initialize, penalty_parameters
from preparation.images import HouseImages, channel_stats
from training.loading import LoaderPool


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(ds, c, training=False):
    generator = torch.Generator().manual_seed(stable_seed(c["seed"], "shuffle" if training else "evaluation", ds.epoch))
    return DataLoader(ds, batch_size=c["training"]["batch_size"], shuffle=training,
                      num_workers=c["training"]["workers"], drop_last=False,
                      worker_init_fn=worker_seed, generator=generator, persistent_workers=False)


def optimizer_for(model, cfg):
    if cfg["optimizer"] in {"adam", "adagrad", "muon"}:
        from training.optimizers import optimizer_for as extended_optimizer
        return extended_optimizer(model, cfg)
    named = penalty_parameters(model)
    names = {name for name, _ in named}
    groups = [{"params": [p for _, p in named], "weight_decay": cfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if n not in names], "weight_decay": 0.}]
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=cfg["lr"], betas=(.9, .999), eps=1e-8)
    return torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"], dampening=0, nesterov=False)


def regularizer(parameters, cfg):
    if cfg["penalty"] == "none":
        return 0.
    if cfg["penalty"] == "l1":
        return cfg["penalty_coefficient"] * sum(p.abs().sum() for p in parameters)
    return cfg["penalty_coefficient"] * sum(p.square().sum() for p in parameters) / 2


class EarlyStop:
    def __init__(self, cfg):
        self.cfg, self.reference, self.bad = cfg, float("inf"), 0

    def update(self, value, epoch):
        if not np.isfinite(value):
            raise FloatingPointError("Early stopping received a non-finite validation metric")
        if value < self.reference * (1 - self.cfg["relative_improvement"]):
            self.reference, self.bad = value, 0
        else:
            self.bad += 1
        return self.cfg["enabled"] and epoch >= self.cfg["min_epochs"] and self.bad >= self.cfg["patience"]


@torch.no_grad()
def predict(model, ds, c, device, batches=None):
    model.eval()
    values, ids = [], []
    for batch in (loader(ds, c) if batches is None else batches):
        x, _, names = batch[:3]
        values.extend(model(x.to(device)).cpu().numpy().tolist())
        ids.extend(names)
    if ids != list(ds.frame.imageid):
        raise ValueError("预测顺序错误")
    return inverse_target(np.asarray(values, dtype=np.float64), c["target"])


def train_epoch(model, ds, c, optimizer, device, batches=None):
    """Normalize accumulated data loss by real samples; add penalty once per update."""
    model.train()
    cfg = c["training"]
    parameters = [p for _, p in penalty_parameters(model)]
    batches = loader(ds, c, True) if batches is None else batches
    optimizer.zero_grad(set_to_none=True)
    sample_count = window = updates = 0
    data_sum = penalty_sum = norm_sum = 0.
    augmentation_events = {}
    for i, batch in enumerate(batches):
        x, price, _ = batch[:3]
        if len(batch) > 3:
            for record in batch[3]:
                for name, event in json.loads(record).items():
                    total = augmentation_events.setdefault(name, {"samples": 0, "triggered": 0, "skipped": 0, "observed_ranges": {}})
                    total["samples"] += 1
                    total["triggered"] += int(event["triggered"])
                    total["skipped"] += int(event["skipped"])
                    for field, value in event.items():
                        if field in {"triggered", "skipped"}:continue
                        values = value if isinstance(value, list) else [value]
                        lo, hi = min(values), max(values)
                        bounds = total["observed_ranges"].setdefault(field, [lo, hi])
                        bounds[0], bounds[1] = min(bounds[0], lo), max(bounds[1], hi)
        x = x.to(device)
        price = price.to(device, dtype=torch.float32)
        y = torch.log1p(price) if c["target"]["transform"] == "log1p" else price / c["target"]["scale"]
        prediction = model(x)
        loss = (nn.functional.mse_loss(prediction, y, reduction="sum") if c["target"]["loss"] == "mse" else
                nn.functional.huber_loss(prediction, y, reduction="sum", delta=c["target"]["huber_delta"]))
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失为非有限值")
        loss.backward()
        n = len(x)
        window += n
        sample_count += n
        data_sum += loss.item()
        if (i + 1) % cfg["accumulation"] == 0 or i + 1 == len(batches):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(window)
            penalty = regularizer(parameters, cfg)
            if isinstance(penalty, torch.Tensor):
                penalty.backward()
                penalty_sum += float(penalty.detach())
            grad_norm = sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None) ** .5
            if not np.isfinite(grad_norm):
                raise FloatingPointError("梯度为非有限值")
            norm_sum += grad_norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window = 0
            updates += 1
    return {"data_loss": data_sum / sample_count, "penalty": penalty_sum / updates,
            "gradient_norm": norm_sum / updates, "samples": sample_count, "updates": updates,
            "augmentation": augmentation_events}


def fit_neural(c, root, train, validation, threshold, out, full=False):
    with LoaderPool(c, loader, worker_seed) as pool:
        return _fit_neural(c, root, train, validation, threshold, out, full, pool)


def _fit_neural(c, root, train, validation, threshold, out, full, pool):
    cfg = c["training"]
    torch.set_num_threads(cfg["cpu_threads"])
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(stable_seed(c["seed"], "dropout"))
    torch.use_deterministic_algorithms(True)
    device_name = c["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    stats = (channel_stats(root, train.imageid) if c["preprocess"]["normalization"] == "train_stats"
             else {"mean": [.5] * 3, "std": [.5] * 3})
    write_json(out / "normalization.json", stats)
    train_ds = HouseImages(root, train, c, stats, training=True)
    val_ds = HouseImages(root, validation, c, stats)
    probe = train.sample(min(32, len(train)), random_state=2026)
    probe_ds = HouseImages(root, probe, c, stats)
    model = AlexNetRegressor(c["model"])
    initialize(model, c["seed"], float(np.mean(transform_target(train.price.to_numpy(), c["target"]))), cfg["initialization"])
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.cpu().numpy().tobytes())
    write_json(out / "model_summary.json", {"architecture": str(model), "parameters": sum(p.numel() for p in model.parameters()),
               "initialization_sha256": h.hexdigest(), "penalty_parameters": [n for n, _ in penalty_parameters(model)],
               "device": str(device), "probe_ids": list(probe.imageid)})
    model.to(device)
    torch.manual_seed(stable_seed(c["seed"], "dropout"))
    optimizer = optimizer_for(model, cfg)
    setup_start = time.perf_counter()
    probe_batches = pool.get(probe_ds, probe=True)
    train_batches = pool.get(train_ds, training=True) if pool.optimized else None
    val_batches = pool.get(val_ds) if pool.optimized and not full else None
    loader_setup_seconds = time.perf_counter() - setup_start
    initial_pred = predict(model, probe_ds, c, device, probe_batches)
    write_json(out / "initial_probe.json", evaluate(probe.price.to_numpy(), initial_pred, threshold))
    schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * .01)
                if cfg["schedule"] == "cosine" else None)
    stopper, history = EarlyStop(cfg["early_stopping"]), []
    best, best_epoch = float("inf"), 0
    start = time.perf_counter()
    last_pred = None
    augmentation_history = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_ds.epoch = epoch
        step_start = time.perf_counter()
        row = train_epoch(model, train_ds, c, optimizer, device, train_batches)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row["training_seconds"] = time.perf_counter() - step_start
        row["seconds_per_update"] = row["training_seconds"] / row["updates"]
        augmentation_history.append({"epoch": epoch, "operations": row.pop("augmentation")})
        write_json(out / "augmentation_history.json", augmentation_history)
        row.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"])
        probe_start = time.perf_counter()
        probe_pred = predict(model, probe_ds, c, device, probe_batches)
        row["probe_mse"] = evaluate(probe.price.to_numpy(), probe_pred, threshold)["mse"]
        row["probe_seconds"] = time.perf_counter() - probe_start
        row["validation_seconds"] = row["checkpoint_seconds"] = 0.
        if not full:
            validation_start = time.perf_counter()
            last_pred = predict(model, val_ds, c, device, val_batches)
            metrics = evaluate(validation.price.to_numpy(), last_pred, threshold)
            row["validation_seconds"] = time.perf_counter() - validation_start
            row.update({"val_" + k: v for k, v in metrics.items() if k in {"mse", "mae", "tail_mse"}})
            if metrics["mse"] < best:
                best, best_epoch = metrics["mse"], epoch
                checkpoint_start = time.perf_counter()
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": c, "stats": stats}, out / "best.pt")
                row["checkpoint_seconds"] = time.perf_counter() - checkpoint_start
            stop = stopper.update(metrics["mse"], epoch)
        else:
            stop = False
        row["early_stop_counter"] = stopper.bad
        row["early_stop_reference"] = stopper.reference if not full else None
        row["seconds"] = time.perf_counter() - step_start
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False, encoding="utf-8-sig")
        print(f"epoch={epoch} loss={row['data_loss']:.6f} val_mse={row.get('val_mse', 'full-data')}", flush=True)
        if schedule:
            schedule.step()
        if stop:
            break
    if full:
        best_epoch = epoch
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": c, "stats": stats}, out / "best.pt")
        predict_seconds = None
    else:
        saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        predict_start = time.perf_counter()
        last_pred = predict(model, val_ds, c, device, val_batches)
        predict_seconds = time.perf_counter() - predict_start
        # Evaluate the selected checkpoint on all unaugmented training images to
        # diagnose generalization. These predictions never select the checkpoint.
        train_eval_start = time.perf_counter()
        train_eval_ds = HouseImages(root, train, c, stats)
        train_raw = predict(model, train_eval_ds, c, device, pool.get(train_eval_ds))
        train_eval_out = out / "train_evaluation"
        train_eval_out.mkdir()
        save_predictions(train_eval_out, train.reset_index(drop=True), train_raw, threshold)
        write_json(train_eval_out / "evaluation.json", {"checkpoint_epoch": best_epoch,
                   "seconds": time.perf_counter() - train_eval_start, "augmentation": False,
                   "purpose": "Post-selection train/validation gap diagnosis; not used to select checkpoint"})
    # Save fixed intermediate convolution features, without changing model state/RNG.
    model.eval()
    with torch.no_grad():
        x = next(iter(probe_batches))[0]
        features = model.features(x.to(device)).cpu().numpy()
    np.savez_compressed(out / "feature_maps.npz", features=features, imageids=probe.imageid.iloc[:len(features)].to_numpy(dtype=str))
    stop_reason = "early_stopping" if stop else "max_epochs"
    write_json(out / "stopping.json", {"reason": stop_reason, "epochs": epoch, "max_epochs": cfg["epochs"],
               "best_epoch": best_epoch, "counter": stopper.bad, "reference": stopper.reference if not full else None,
               "settings": cfg["early_stopping"], "monitor": "validation_original_price_mse" if not full else None})
    write_json(out / "cost.json", {"train_seconds": time.perf_counter()-start, "epochs": epoch, "best_epoch": best_epoch,
               "loader_setup_seconds": loader_setup_seconds, "stop_reason": stop_reason,
               "probe_seconds": sum(x["probe_seconds"] for x in history),
               "validation_seconds": sum(x["validation_seconds"] for x in history),
               "checkpoint_seconds": sum(x["checkpoint_seconds"] for x in history),
               "optimizer_updates": sum(x["updates"] for x in history), "samples_seen": sum(x["samples"] for x in history),
               "training_seconds": sum(x["training_seconds"] for x in history),
               "seconds_per_update": sum(x["training_seconds"] for x in history) / sum(x["updates"] for x in history),
               "predict_seconds": predict_seconds, "timing_note": "train_seconds包括逐轮验证与固定样本诊断；predict_seconds包括读取和预处理",
               "stopped_early": bool(stop), "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
    # Lightweight curve artifact; numerical history remains the authoritative record.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["data_loss"] for r in history])
    axes[0].set(title="Training loss", xlabel="Epoch")
    axes[1].plot([r["epoch"] for r in history], [r["probe_mse"] for r in history], label="Fixed train probe")
    if not full:
        axes[1].plot([r["epoch"] for r in history], [r["val_mse"] for r in history], label="Validation")
    axes[1].set(title="Original-price MSE", xlabel="Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=130)
    plt.close(fig)
    return last_pred
