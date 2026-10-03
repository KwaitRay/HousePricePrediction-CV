"""One evaluation definition for all methods; price unit is thousand USD."""
import numpy as np
import pandas as pd
from common.config import write_json


def evaluate(y, prediction, tail_threshold):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("真实值和预测值必须为同长的非空一维数组")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("真实值或预测值含非有限数")
    pred = np.maximum(pred, 0)
    err = pred - y
    tail = y > tail_threshold
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(abs(err))),
            "tail_mse": float(np.mean(err[tail] ** 2)) if tail.any() else None,
            "n": len(y), "tail_n": int(tail.sum()), "tail_threshold": float(tail_threshold)}


def save_predictions(out, frame, raw, threshold):
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    metrics = evaluate(frame.price.to_numpy(), raw, threshold)
    pred = np.maximum(raw, 0)
    result = frame[["imageid", "group_id", "price"]].copy()
    result["raw_prediction"], result["prediction"] = raw, pred
    result["error"] = pred - result.price
    result["squared_error"] = result.error ** 2
    result.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    write_json(out / "metrics.json", metrics)
    write_json(out / "prediction_checks.json", {"nonpositive_count": int((raw <= 0).sum()),
               "unclipped_mse": float(np.mean((raw - frame.price.to_numpy()) ** 2))})
    return metrics


def transform_target(y, config):
    return np.log1p(y) if config["transform"] == "log1p" else y / config["scale"]


def inverse_target(y, config):
    return np.expm1(y) if config["transform"] == "log1p" else y * config["scale"]
