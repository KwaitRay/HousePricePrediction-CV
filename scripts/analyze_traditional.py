"""Turn verified stage-03 development runs into compact, shareable evidence.

No fitting, test predictions, parameter search or neural training happens here.
Raw per-image predictions and models remain under the ignored local run root.
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from common.config import read_json, write_json, path_at_root, digest
from common.metrics import evaluate
from preparation.split import load_split
from traditional.pipeline import extract


def keypoint_summary(values):
    values = np.asarray(values, dtype=int)
    return {"n": len(values), "zero_count": int((values == 0).sum()),
            "zero_fraction": float((values == 0).mean()),
            "mean": float(values.mean()),
            "quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                                  np.quantile(values, [0, .25, .5, .75, 1]).tolist()))}


def analyze(runs, output, mean_run):
    runs, output, mean_run = [Path(p).resolve() for p in runs], Path(output), Path(mean_run).resolve()
    if output.exists():
        raise FileExistsError("Analysis output exists; choose a new directory instead of replacing evidence")
    configs = [read_json(p / "config.json") for p in runs]
    cfg = configs[0]
    split, meta = load_split(cfg)
    train = split.loc[split.partition == "train"]
    validation = split.loc[split.partition == "validation"]
    if read_json(mean_run / "status.json") != {"status": "completed", "kind": "development"}:
        raise ValueError("Mean reference must be a completed development run")
    if read_json(mean_run / "data_manifest.json")["split_sha256"] != meta["split_sha256"]:
        raise ValueError("Mean reference and traditional runs use different splits")
    expected = validation.set_index("imageid").price.sort_index()
    rows, group_rows, frames, feature_stats = [], [], [], []
    seen_seeds = set()
    for run, config in zip(runs, configs):
        if read_json(run / "status.json") != {"status": "completed", "kind": "development"}:
            raise ValueError("Never report smoke, incomplete or full-retraining runs as validation evidence")
        if config["method"] != "traditional" or config["traditional"] != cfg["traditional"] or config["target"] != cfg["target"]:
            raise ValueError("This summary accepts repeated seeds of one registered traditional method")
        if config["seed"] in seen_seeds:
            raise ValueError("Repeated run of the same seed is not an independent seed")
        seen_seeds.add(config["seed"])
        manifest = read_json(run / "data_manifest.json")
        if manifest["split_sha256"] != meta["split_sha256"]:
            raise ValueError("Different frozen splits")
        if set(manifest["train_ids"]) != set(train.imageid) or set(manifest["validation_ids"]) != set(validation.imageid):
            raise ValueError("Run does not contain the full registered train/validation samples")
        pred = pd.read_csv(run / "predictions.csv", dtype={"imageid": str, "group_id": str})
        if pred.imageid.duplicated().any() or not pred.set_index("imageid").price.sort_index().equals(expected):
            raise ValueError("Prediction IDs or labels differ from frozen validation set")
        metrics = read_json(run / "metrics.json")
        recalculated = evaluate(pred.price, pred.raw_prediction, meta["tail_threshold"])
        for key in ("mse", "mae", "tail_mse"):
            if not np.isclose(metrics[key], recalculated[key], rtol=1e-10, atol=1e-8):
                raise ValueError(f"Saved metric does not match predictions: {key}")
        features = read_json(run / "features.json")
        val_features = read_json(run / "validation_features.json")
        if len(features["keypoints"]) != len(train) or len(val_features["keypoints"]) != len(validation):
            raise ValueError("Missing feature coverage records")
        cost = read_json(run / "cost.json")
        rows.append({"seed": config["seed"], **metrics,
                     "train_mse": features["train_metrics"]["mse"],
                     "train_seconds": cost["train_seconds"], "predict_seconds": cost["predict_seconds"],
                     "feature_dimension": features["feature_dimension"],
                     "train_cache_hits": features["extraction_cache"]["hits"],
                     "validation_cache_hits": val_features["extraction_cache"]["hits"]})
        # Price bins are fixed from TRAINING labels, not tuned on held-out errors.
        q1, q2 = train.price.quantile([1/3, 2/3]).to_numpy()
        threshold = meta["tail_threshold"]
        masks = {"low": pred.price <= q1,
                 "middle": (pred.price > q1) & (pred.price <= q2),
                 "high_non_tail": (pred.price > q2) & (pred.price <= threshold),
                 "tail": pred.price > threshold}
        for name, mask in masks.items():
            part = pred.loc[mask]
            group_rows.append({"seed": config["seed"], "group": name, "n": len(part),
                               "mse": float(part.squared_error.mean()),
                               "mae": float(abs(part.error).mean()),
                               "mean_prediction_minus_price": float(part.error.mean())})
        feature_stats.append({"seed": config["seed"],
                              "train": keypoint_summary(features["keypoints"]),
                              "validation": keypoint_summary(val_features["keypoints"]),
                              "vocabulary_cache_key": features["vocabulary_cache_key"],
                              "vocabulary_descriptors": features["vocabulary_descriptors"]})
        frames.append(pred)
    results = pd.DataFrame(rows).sort_values("seed")
    output.mkdir(parents=True)
    results.to_csv(output / "seed_results.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "price_group_results.csv", index=False)
    means = results[["mse", "mae", "tail_mse", "train_mse"]].mean().to_dict()
    deviations = results[["mse", "mae", "tail_mse", "train_mse"]].std(ddof=1).to_dict() if len(results) > 1 else None
    baseline = read_json(mean_run / "metrics.json")
    summary = {"n_seeds": len(results), "mean": means, "sample_standard_deviation": deviations,
               "mean_baseline": baseline,
               "mse_reduction_from_mean_percent": 100*(1 - means["mse"]/baseline["mse"]),
               "feature_coverage": feature_stats,
               "split": {**meta, "train_n": len(train), "validation_n": len(validation)},
               "traditional_settings": cfg["traditional"], "target": cfg["target"],
               "price_bin_thresholds_from_training": [float(q1), float(q2), float(threshold)],
               "notes": ["Metrics are per-model scores, NOT an ensemble of seed predictions.",
                         "Cold and cached fit times are recorded separately, not treated as model speed differences.",
                         "Split reconstructed with repository algorithm; teammate original hash still needs confirmation."]}
    write_json(output / "summary.json", summary)
    for run, config in zip(runs, configs):
        evidence = output / "runs" / str(config["seed"])
        # Keep compact configurations/metrics but do not export images, weights,
        # per-image labels/predictions, absolute local paths or the full split.
        share_config = {k: config[k] for k in ("method", "seed", "traditional", "target", "experiment")}
        write_json(evidence / "config.json", share_config)
        for filename in ("metrics.json", "cost.json", "status.json", "prediction_checks.json"):
            write_json(evidence / filename, read_json(run / filename))
        environment = read_json(run / "environment.json")
        write_json(evidence / "environment.json", {k: environment[k] for k in ("python", "packages", "source_hashes")})
        write_json(evidence / "evidence_hashes.json", {name: digest(run / name) for name in
                   ("config.json", "data_manifest.json", "predictions.csv", "features.json", "traditional.joblib")})
    # Example plots use one registered seed, never an unreported averaged model.
    seed = configs[0]["seed"]
    pred = frames[0]
    figures, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].scatter(pred.price, pred.prediction, s=10, alpha=.4)
    maximum = max(pred.price.max(), pred.prediction.max())
    axes[0].plot([0, maximum], [0, maximum], "k--", lw=1)
    axes[0].set(xlabel="Actual price (thousand USD)", ylabel="Predicted price (thousand USD)",
                title=f"SIFT-BoVW-Ridge, seed {seed}, n={len(pred)}")
    axes[1].scatter(pred.price, pred.error, s=10, alpha=.4)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="Actual price (thousand USD)", ylabel="Prediction - actual (thousand USD)",
                title="Validation residuals (clipped predictions)")
    figures.savefig(output / "validation_diagnostics.png", dpi=170)
    plt.close(figures)
    counts = read_json(runs[0] / "features.json")["keypoints"]
    val_counts = read_json(runs[0] / "validation_features.json")["keypoints"]
    figure, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bins = np.linspace(0, cfg["traditional"]["max_keypoints"], 33)
    ax.hist(counts, bins=bins, weights=np.full(len(counts), 1/len(counts)), alpha=.6, label=f"Train n={len(counts)}")
    ax.hist(val_counts, bins=bins, weights=np.full(len(val_counts), 1/len(val_counts)), alpha=.6, label=f"Validation n={len(val_counts)}")
    ax.set(xlabel="Retained SIFT keypoints per image", ylabel="Fraction of images", title="Feature coverage (including zero-keypoint images)")
    ax.legend()
    figure.savefig(output / "keypoint_distribution.png", dpi=170)
    plt.close(figure)
    # Deterministic TRAINING illustrations span the keypoint-count distribution;
    # they diagnose where descriptors occur, not causality or semantic objects.
    order = np.argsort(counts, kind="stable")
    chosen = order[np.linspace(0, len(order)-1, 6, dtype=int)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    image_root = path_at_root(cfg["data"]["root"])
    for ax, index in zip(axes.ravel(), chosen):
        name = train.iloc[index].imageid
        _, coords, _ = extract(image_root / "train" / name, cfg["traditional"])
        with Image.open(image_root / "train" / name) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        ax.imshow(rgb)
        if len(coords):
            ax.scatter(coords[:, 0]*rgb.shape[1], coords[:, 1]*rgb.shape[0],
                       s=8, facecolors="none", edgecolors="lime", linewidths=.6)
        ax.set_title(f"Train {name}: {len(coords)} keypoints")
        ax.axis("off")
    figure.suptitle("SIFT locations: inspect house vs vegetation/background coverage")
    figure.savefig(output / "keypoint_examples.png", dpi=150)
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--mean-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.runs, args.output, args.mean_run))
