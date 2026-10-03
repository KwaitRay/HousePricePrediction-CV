"""Offline dataset audit and training-only EDA. Never trains on test images.

Run from any directory: python /path/to/scripts/analyze_dataset.py --help
All results go into a new run directory; source data are never modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts/experiments"))
from common.local_paths import local_path
DEFAULT_DATA = local_path("data_root")
FEATURES = ["width", "height", "aspect_ratio", "file_bytes", "brightness",
            "contrast", "dark_fraction", "bright_fraction", "laplacian_variance"]
NEAR_COLUMNS = ["imageid_a", "imageid_b", "hamming_distance", "price_a",
                "price_b", "absolute_price_gap"]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def audit_tables(root: Path, expected_train: int, expected_test: int):
    """Reject ambiguous schemas and unsafe filenames before accessing images."""
    tables, errors, warnings, details = {}, [], [], {}
    definitions = {"train": ["imageid", "price"], "test": ["imageid"],
                   "sample_solution": ["imageid", "price"]}
    for name, columns in definitions.items():
        path = root / f"{name}.csv"
        if not path.is_file():
            errors.append(f"Missing CSV: {path.name}")
            continue
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False,
                                encoding="utf-8-sig")
        except Exception as exc:
            errors.append(f"Cannot read {path.name}: {exc}")
            continue
        tables[name] = frame
        details[name] = {"rows": len(frame), "columns": list(frame.columns),
                         "sha256": file_sha256(path)}
        if list(frame.columns) != columns:
            errors.append(f"{path.name}: expected columns {columns}, got {list(frame.columns)}")
            continue
        expected = expected_train if name == "train" else expected_test
        if len(frame) != expected:
            errors.append(f"{path.name}: expected {expected} rows, got {len(frame)}")
        if frame.empty:
            errors.append(f"{path.name}: empty table")
        invalid = ~frame.imageid.str.fullmatch(r"[0-9]+\.jpg")
        if invalid.any():
            errors.append(f"{path.name}: {int(invalid.sum())} unsafe/nonstandard image IDs")
        if frame.imageid.duplicated().any():
            errors.append(f"{path.name}: duplicate image IDs")
        if "price" in columns:
            prices = pd.to_numeric(frame.price, errors="coerce")
            bad = ~np.isfinite(prices) | (prices <= 0)
            if bad.any():
                errors.append(f"{path.name}: {int(bad.sum())} invalid/nonpositive prices")
            frame["price"] = prices
    if not errors:
        train_ids = set(tables["train"].imageid)
        test_ids = set(tables["test"].imageid)
        if train_ids & test_ids:
            errors.append("Train/test image IDs overlap")
        sample_ids = tables["sample_solution"].imageid.tolist()
        if set(sample_ids) != test_ids:
            errors.append("Sample solution IDs differ from test.csv")
        elif sample_ids != tables["test"].imageid.tolist():
            warnings.append("Sample solution row order differs from test.csv; join by imageid")
        for split in ["train", "test"]:
            directory = root / split
            if not directory.is_dir():
                errors.append(f"Missing image directory: {split}")
                continue
            disk = {p.name for p in directory.iterdir() if p.is_file()}
            ids = set(tables[split].imageid)
            missing, extra = sorted(ids - disk), sorted(disk - ids)
            details[split]["missing_files"] = missing
            details[split]["unlisted_files"] = extra
            if missing:
                errors.append(f"{split}: {len(missing)} listed images are missing")
            if extra:
                warnings.append(f"{split}: {len(extra)} unlisted files (not analyzed)")
    return tables, {"errors": errors, "warnings": warnings, "tables": details}


def dhash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


def inspect_image(task: tuple[str, str, Path]) -> dict:
    split, imageid, path = task
    row = {"split": split, "imageid": imageid, "error": ""}
    try:
        row["file_bytes"] = path.stat().st_size
        row["file_sha256"] = file_sha256(path)
        with Image.open(path) as source:
            source.load()  # Force a full decode, not just a header check.
            row.update({"format": source.format, "mode": source.mode,
                        "raw_width": source.width, "raw_height": source.height})
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            row.update({"width": rgb.width, "height": rgb.height,
                        "aspect_ratio": rgb.width / rgb.height})
            payload = f"RGB:{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
            row["pixel_sha256"] = hashlib.sha256(payload).hexdigest()
            if split == "train":
                row["dhash"] = f"{dhash(rgb):016x}"
                gray = np.asarray(rgb.convert("L").resize((128, 128),
                                  Image.Resampling.LANCZOS), dtype=np.float64)
                laplace = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1]
                           + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
                row.update({"brightness": float(gray.mean()),
                            "contrast": float(gray.std()),
                            "dark_fraction": float((gray <= 5).mean()),
                            "bright_fraction": float((gray >= 250).mean()),
                            "laplacian_variance": float(laplace.var())})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def scan_images(root: Path, tables: dict, max_images: int, workers: int,
                seed: int, out: Path) -> pd.DataFrame:
    tasks = []
    for offset, split in enumerate(["train", "test"]):
        frame = tables[split]
        if max_images and max_images < len(frame):
            frame = frame.sample(n=max_images, random_state=seed + offset)
        tasks.extend((split, imageid, root / split / imageid) for imageid in frame.imageid)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(inspect_image, tasks), 1):
            rows.append(row)
            if index % 250 == 0 or index == len(tasks):
                print(f"Images: {index}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows)
    save_csv(frame, out / "tables" / "image_inventory.csv")
    return frame


def exact_duplicate_members(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in ["file_sha256", "pixel_sha256"]:
        if kind not in inventory:
            continue
        valid = inventory[inventory.error.eq("") & inventory[kind].notna()]
        for key, members in valid.groupby(kind, sort=True):
            if len(members) < 2:
                continue
            scope = "cross_train_test" if members.split.nunique() > 1 else members.split.iloc[0]
            for item in members.itertuples():
                # Deliberately omit labels from this cross-split audit table.
                rows.append({"hash_kind": kind, "group_hash": key, "scope": scope,
                             "group_size": len(members), "split": item.split,
                             "imageid": item.imageid})
    return pd.DataFrame(rows, columns=["hash_kind", "group_hash", "scope",
                                     "group_size", "split", "imageid"])


class BKTree:
    """Exact Hamming-radius lookup for 64-bit hashes; no quadratic distance matrix."""
    def __init__(self):
        self.root = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int):
        pending = [self.root] if self.root is not None else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for index in node[1]:
                    yield index, distance
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def near_duplicate_pairs(train: pd.DataFrame, radius: int, max_pairs: int):
    records = train.sort_values("imageid").to_dict("records")
    tree, pairs = BKTree(), []
    truncated = False
    for index, row in enumerate(records):
        value = int(row["dhash"], 16)
        for other_index, distance in tree.query(value, radius):
            other = records[other_index]
            if row["pixel_sha256"] == other["pixel_sha256"]:
                continue  # Exact groups have their own complete report.
            if len(pairs) == max_pairs:
                truncated = True
                break
            pairs.append({"imageid_a": other["imageid"], "imageid_b": row["imageid"],
                          "hamming_distance": distance, "price_a": other["price"],
                          "price_b": row["price"],
                          "absolute_price_gap": abs(row["price"] - other["price"])})
        if truncated:
            break
        tree.add(value, index)
        if (index + 1) % 1000 == 0:
            print(f"Near-duplicate search: {index + 1}/{len(records)}", flush=True)
    frame = pd.DataFrame(pairs, columns=NEAR_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["hamming_distance", "imageid_a", "imageid_b"])
    return frame, truncated


def make_split(train: pd.DataFrame, validation_fraction: float, seed: int) -> pd.DataFrame:
    """Draft: exact-pixel groups stay intact, group median prices define strata.

    Equal-count rank strata are used to handle tied group prices. They are not
    class labels. Group counts, rather than row counts, target the fraction.
    """
    groups = (train.groupby("pixel_sha256", sort=True).price
              .agg(["median", "size"]).reset_index()
              .sort_values(["median", "pixel_sha256"]).reset_index(drop=True))
    if len(groups) < 2:
        raise ValueError("Need at least two distinct exact-image groups for a split")
    min_per_bin = max(2, math.ceil(1 / validation_fraction))
    bins = max(1, min(10, len(groups) // min_per_bin))
    groups["price_stratum"] = np.minimum(np.arange(len(groups)) * bins // len(groups), bins - 1)
    groups["partition"] = "train"
    rng = np.random.default_rng(seed)
    for _, members in groups.groupby("price_stratum", sort=True):
        count = max(1, min(len(members) - 1, round(len(members) * validation_fraction)))
        chosen = rng.permutation(members.index.to_numpy())[:count]
        groups.loc[chosen, "partition"] = "validation"
    manifest = train[["imageid", "price", "pixel_sha256"]].merge(
        groups[["pixel_sha256", "price_stratum", "partition"]],
        on="pixel_sha256", how="left", validate="many_to_one")
    manifest = manifest.rename(columns={"pixel_sha256": "exact_group"})
    if manifest.partition.nunique() != 2:
        raise ValueError("Could not create two nonempty partitions")
    if manifest.groupby("exact_group").partition.nunique().max() != 1:
        raise AssertionError("Exact duplicates crossed the split")
    return manifest.sort_values("imageid").reset_index(drop=True)


def baseline_metrics(manifest: pd.DataFrame):
    training = manifest.loc[manifest.partition.eq("train"), "price"].to_numpy()
    validation = manifest.loc[manifest.partition.eq("validation"), ["imageid", "price"]].copy()
    rows = []
    for name, prediction in [("training_mean", training.mean()),
                             ("training_median", np.median(training))]:
        residual = prediction - validation.price.to_numpy()
        mse = float(np.mean(residual ** 2))
        rows.append({"baseline": name, "constant_prediction": float(prediction),
                     "validation_rows": len(validation), "mse": mse,
                     "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(residual)))})
        validation[f"{name}_prediction"] = prediction
        validation[f"{name}_squared_error"] = residual ** 2
    return pd.DataFrame(rows), validation


def save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def price_analysis(prices: pd.Series, out: Path) -> dict:
    statistics = prices.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99])
    q1, q3 = prices.quantile([.25, .75])
    lower, upper = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
    result = {"unit": "1000 USD", "statistics": statistics.to_dict(),
              "skewness": float(prices.skew()), "unique_values": int(prices.nunique()),
              "iqr_lower_fence": lower, "iqr_upper_fence": upper,
              "iqr_flagged_rows": int(((prices < lower) | (prices > upper)).sum()),
              "rows_equal_to_min": int(prices.eq(prices.min()).sum()),
              "rows_equal_to_max": int(prices.eq(prices.max()).sum())}
    save_csv(statistics.rename("value").rename_axis("statistic").reset_index(),
             out / "tables" / "price_statistics.csv")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(prices, bins=50, color="#3579ac")
    axes[0, 0].set(xlabel="Price (1000 USD)", ylabel="Training rows", title="Raw target")
    axes[0, 1].hist(np.log1p(prices), bins=50, color="#529c77")
    axes[0, 1].set(xlabel="log(1 + price in 1000 USD)", ylabel="Training rows", title="Log target (EDA only)")
    ordered = np.sort(prices)
    axes[1, 0].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered))
    axes[1, 0].set(xlabel="Price (1000 USD)", ylabel="Cumulative fraction", title="Empirical CDF")
    axes[1, 1].boxplot(prices)
    axes[1, 1].set(ylabel="Price (1000 USD)", xticks=[], title="IQR flags are not deletion rules")
    save_figure(figure, out / "figures" / "price_distribution.png")
    return result


def image_analysis(train: pd.DataFrame, out: Path, seed: int) -> dict:
    if train.empty:
        return {"rows": 0}
    descriptions = train[FEATURES].describe(percentiles=[.01, .05, .5, .95, .99]).T
    save_csv(descriptions.rename_axis("feature").reset_index(),
             out / "tables" / "train_image_statistics.csv")
    correlation = train[FEATURES + ["price"]].rank(method="average").corr(method="pearson")
    save_csv(correlation.rename_axis("feature").reset_index(),
             out / "tables" / "train_feature_spearman.csv")
    candidates = []
    for feature in ["brightness", "contrast", "laplacian_variance", "aspect_ratio"]:
        low, high = train[feature].quantile([.01, .99])
        for tail, mask in [("bottom_1_percent", train[feature] <= low),
                           ("top_1_percent", train[feature] >= high)]:
            for row in train.loc[mask, ["imageid", "price", feature]].itertuples(index=False, name=None):
                candidates.append({"imageid": row[0], "price": row[1], "feature": feature,
                                   "tail": tail, "value": row[2]})
    save_csv(pd.DataFrame(candidates), out / "tables" / "quality_review_candidates.csv")
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    sampled = train.sample(n=min(2000, len(train)), random_state=seed)
    for axis, feature in zip(axes.flat, ["aspect_ratio", "file_bytes", "brightness",
                                       "contrast", "dark_fraction", "laplacian_variance"]):
        axis.scatter(sampled[feature], sampled.price, s=7, alpha=.25)
        axis.set(xlabel=feature, ylabel="Price (1000 USD)")
    save_figure(figure, out / "figures" / "train_quality_vs_price.png")
    return {"rows": len(train), "spearman_with_price": correlation.price.to_dict(),
            "quality_flag_policy": "training-scan 1st/99th percentiles; manual review only"}


def contact_sheet(root: Path, train: pd.DataFrame, ids: list[str], output: Path,
                  title: str, columns: int = 4) -> None:
    if not ids:
        return
    prices = train.set_index("imageid").price.to_dict()
    rows = math.ceil(len(ids) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 2.7), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, imageid in zip(axes.flat, ids):
        with Image.open(root / "train" / imageid) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.set_title(f"{imageid} | {prices[imageid]:.1f}k USD", fontsize=9)
    figure.suptitle(title)
    save_figure(figure, output)


def make_galleries(root: Path, train: pd.DataFrame, near: pd.DataFrame,
                   out: Path, seed: int) -> None:
    if train.empty:
        return
    random_ids = train.sample(n=min(16, len(train)), random_state=seed).imageid.tolist()
    contact_sheet(root, train, random_ids, out / "figures" / "train_random_examples.png",
                  "Random training examples; not a quality ranking")
    tails = pd.concat([train.nsmallest(8, "price"), train.nlargest(8, "price")])
    contact_sheet(root, train, tails.imageid.tolist(), out / "figures" / "train_price_extremes.png",
                  "Lowest then highest labeled prices; not automatic outliers")
    blurred = train.nsmallest(16, "laplacian_variance").imageid.tolist()
    contact_sheet(root, train, blurred, out / "figures" / "train_low_sharpness.png",
                  "Lowest resized Laplacian variance; inspect before deciding")
    if not near.empty:
        ids = [imageid for row in near.head(8).itertuples()
               for imageid in [row.imageid_a, row.imageid_b]]
        contact_sheet(root, train, ids, out / "figures" / "train_near_duplicate_pairs.png",
                      "Each adjacent pair is a dHash candidate, not confirmed duplication")


def split_analysis(train: pd.DataFrame, near: pd.DataFrame, out: Path,
                   fraction: float, seed: int, near_status: str) -> dict:
    manifest = make_split(train, fraction, seed)
    partitions = manifest.set_index("imageid").partition
    crossing = near[near.imageid_a.map(partitions) != near.imageid_b.map(partitions)].copy()
    save_csv(manifest, out / "tables" / "split_candidate.csv")
    save_csv(crossing, out / "tables" / "near_pairs_crossing_split.csv")
    metrics, residuals = baseline_metrics(manifest)
    save_csv(metrics, out / "tables" / "baseline_metrics.csv")
    save_csv(residuals, out / "tables" / "baseline_validation_predictions.csv")
    training_prices = manifest.loc[manifest.partition.eq("train"), "price"]
    cutpoints = np.unique(training_prices.quantile(np.arange(.1, 1, .1)).to_numpy())
    residuals["price_band"] = pd.cut(residuals.price, [-np.inf, *cutpoints, np.inf]).astype(str)
    contribution = residuals.groupby("price_band", sort=False).agg(
        rows=("imageid", "size"), mean_price=("price", "mean"),
        mean_baseline_mse=("training_mean_squared_error", "mean"),
        squared_error_sum=("training_mean_squared_error", "sum")).reset_index()
    total = contribution.squared_error_sum.sum()
    contribution["fraction_of_total_squared_error"] = (
        contribution.squared_error_sum / total if total > 0 else 0.0)
    save_csv(contribution, out / "tables" / "baseline_error_by_price_band.csv")
    distribution = manifest.groupby("partition").price.describe().reset_index()
    save_csv(distribution, out / "tables" / "split_price_statistics.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    edges = np.histogram_bin_edges(manifest.price, bins=30)
    for partition, color in [("train", "#3579ac"), ("validation", "#d68339")]:
        axis.hist(manifest.loc[manifest.partition.eq(partition), "price"], bins=edges,
                  density=True, histtype="step", label=partition, color=color)
    axis.set(xlabel="Price (1000 USD)", ylabel="Density", title="Draft split target distribution")
    axis.legend()
    save_figure(figure, out / "figures" / "split_price_distribution.png")
    return {"status": "draft_requires_review", "seed": seed,
            "requested_validation_fraction": fraction,
            "actual_validation_fraction": float(manifest.partition.eq("validation").mean()),
            "row_counts": manifest.partition.value_counts().to_dict(),
            "exact_group_count": int(manifest.exact_group.nunique()),
            "exact_groups_crossing_split": 0,
            "known_near_pairs_crossing_split": len(crossing),
            "near_search_status": near_status, "baselines": metrics.to_dict("records")}


def write_report(summary: dict, out: Path) -> None:
    inventory = summary.get("images", {})
    split = summary.get("split", {})
    lines = ["# 数据分析运行报告", "",
             f"运行状态：`{summary['status']}`；是否全量扫描：`{summary.get('full_scan', False)}`。", "",
             "本报告由本地脚本生成。价格单位为千美元；测试图片仅用于完整性和完全重复审计。", "",
             "## 1. 数据完整性", "",
             f"- Schema 错误：{len(summary['audit']['errors'])}。",
             f"- 成功解码图片：{inventory.get('decoded', 0)}；解码失败：{inventory.get('failed', 0)}。"]
    lines.extend(f"- 错误：{item}" for item in summary['audit']['errors'])
    lines.extend(f"- 提示：{item}" for item in summary.get("warnings", []))
    if "prices" in summary:
        prices = summary["prices"]
        stats = prices["statistics"]
        lines += ["", "## 2. 训练售价", "",
                  f"- 数量 {int(stats['count'])}；范围 {stats['min']:.3f}–{stats['max']:.3f} 千美元。",
                  f"- 均值 {stats['mean']:.3f}；中位数 {stats['50%']:.3f} 千美元。",
                  f"- IQR 规则标记 {prices['iqr_flagged_rows']} 条，仅供复核，不自动删除。",
                  "- 最大值重复出现不证明数据被截断；需要进一步证据。", "",
                  "![售价分布](figures/price_distribution.png)"]
    if "duplicates" in summary:
        dup = summary["duplicates"]
        lines += ["", "## 3. 重复与近重复", "",
                  f"- 完全重复组统计：`{dup['groups_by_kind_and_scope']}`。",
                  f"- 训练近重复候选对：{dup['near_pairs_saved']}；搜索状态：`{dup['near_search_status']}`。",
                  "- 字节哈希识别相同文件；像素哈希识别 EXIF 校正后的相同 RGB 像素及尺寸。",
                  "- dHash 只产生候选，不能确认同一房屋。跨训练/测试审计表不包含售价，不用于复制标签。"]
    lines += ["", "## 4. 验证集划分草案", "",
              f"- 状态：`{split.get('status', 'not_generated')}`。"]
    if "row_counts" in split:
        lines += [f"- 样本数：`{split['row_counts']}`；实际验证比例 {split['actual_validation_fraction']:.3%}。",
                  f"- 完全重复组跨集合数量：0；已知近重复候选跨集合数量：{split['known_near_pairs_crossing_split']}。",
                  "- 此划分是草案；先复核近重复、标签冲突及比例，再决定是否冻结为训练配置。",
                  "- 下表是本地常数基线，不是课程官方 baseline，也不是 Kaggle 测试成绩。", "",
                  "| 基线 | 验证 MSE | 验证 RMSE | 验证 MAE |", "| --- | ---: | ---: | ---: |"]
        for row in split["baselines"]:
            lines.append(f"| {row['baseline']} | {row['mse']:.3f} | {row['rmse']:.3f} | {row['mae']:.3f} |")
    else:
        lines.append(f"- 原因：{split.get('reason', 'schema 检查未通过')}。")
    lines += ["", "## 5. 阅读顺序与解释边界", "",
              "1. 先查看 `summary.json` 的错误、告警和全量标志。",
              "2. 查看 `tables/train_exact_label_conflicts.csv` 和近重复候选表及配对图。",
              "3. 查看售价分布、质量特征、训练图片示例，确认需要研究的问题。",
              "4. 查看划分草案及本地基线，复核后再冻结实验划分。", "",
              "质量分位数和相关性只用于探索；低清晰度不一定是坏图，相关性不表示因果。",
              "EDA 使用全部已提供训练标签，因此本地验证不应被描述为完全未观察过的盲测集。",
              "本次运行不删除图片、不改标签、不训练视觉模型、不生成或上传 Kaggle 提交。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args, out: Path) -> dict:
    tables, audit = audit_tables(args.data_root, args.expected_train, args.expected_test)
    summary = {"status": "running", "audit": audit, "warnings": list(audit["warnings"])}
    if audit["errors"]:
        summary["status"] = "failed_schema"
        return summary
    print("CSV schema passed; scanning images...", flush=True)
    inventory = scan_images(args.data_root, tables, args.max_images, args.workers, args.seed, out)
    good = inventory[inventory.error.eq("")].copy()
    save_csv(inventory[~inventory.error.eq("")], out / "tables" / "image_errors.csv")
    full = len(inventory) == len(tables["train"]) + len(tables["test"])
    summary["full_scan"] = full
    summary["images"] = {"scanned": len(inventory), "decoded": len(good),
                         "failed": len(inventory) - len(good),
                         "scanned_by_split": inventory.split.value_counts().to_dict()}
    if not full:
        summary["warnings"].append("Partial image scan: duplicate findings are incomplete; no split/baseline generated")
    if not good.empty:
        for field in ["format", "mode"]:
            counts = good.groupby(["split", field]).size().rename("count").reset_index()
            save_csv(counts, out / "tables" / f"image_{field}_counts.csv")
        sizes = good.groupby(["split", "width", "height"]).size().rename("count").reset_index()
        save_csv(sizes, out / "tables" / "image_size_counts.csv")
    train_columns = ["imageid", "pixel_sha256", "dhash", *FEATURES]
    for column in train_columns:
        if column not in good:
            good[column] = pd.Series(index=good.index, dtype=object)
    train = good.loc[good.split.eq("train")].merge(tables["train"], on="imageid", validate="one_to_one")
    save_csv(train, out / "tables" / "train_image_features.csv")
    summary["prices"] = price_analysis(tables["train"].price, out)
    p = tables["train"].copy()
    p["iqr_flag"] = (p.price < summary["prices"]["iqr_lower_fence"]) | (p.price > summary["prices"]["iqr_upper_fence"])
    save_csv(p[p.iqr_flag], out / "tables" / "price_review_candidates.csv")
    summary["train_image_analysis"] = image_analysis(train, out, args.seed)
    exact = exact_duplicate_members(inventory)
    save_csv(exact, out / "tables" / "exact_duplicate_members.csv")
    conflicts = (train.groupby("pixel_sha256").price.agg(["size", "nunique", "min", "max"])
                 .reset_index())
    conflict_hashes = set(conflicts.loc[conflicts["nunique"] > 1, "pixel_sha256"])
    save_csv(train.loc[train.pixel_sha256.isin(conflict_hashes), ["imageid", "price", "pixel_sha256"]],
             out / "tables" / "train_exact_label_conflicts.csv")
    if args.skip_near_duplicates:
        near, near_status = pd.DataFrame(columns=NEAR_COLUMNS), "skipped"
    else:
        print("Searching training-only near duplicates...", flush=True)
        near, truncated = near_duplicate_pairs(train, args.near_threshold, args.max_near_pairs)
        near_status = "truncated" if truncated else ("complete" if full else "sample_only")
    save_csv(near, out / "tables" / "train_near_duplicate_candidates.csv")
    counts = exact.drop_duplicates(["hash_kind", "group_hash"]).groupby(["hash_kind", "scope"]).size()
    summary["duplicates"] = {"groups_by_kind_and_scope": {f"{k[0]}:{k[1]}": int(v) for k, v in counts.items()},
                             "train_pixel_groups_with_label_conflicts": len(conflict_hashes),
                             "near_pairs_saved": len(near), "near_search_status": near_status}
    if conflict_hashes:
        summary["warnings"].append(f"{len(conflict_hashes)} exact training-image groups have inconsistent prices")
    if near_status != "complete":
        summary["warnings"].append(f"Near-duplicate audit is {near_status}; absence of recorded pairs is not clearance")
    make_galleries(args.data_root, train, near, out, args.seed)
    if full and len(train) == len(tables["train"]):
        try:
            summary["split"] = split_analysis(train, near, out, args.validation_fraction, args.seed, near_status)
        except ValueError as exc:
            summary["split"] = {"status": "not_generated", "reason": str(exc)}
    else:
        summary["split"] = {"status": "not_generated", "reason": "需要全量扫描且全部训练图片成功解码"}
    summary["status"] = "completed" if inventory.error.eq("").all() else "completed_with_image_errors"
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=local_path("eda_output_root"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="Per split; 0 = all. Partial mode skips split/baselines.")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--near-threshold", type=int, default=4, help="Training dHash Hamming radius (0..8).")
    parser.add_argument("--max-near-pairs", type=int, default=50000)
    parser.add_argument("--skip-near-duplicates", action="store_true")
    parser.add_argument("--expected-train", type=int, default=8000, help="Override only for a deliberately different dataset/fixture.")
    parser.add_argument("--expected-test", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_images < 0 or args.max_near_pairs < 1:
        parser.error("workers/max-near-pairs must be positive; max-images must be nonnegative")
    if not 0 < args.validation_fraction <= .5:
        parser.error("validation-fraction must be in (0, 0.5]")
    if not 0 <= args.near_threshold <= 8 or args.seed < 0 or args.seed >= 2**32 - 1:
        parser.error("near-threshold must be 0..8 and seed must be 0..4294967294")
    if args.expected_train < 2 or args.expected_test < 1:
        parser.error("expected-train must be >= 2 and expected-test >= 1")
    args.data_root, args.output_root = args.data_root.resolve(), args.output_root.resolve()
    if args.output_root == args.data_root or args.data_root in args.output_root.parents:
        parser.error("output-root must be outside data-root")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out = args.output_root / run_id
    for folder in [out, out / "tables", out / "figures"]:
        folder.mkdir(parents=True, exist_ok=False)
    print(f"Output: {out}", flush=True)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": file_sha256(Path(__file__)), "python": platform.python_version(),
                "versions": {name: version(name) for name in ["numpy", "pandas", "Pillow", "matplotlib"]},
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "status": "running"}
    write_json(out / "run_manifest.json", manifest)
    try:
        summary = run_analysis(args, out)
        write_json(out / "summary.json", summary)
        write_report(summary, out)
        manifest["status"] = summary["status"]
        code = 0 if summary["status"] == "completed" else 2
    except Exception as exc:
        manifest["status"] = "failed_runtime"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Analysis failed: {manifest['error']}", file=sys.stderr)
        code = 2
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "run_manifest.json", manifest)
    print(f"Status: {manifest['status']}. Results: {out}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
