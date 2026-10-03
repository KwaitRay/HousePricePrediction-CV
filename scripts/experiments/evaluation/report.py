from pathlib import Path
import numpy as np
import pandas as pd
from common.config import read_json, write_json, differences


def compare_runs(paths, output, paired=False):
    paths = [Path(p) for p in paths]
    rows, configs, frames, costs = [], [], [], []
    expected = None
    for path in paths:
        status = read_json(path / "status.json")
        if status["status"] != "completed" or status["kind"] != "development":
            raise ValueError("正式比较只接受完整开发运行，拒绝小样本检查和全量重训")
        cfg = read_json(path / "config.json")
        manifest = read_json(path / "data_manifest.json")
        frame = pd.read_csv(path / "predictions.csv").sort_values("imageid").reset_index(drop=True)
        key = (manifest["split_sha256"], manifest["tail_threshold"])
        if expected is not None and expected != key:
            raise ValueError("不同划分或尾部阈值不能直接比较")
        expected = key
        if frames and not frame[["imageid", "price"]].equals(frames[0][["imageid", "price"]]):
            raise ValueError("预测记录或真实价格不一致")
        metric = read_json(path / "metrics.json")
        # Same method label cannot silently combine different hyperparameters.
        signature = {k: cfg[k] for k in ("method", "model", "training", "preprocess", "augmentation", "target", "traditional")}
        import json, hashlib
        version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
        rows.append({"experiment": cfg["experiment"]["name"], "config_version": version, "seed": cfg["seed"], "run": str(path),
                     **{k: metric[k] for k in ("mse", "mae", "tail_mse", "n", "tail_n")}})
        configs.append(cfg)
        frames.append(frame)
        if (path / "cost.json").exists():
            costs.append({"experiment": cfg["experiment"]["name"], "config_version": version,
                          "seed": cfg["seed"], "run": str(path), **read_json(path / "cost.json")})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runs = pd.DataFrame(rows)
    if runs.duplicated(["experiment", "config_version", "seed"]).any():
        raise ValueError("同一配置和种子的重复运行不能计作独立种子")
    runs.to_csv(output / "runs.csv", index=False, encoding="utf-8-sig")
    if costs:
        pd.DataFrame(costs).to_csv(output / "costs.csv", index=False, encoding="utf-8-sig")
    runs.groupby(["experiment", "config_version"])[["mse", "mae", "tail_mse"]].agg(["mean", "std", "count"]).to_csv(output / "summary.csv", encoding="utf-8-sig")
    if paired:
        if len(paths) != 2 or configs[0]["seed"] != configs[1]["seed"]:
            raise ValueError("单次成对比较需要两个相同种子的运行")
        if all((p / "environment.json").exists() for p in paths):
            environments = [read_json(p / "environment.json") for p in paths]
            for key in ("packages", "source_hashes", "gpu", "cuda", "cudnn"):
                if environments[0].get(key) != environments[1].get(key):
                    raise ValueError(f"成对比较的环境或代码不同，请重跑对照: {key}")
        diff = differences(configs[0], configs[1])
        allowed = configs[1]["experiment"]["changed_fields"]
        unexpected = [k for k in diff if not any(k == f or k.startswith(f + ".") for f in allowed)]
        if unexpected:
            raise ValueError(f"成对比较存在未登记差异: {unexpected}")
        write_json(output / "paired_config_diff.json", diff)
        result = frames[0][["imageid", "group_id", "price"]].copy()
        result["squared_error_change"] = frames[1].squared_error - frames[0].squared_error
        result.to_csv(output / "paired_errors.csv", index=False, encoding="utf-8-sig")
        groups = [g.squared_error_change.to_numpy() for _, g in result.groupby("group_id")]
        rng = np.random.default_rng(2026)
        samples = [float(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()) for _ in range(1000)]
        write_json(output / "paired_bootstrap.json", {"candidate_minus_control_mse": float(result.squared_error_change.mean()),
                   "interval_95": np.quantile(samples, [.025, .975]).tolist(), "unit": "original_image_group", "draws": 1000})
    return output
