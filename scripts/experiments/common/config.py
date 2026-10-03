"""JSON inheritance, explicit experiment lineage, and immutable run records."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import re
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def merge(parent, child):
    out = deepcopy(parent)
    for key, val in child.items():
        out[key] = merge(out[key], val) if isinstance(val, dict) and isinstance(out.get(key), dict) else deepcopy(val)
    return out


def resolve(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"配置继承循环: {path}")
    if not path.exists():
        raise FileNotFoundError(f"尚未登记前一阶段选定配置，或配置不存在: {path}")
    raw = read_json(path)
    parent = resolve(path.parent / raw["extends"], (*stack, path)) if raw.get("extends") else {}
    result = merge(parent, {k: v for k, v in raw.items() if k != "extends"})
    result["_source"] = str(path)
    result["_parent_source"] = str((path.parent / raw["extends"]).resolve()) if raw.get("extends") else None
    return result


def path_at_root(value):
    path = Path(value)
    from common.local_paths import local_path
    aliases={"../../comp-90086-2026/house_dataset":"data_root",
             "artifacts/split_v1.csv":"split_path",
             "../../output/experiment_record":"records_root"}
    key=aliases.get(str(value).replace("\\", "/"))
    if key:return local_path(key)
    return path if path.is_absolute() else ROOT / path


def differences(a, b, prefix=""):
    changes = {}
    for key in sorted(set(a) | set(b)):
        if key.startswith("_") or key in {"experiment", "selection"}:
            continue
        name = f"{prefix}.{key}".strip(".")
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            changes.update(differences(x, y, name))
        elif x != y:
            changes[name] = {"before": x, "after": y}
    return changes


def validate(c):
    schema = read_json(ROOT / "configs/base.json")
    def known_keys(value, reference, prefix=""):
        for key, val in value.items():
            path = f"{prefix}.{key}".strip(".")
            if key.startswith("_") or path in {"selection", "run_kind", "experiment", "augmentation"}:
                continue
            if key not in reference:
                raise ValueError(f"未知配置字段（避免拼写错误被忽略）: {path}")
            if isinstance(reference[key], dict):
                if not isinstance(val, dict):
                    raise ValueError(f"配置字段应为对象: {path}")
                known_keys(val, reference[key], path)
    known_keys(c, schema)
    for key in ("id", "name", "stage", "question", "comparison", "changed_fields"):
        if key not in c["experiment"]:
            raise ValueError(f"实验登记缺少 {key}")
    if not re.fullmatch(r"[a-z0-9_\-]+", c["experiment"]["id"]):
        raise ValueError("内部目录编号只能含英文小写、数字、下划线与短横线；中文全名写在 name")
    if c["method"] not in {"mean", "traditional", "neural"}:
        raise ValueError("不支持的方法")
    if c["experiment"].get("status") == "design_required":
        raise ValueError("此项为未登记具体候选的条件实验，请先填写候选及研究依据")
    t = c["training"]
    if t.get("loader_mode", "legacy") not in {"legacy", "persistent"} or t["workers"] < 0:
        raise ValueError("Invalid loader mode or worker count")
    stop = t["early_stopping"]
    if stop["min_epochs"] < 1 or stop["patience"] < 1 or not 0 <= stop["relative_improvement"] < 1:
        raise ValueError("Invalid early stopping settings")
    if t["epochs"] < 1 or t["batch_size"] < 1 or t["accumulation"] < 1:
        raise ValueError("轮次、实际批量、累积次数必须为正")
    if t["lr"] <= 0 or t["weight_decay"] < 0 or t["penalty_coefficient"] < 0 or not 0 <= c["model"]["dropout"] < 1:
        raise ValueError("学习率、惩罚或随机失活范围不合法")
    if c["preprocess"]["size"] < 64:
        raise ValueError("AlexNet 输入边长至少为64")
    if c["model"]["embedding_dim"] % c["model"]["attention_heads"] or c["model"]["encoder_layers"] < 1:
        raise ValueError("编码器维度须整除头数，层数须为正")
    if t["optimizer"] not in {"adamw", "sgd", "adam", "adagrad", "muon"} or t["schedule"] not in {"constant", "cosine"}:
        raise ValueError("优化器或学习率调度不受支持")
    if t["optimizer"] == "muon" and (t["schedule"] != "constant" or c["model"]["head"] != "alexnet"):
        raise ValueError("当前登记的Muon混合方案只支持AlexNet头及固定学习率")
    if "tune_06_decay" in c["experiment"]["id"] and t["optimizer"] != "adamw":
        raise ValueError("解耦衰减候选仅在已选 AdamW 时运行；SGD 使用显式平方惩罚")
    if t["initialization"] not in {"default", "xavier_convolution"}:
        raise ValueError("未知初始化，不能静默回退")
    if c["traditional"]["regressor"] not in {"ridge", "svr"}:
        raise ValueError("未知传统回归器")
    if c["traditional"]["appearance"] not in {"none", "color", "shading", "gradient"}:
        raise ValueError("未知手工特征")
    if t["penalty"] not in {"none", "l1", "l2"}:
        raise ValueError("惩罚仅支持 none/l1/l2")
    if t["penalty"] != "none" and t["weight_decay"] != 0:
        raise ValueError("显式正则化与优化器衰减不能重复启用")
    if c["model"]["head"] not in {"alexnet", "mean", "encoder"}:
        raise ValueError("未知回归头")
    if c["preprocess"]["geometry"] not in {"letterbox", "stretch", "center_crop"}:
        raise ValueError("未知尺寸处理")
    if c["preprocess"]["normalization"] not in {"fixed", "train_stats"}:
        raise ValueError("未知归一化")
    if c["preprocess"]["filter"] not in {"none", "spatial", "frequency"}:
        raise ValueError("未知滤波")
    allowed = {"flip", "brightness", "contrast", "saturation", "rotation", "translation", "crop", "blur", "noise", "erase"}
    if set(c["augmentation"]) - allowed:
        raise ValueError("存在未知增强名称")
    for name, op in c["augmentation"].items():
        if set(op) - {"probability", "range", "std", "aspect"}:
            raise ValueError(f"未知增强参数: {name}")
        if not 0 <= op["probability"] <= 1:
            raise ValueError("增强概率超出范围")
        if name not in {"flip", "noise"} and ("range" not in op or len(op["range"]) != 2 or op["range"][0] > op["range"][1]):
            raise ValueError(f"增强范围无效: {name}")
    stage = c["experiment"]["stage"]
    if stage == "03_augmentation" and "combination" not in c["experiment"]["id"] and len(c["augmentation"]) != 1:
        raise ValueError("单项增强配置必须从无增强方案继承，不能带入其他增强")
    if stage == "02_training" and c["augmentation"]:
        raise ValueError("前期训练调优必须保持无增强")
    if c["model"].get("pretrained", False):
        raise ValueError("本项目禁止预训练权重")
    if c["target"]["transform"] not in {"scale", "log1p"} or c["target"]["scale"] <= 0:
        raise ValueError("目标变换不合法")
    if c["target"]["loss"] not in {"mse", "huber"}:
        raise ValueError("损失不合法")
    if c["_parent_source"] and c["experiment"].get("enforce_changes", False):
        parent = resolve(c["_parent_source"])
        changes = differences(parent, c)
        fields = c["experiment"]["changed_fields"]
        unexpected = [k for k in changes if not any(k == f or k.startswith(f + ".") for f in fields)]
        if unexpected:
            raise ValueError(f"出现未登记改动: {unexpected}")
    return c


def stage_directory(c, kind="development"):
    """One stage owns its readable README, logs and immutable run artifacts."""
    root = path_at_root(c["output_root"])
    if root.name == "runs":  # Accept configurations saved before stage folders existed.
        root = root.parent
    stage = c["experiment"]["stage"]
    if kind == "full_retrain":
        folder = "08_final"
    elif kind in {"smoke", "diagnostic"} or stage == "preparation":
        folder = "00_preparation"
    else:
        folder = {"00_preparation": "01_preprocessing", "01_baselines": "02_baselines",
                  "神经网络基线": "02_baselines", "02_training": "04_training",
                  "03_augmentation": "05_augmentation", "04_architecture": "06_architecture",
                  "05_encoder": "07_encoder"}.get(stage, "08_final")
        if stage == "01_baselines" and c.get("method") == "traditional":
            folder = "03_traditional"
    return root / folder


def new_run(c, kind="development"):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = stage_directory(c, kind) / "runs" / c["experiment"]["id"] / f"{stamp}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", c)
    parent = resolve(c["_parent_source"]) if c.get("_parent_source") else {}
    write_json(out / "config_diff.json", differences(parent, c))
    write_json(out / "status.json", {"status": "running", "kind": kind})
    import importlib.metadata
    versions = {}
    for package in ("torch", "torchvision", "numpy", "pandas", "Pillow", "opencv-python-headless", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = list(ROOT.glob("*.py")) + [p for d in ("common", "preparation", "traditional", "models", "training", "evaluation") for p in (ROOT / d).glob("*.py")]
    import os, sys, torch
    write_json(out / "environment.json", {"python": platform.python_version(), "executable": sys.executable,
                "platform": platform.platform(), "packages": versions, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources}})
    return out


def record_run(c, out):
    """Append completed/failed run evidence to its stage document, including diagnostics."""
    out = Path(out)
    status = read_json(out / "status.json")
    stage = c["experiment"]["stage"]
    root = stage_directory(c, status["kind"])
    root.mkdir(parents=True, exist_ok=True)
    doc = root / "README.md"
    if not doc.exists():
        doc.write_text(f"# {stage} 阶段记录\n\n每次运行自动追加；失败与小样本检查不计入正式性能比较。\n", encoding="utf-8")
    import os
    link = Path(os.path.relpath(out, root)).as_posix()
    lines = [f"\n## {out.name} — {c['experiment']['name']}\n",
             f"- 状态：{status['status']}；用途：{status['kind']}；种子：{c['seed']}。",
             f"- 研究问题：{c['experiment']['question']}",
             f"- 对照：{c['experiment']['comparison']}",
             f"- 登记变量：{', '.join(c['experiment']['changed_fields']) or '无'}。",
             f"- [完整运行目录]({link})；[实际配置]({link}/config.json)；[环境与代码指纹]({link}/environment.json)。"]
    for name in ("metrics.json", "diagnostic.json", "cost.json", "audit.json"):
        if (out / name).exists():
            lines.append(f"- [{name}]({link}/{name})：`{json.dumps(read_json(out / name), ensure_ascii=False)}`")
    if status.get("error"):
        lines.append(f"- 失败原因：{status['error']}。")
    with doc.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
