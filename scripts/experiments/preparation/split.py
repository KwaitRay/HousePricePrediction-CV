"""Build and verify a frozen duplicate-aware split from the actual images."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import math
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from common.config import digest, path_at_root, write_json, read_json


def tables(root, expected_train=8000, expected_test=3000):
    root = Path(root)
    train = pd.read_csv(root / "train.csv", dtype={"imageid": str})
    test = pd.read_csv(root / "test.csv", dtype={"imageid": str})
    for frame, part, expected in ((train, "train", expected_train), (test, "test", expected_test)):
        required = ["imageid", "price"] if part == "train" else ["imageid"]
        if list(frame.columns) != required or len(frame) != expected:
            raise ValueError(f"{part} 列名或数量不符合登记值")
        if frame.imageid.isna().any() or frame.imageid.duplicated().any():
            raise ValueError("编号缺失或重复")
        for name in frame.imageid:
            if not re.fullmatch(r"\d+\.jpg", name) or not (root / part / name).is_file():
                raise ValueError(f"非法编号或缺图: {part}/{name}")
    train["price"] = pd.to_numeric(train.price, errors="raise")
    if not np.isfinite(train.price).all() or (train.price <= 0).any():
        raise ValueError("训练价格必须为有限正数")
    return train, test


def pixel_hash(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def group_split(train, hashes, near_pairs, seed, fraction):
    parents = {name: name for name in train.imageid}
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    first = {}
    for name, h in zip(train.imageid, hashes):
        if h in first:
            union(name, first[h])
        else:
            first[h] = name
    for a, b in near_pairs:
        if a not in parents or b not in parents:
            raise ValueError(f"人工复核对不在训练清单: {a}, {b}")
        union(a, b)
    frame = train.copy()
    frame["group_id"] = [find(x) for x in frame.imageid]
    frame["pixel_hash"] = hashes
    groups = frame.groupby("group_id").price.median().reset_index().sort_values(["price", "group_id"])
    if len(groups) < 2 or not 0 < fraction < 1:
        raise ValueError("无法建立非空分组划分")
    bins = max(1, min(10, len(groups) // max(2, math.ceil(1 / fraction))))
    groups["stratum"] = np.minimum(np.arange(len(groups)) * bins // len(groups), bins - 1)
    groups["partition"] = "train"
    rng = np.random.default_rng(seed)
    for _, members in groups.groupby("stratum", sort=True):
        n = max(1, min(len(members) - 1, round(len(members) * fraction)))
        groups.loc[rng.choice(members.index, n, replace=False), "partition"] = "validation"
    return frame.merge(groups[["group_id", "partition"]], on="group_id", validate="many_to_one")


def prepare(c):
    root = path_at_root(c["data"]["root"])
    dest = path_at_root(c["data"]["split"])
    if dest.exists() or dest.with_suffix(".json").exists():
        raise FileExistsError("冻结划分已存在；修改分组时请使用新的输出文件名")
    train, test = tables(root, c["data"]["expected_train"], c["data"]["expected_test"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(pixel_hash, [root / "train" / x for x in train.imageid]))
        # Decode test images for integrity only, never use them to group or fit.
        list(pool.map(pixel_hash, [root / "test" / x for x in test.imageid]))
    frame = group_split(train, hashes, c["data"]["near_pairs"], c["data"]["split_seed"], c["data"]["validation_fraction"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False, encoding="utf-8-sig")
    meta = {"train_csv_sha256": digest(root / "train.csv"), "test_csv_sha256": digest(root / "test.csv"),
            "split_sha256": digest(dest), "tail_threshold": float(frame.loc[frame.partition == "train", "price"].quantile(.9)),
            "seed": c["data"]["split_seed"], "near_pairs": c["data"]["near_pairs"], "n": len(frame)}
    write_json(dest.with_suffix(".json"), meta)
    return dest


def load_split(c):
    root, path = path_at_root(c["data"]["root"]), path_at_root(c["data"]["split"])
    train, _ = tables(root, c["data"]["expected_train"], c["data"]["expected_test"])
    meta = read_json(path.with_suffix(".json"))
    if digest(root / "test.csv") != meta["test_csv_sha256"]:
        raise ValueError("测试清单内容变化，请另建数据版本")
    if c["data"]["split_seed"] != meta["seed"] or c["data"]["near_pairs"] != meta["near_pairs"]:
        raise ValueError("当前划分种子或近重复分组规则与冻结记录不同")
    if digest(path) != meta["split_sha256"] or digest(root / "train.csv") != meta["train_csv_sha256"]:
        raise ValueError("冻结划分或训练标签内容变化，请另建版本")
    frame = pd.read_csv(path, dtype={"imageid": str, "group_id": str})
    if frame.imageid.duplicated().any() or set(frame.imageid) != set(train.imageid):
        raise ValueError("划分未唯一覆盖训练清单")
    check = frame.set_index("imageid").price.reindex(train.imageid).to_numpy()
    if not np.array_equal(check, train.price.to_numpy()):
        raise ValueError("划分标签与训练标签不同")
    if set(frame.partition) != {"train", "validation"} or frame.groupby("group_id").partition.nunique().max() != 1:
        raise ValueError("划分存在组泄漏或分区不合法")
    if frame.groupby("pixel_hash").partition.nunique().max() != 1:
        raise ValueError("相同像素内容跨越训练与验证")
    # Recheck content, not just filenames or CSV: changed images need a new split version.
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(pixel_hash, [root / "train" / name for name in frame.imageid]))
    if actual != list(frame.pixel_hash):
        raise ValueError("图片内容已改变，请审计并生成新的划分版本")
    rebuilt = group_split(train, frame.set_index("imageid").pixel_hash.reindex(train.imageid).tolist(),
                          c["data"]["near_pairs"], c["data"]["split_seed"], c["data"]["validation_fraction"])
    columns = ["group_id", "partition"]
    if not rebuilt.set_index("imageid")[columns].sort_index().equals(frame.set_index("imageid")[columns].sort_index()):
        raise ValueError("冻结划分与当前分组规则、验证比例或划分算法不一致")
    threshold = float(frame.loc[frame.partition == "train", "price"].quantile(.9))
    if meta["n"] != len(frame) or meta["tail_threshold"] != threshold:
        raise ValueError("冻结元信息的数量或训练侧尾部阈值不一致")
    return frame, meta
