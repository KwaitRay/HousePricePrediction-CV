"""Preparation acceptance: frozen-data audit and train-only scratch learning check."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from common.config import ROOT, resolve, validate, path_at_root, new_run, write_json, read_json, record_run, digest
from preparation.split import load_split, tables, pixel_hash
from preparation.images import HouseImages, read_rgb, geometry
from training.engine import fit_neural


def audit(c, out, frame, meta):
    root = path_at_root(c["data"]["root"])
    train, test = tables(root, c["data"]["expected_train"], c["data"]["expected_test"])
    def inspect(item):
        part, name = item
        path = root / part / name
        with Image.open(path) as image:
            mode, orientation = image.mode, image.getexif().get(274, 1)
            corrected = ImageOps.exif_transpose(image)
            width, height = corrected.size
        x = geometry(read_rgb(path), c["preprocess"])
        normalized = (x - .5) / .5
        if tuple(x.shape) != (3, 224, 224) or not torch.isfinite(x).all() or x.min() < -1e-6 or x.max() > 1 + 1e-6:
            raise ValueError(f"Invalid preprocessed input: {part}/{name}")
        return {"partition": part, "imageid": name, "width": width, "height": height,
                "mode": mode, "exif_orientation": orientation,
                "pixel_hash": pixel_hash(path), "file_sha256": digest(path),
                "tensor_min": float(normalized.min()), "tensor_max": float(normalized.max())}
    items = [("train", x) for x in train.imageid] + [("test", x) for x in test.imageid]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, items))
    images = pd.DataFrame(rows)
    images.to_csv(out / "image_inventory.csv", index=False, encoding="utf-8-sig")
    duplicate = images[images.duplicated("pixel_hash", keep=False)]
    duplicate.to_csv(out / "exact_duplicates.csv", index=False, encoding="utf-8-sig")
    # Fixed train-side preview only. Validation/test images are not selected for visual diagnosis.
    subset = frame.loc[frame.partition == "train"].sample(32, random_state=c["seed"]).sort_values("imageid")
    ds_train = HouseImages(root, subset, c, training=True)
    ds_eval = HouseImages(root, subset, c)
    sheet = Image.new("RGB", (4 * 224, 8 * 248), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(len(subset)):
        x, price, name, _ = ds_train[i]
        if not torch.equal(x, ds_eval[i][0]) or price != float(subset.iloc[i].price):
            raise ValueError("Train/inference preprocessing or image-label alignment mismatch")
        arr = ((x * .5 + .5).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        left, top = (i % 4) * 224, (i // 4) * 248
        sheet.paste(Image.fromarray(arr), (left, top))
        draw.text((left, top + 225), f"{name} price={price:g}", fill="black")
    sheet.save(out / "preprocessing_preview.png")
    write_json(out / "preview_ids.json", list(subset.imageid))
    # Independent numeric checks: RGB order, exact 0.5 padding, odd remainder on bottom.
    rgb = torch.zeros(3, 21, 40)
    rgb[0] = 1
    padded = geometry(rgb, c["preprocess"])
    nh = round(21 * 224 / 40)
    top = (224 - nh) // 2
    assert torch.equal(padded[:, :top], torch.full_like(padded[:, :top], .5))
    assert torch.equal(padded[:, top + nh:], torch.full_like(padded[:, top + nh:], .5))
    torch.testing.assert_close(padded[:, top:top+nh, :].mean((1, 2)), torch.tensor([1., 0., 0.]))
    summary = {"images_decoded_and_preprocessed": len(rows), "train_rows": len(train), "test_rows": len(test),
               "partitions": frame.partition.value_counts().to_dict(), "split_sha256": meta["split_sha256"],
               "groups": int(frame.group_id.nunique()), "cross_partition_groups": int((frame.groupby("group_id").partition.nunique() > 1).sum()),
               "tail_threshold": meta["tail_threshold"], "validation_tail_n": int(((frame.partition == "validation") & (frame.price > meta["tail_threshold"])).sum()),
               "price_min": float(train.price.min()), "price_max": float(train.price.max()),
               "image_modes": dict(Counter(images['mode'])), "exif_orientations": dict(Counter(map(str, images.exif_orientation))),
               "normalized_min": float(images.tensor_min.min()), "normalized_max": float(images.tensor_max.max()),
               "float32_input_tolerance": 1e-6,
               "train_eval_equal_samples": len(subset), "rgb_and_padding_check": "passed",
               "test_use": "integrity, preprocessing contract and exact duplicate audit only"}
    write_json(out / "audit.json", summary)


def learn(c, out, frame, meta, samples):
    if not torch.cuda.is_available() and c["device"] == "cuda":
        raise RuntimeError("CUDA unavailable; this acceptance run requires GPU")
    train = frame.loc[frame.partition == "train"].sample(samples, random_state=c["seed"]).sort_values("imageid")
    write_json(out / "data_manifest.json", {**meta, "train_ids": list(train.imageid), "validation_ids": [],
               "diagnostic_only": True, "acceptance": "final fixed-train MSE <= 50% initial fixed-train MSE; finite nonzero gradients"})
    train.to_csv(out / "diagnostic_samples.csv", index=False, encoding="utf-8-sig")
    fit_neural(c, path_at_root(c["data"]["root"]), train, train.iloc[:0], meta["tail_threshold"], out, full=True)
    history = pd.read_csv(out / "history.csv")
    initial = read_json(out / "initial_probe.json")["mse"]
    final = float(history.probe_mse.iloc[-1])
    summary = {"samples": len(train), "epochs": len(history), "initial_train_mse": initial,
               "final_train_mse": final, "reduction_fraction": 1-final/initial,
               "finite_nonzero_gradients": bool(np.isfinite(history.gradient_norm).all() and (history.gradient_norm > 0).all()),
               "passed_learning_check": bool(final <= .5 * initial and np.isfinite(history.gradient_norm).all() and (history.gradient_norm > 0).all()),
               "validation_used": False, "test_used": False,
               "note": "Training-only implementation diagnostic; checkpoint prohibited for selection, comparison and submission."}
    write_json(out / "diagnostic.json", summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["audit", "learn"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    c = resolve(ROOT / "configs/01_baselines/alexnet.json")
    c["device"] = args.device
    c["training"].update(epochs=args.epochs, workers=args.workers)
    c["experiment"].update(id=f"preflight_{args.command}_w{args.workers}", stage="preparation",
        name="实验前准备：" + ("全量数据与预处理验收" if args.command == "audit" else "AlexNet 训练侧小样本学习检查"),
        question="冻结数据、输入和梯度是否满足正式实验条件", comparison="设计第3章与初始AlexNet配置；不比较泛化性能",
        changed_fields=["device", "training.epochs", "training.workers"], enforce_changes=False)
    c["run_kind"] = "diagnostic"
    validate(c)
    if not 1 <= args.samples <= 32:
        raise ValueError("学习检查固定训练侧1至32张；不使用验证样本")
    torch.set_num_threads(c["training"]["cpu_threads"])
    out = new_run(c, "diagnostic")
    print(f"Output: {out}", flush=True)
    start = time.perf_counter()
    try:
        frame, meta = load_split(c)
        if args.command == "audit":
            audit(c, out, frame, meta)
        else:
            learn(c, out, frame, meta, args.samples)
        write_json(out / "status.json", {"status": "completed", "kind": "diagnostic"})
    except BaseException as exc:
        write_json(out / "status.json", {"status": "failed", "kind": "diagnostic", "error": str(exc)})
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_json(out / "wall_time.json", {"seconds": time.perf_counter() - start})
        record_run(c, out)


if __name__ == "__main__":
    main()
