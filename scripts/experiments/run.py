"""Shared entry point; run --help for preparation, experiments, selection and export."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
from common.config import ROOT, resolve, validate, path_at_root, read_json, write_json, new_run, digest, record_run
from common.metrics import save_predictions
from preparation.split import prepare, load_split, tables


def experiment(config_path, seed=None, smoke=False, full_epochs=None):
    c = validate(resolve(config_path))
    if seed is not None:
        c["seed"] = seed
    full = full_epochs is not None
    if full and smoke:
        raise ValueError("全量重训与小样本检查不能混用")
    frame, meta = load_split(c)
    train = frame if full else frame.loc[frame.partition == "train"]
    val = frame.iloc[:0] if full else frame.loc[frame.partition == "validation"]
    c["run_kind"] = "full_retrain" if full else "smoke" if smoke else "development"
    if full:
        if full_epochs < 1:
            raise ValueError("全量训练轮次必须为正，按开发结果预先确定")
        c["training"]["epochs"] = full_epochs
        c["training"]["early_stopping"]["enabled"] = False
    if smoke:
        train, val = train.iloc[:8], val.iloc[:4]
        c["training"].update(epochs=1, workers=0)
        c["traditional"].update(words=8, max_descriptors=2048)
    out = new_run(c, c["run_kind"])
    print(f"Output: {out}", flush=True)
    try:
        write_json(out / "data_manifest.json", {**meta, "train_ids": list(train.imageid), "validation_ids": list(val.imageid)})
        root = path_at_root(c["data"]["root"])
        if c["method"] == "mean":
            start = time.perf_counter()
            value = float(train.price.mean())
            write_json(out / "mean.json", {"price": value})
            raw = np.full(len(val), value)
            write_json(out / "cost.json", {"fit_and_predict_seconds": time.perf_counter()-start, "timing_note": "均值计算、保存和生成预测；无需神经网络训练"})
        elif c["method"] == "traditional":
            from traditional.pipeline import fit_traditional
            raw = fit_traditional(c, root, train, val, out)
        else:
            from training.engine import fit_neural
            raw = fit_neural(c, root, train, val, meta["tail_threshold"], out, full)
        if not full:
            metrics = save_predictions(out, val.reset_index(drop=True), raw, meta["tail_threshold"])
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
        write_json(out / "status.json", {"status": "completed", "kind": c["run_kind"]})
        return out
    except BaseException as exc:
        write_json(out / "status.json", {"status": "failed", "kind": c["run_kind"], "error": str(exc)})
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        record_run(c, out)


def select_run(run, slot, reason, replace=False):
    import re
    if not re.fullmatch(r"[a-z_]+", slot):
        raise ValueError("选择文件名只能为小写英文和下划线")
    run = Path(run).resolve()
    status = read_json(run / "status.json")
    if status != {"status": "completed", "kind": "development"}:
        raise ValueError("只能登记成功的正式开发实验，不能登记小样本或全量训练结果")
    metrics = read_json(run / "metrics.json")
    c = read_json(run / "config.json")
    if slot not in {"input", "optimizer", "schedule", "batch", "dropout", "regularization", "training", "augmentation", "small", "multi", "convolution", "head", "encoder", "final"}:
        raise ValueError("未知阶段选择名称，请查看 README 的阶段衔接表")
    if slot != "final" and c["method"] != "neural":
        raise ValueError("神经网络阶段不能继承常数或传统回归结果")
    if slot in {"input", "optimizer", "schedule", "batch", "dropout", "regularization", "training"}:
        if c["augmentation"] or c["model"]["head"] != "alexnet" or any(c["model"][k] for k in ("small_kernels", "inception", "residual")):
            raise ValueError("前期阶段须保持原始无增强 AlexNet 结构")
    if slot == "head" and c["model"]["head"] != "mean":
        raise ValueError("编码器的直接对照必须是小型平均聚合头，即使它弱于原卷积模型也要单独保留")
    if slot == "encoder" and (c["model"]["head"] != "encoder" or c["model"]["encoder_layers"] != 1):
        raise ValueError("增加编码器层数前须选择单层编码器结果")
    target = ROOT / "configs" / "selected" / f"{slot}.json"
    if target.exists() and not replace:
        raise FileExistsError("选定记录已存在；确认更换选择时加 --replace，历史运行仍保留")
    c = {k: v for k, v in c.items() if not k.startswith("_") and k not in {"run_kind", "selection"}}
    c["experiment"]["enforce_changes"] = False
    c["selection"] = {"run": str(run), "reason": reason, "metrics": metrics,
                      "warning": "人工阶段选择记录；单次运行不代表已完成多种子复核"}
    write_json(target, c)
    return target


def submission(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if read_json(run / "status.json") != {"status": "completed", "kind": "full_retrain"}:
        raise ValueError("提交预测只接受完整全量重训产物")
    c = read_json(run / "config.json")
    root = path_at_root(c["data"]["root"])
    _, test = tables(root, c["data"]["expected_train"], c["data"]["expected_test"])
    manifest = read_json(run / "data_manifest.json")
    if digest(root / "train.csv") != manifest["train_csv_sha256"] or digest(root / "test.csv") != manifest["test_csv_sha256"]:
        raise ValueError("训练或测试清单发生变化")
    start = time.perf_counter()
    if c["method"] == "mean":
        pred = np.full(len(test), read_json(run / "mean.json")["price"])
    elif c["method"] == "traditional":
        import joblib
        model = joblib.load(run / "traditional.joblib")  # Only load artifacts created by this project.
        pred = model.predict(root, test, "test")
    else:
        import torch
        from models.alexnet import AlexNetRegressor
        from preparation.images import HouseImages
        from training.engine import predict
        device = torch.device("cuda" if c["device"] == "auto" and torch.cuda.is_available() else "cpu" if c["device"] == "auto" else c["device"])
        saved = torch.load(run / "best.pt", map_location=device, weights_only=True)
        model = AlexNetRegressor(c["model"]).to(device)
        model.load_state_dict(saved["model"])
        ds = HouseImages(root, test, c, saved["stats"], partition="test")
        pred = predict(model, ds, c, device)
    if len(pred) != len(test) or not np.isfinite(pred).all():
        raise ValueError("预测数量或数值不合法")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"imageid": test.imageid, "price": np.maximum(pred, 0)}).to_csv(output, index=False)
    write_json(output.with_suffix(".json"), {"source_run": str(run.resolve()), "n": len(test),
               "unit": "thousand_USD", "predict_seconds_including_load": time.perf_counter() - start,
               "submission_sha256": digest(output)})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="按阶段配置执行房价实验，不自动运行全部候选")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "preview", "train", "retrain"):
        cmd = sub.add_parser(name)
        cmd.add_argument("config", type=Path)
        if name in {"train", "retrain"}:
            cmd.add_argument("--seed", type=int)
        if name == "train":
            cmd.add_argument("--smoke", action="store_true", help="仅做8张训练、4张验证、1轮的检查，禁止用于正式比较")
        if name == "retrain":
            cmd.add_argument("--epochs", type=int, required=True, help="按开发结果确定，传统方法可填1")
        if name == "preview":
            cmd.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select")
    select.add_argument("run", type=Path)
    select.add_argument("--slot", required=True)
    select.add_argument("--reason", required=True)
    select.add_argument("--replace", action="store_true")
    comp = sub.add_parser("compare")
    comp.add_argument("runs", nargs="+", type=Path)
    comp.add_argument("--output", required=True, type=Path)
    comp.add_argument("--paired", action="store_true")
    pred = sub.add_parser("predict")
    pred.add_argument("run", type=Path)
    pred.add_argument("--output", type=Path, required=True)
    sub.add_parser("catalog")
    args = parser.parse_args(argv)
    if args.command == "catalog":
        for path in sorted((ROOT / "configs").glob("*/*.json")):
            if path.parent.name == "selected":
                continue
            raw = read_json(path)
            try:
                validate(resolve(path))
                state = "ready"
            except (ValueError, FileNotFoundError) as exc:
                state = "pending: " + str(exc)
            print(str(path.relative_to(ROOT)), raw["experiment"]["name"], state)
    elif args.command == "check":
        c = validate(resolve(args.config))
        print(json.dumps(c, ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        print(prepare(validate(resolve(args.config))))
    elif args.command == "preview":
        from preparation.images import preview
        c = validate(resolve(args.config))
        frame, _ = load_split(c)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        logs = preview(path_at_root(c["data"]["root"]), frame.loc[frame.partition == "train"], c, args.output)
        write_json(args.output.with_suffix(".json"), logs)
        print(args.output)
    elif args.command in {"train", "retrain"}:
        experiment(args.config, args.seed, getattr(args, "smoke", False), getattr(args, "epochs", None))
    elif args.command == "select":
        print(select_run(args.run, args.slot, args.reason, args.replace))
    elif args.command == "compare":
        from evaluation.report import compare_runs
        print(compare_runs(args.runs, args.output, args.paired))
    elif args.command == "predict":
        print(submission(args.run, args.output))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
