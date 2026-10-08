"""Export an explicit code bundle without internal plans, reports or local caches.

Uses only the standard library. Never trains, uploads, deletes or overwrites.
The report PDF is submitted separately; this command packages code only.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path("scripts/experiments")
MODULES = ("common", "preparation", "traditional", "models", "training", "evaluation")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def config_files(root, configs):
    """Include only requested configurations and their inheritance dependencies."""
    base = (root / EXPERIMENTS / "configs").resolve()
    selected = set()

    def visit(path, stack=()):
        path = path.resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Configuration must stay within configs/: {path}")
        if path in stack:
            raise ValueError("Configuration inheritance cycle")
        if path.suffix != ".json" or not path.is_file():
            raise ValueError(f"Missing JSON configuration: {path}")
        if path in selected:
            return
        config = read_json(path)
        if config.get("extends"):
            visit(path.parent / config["extends"], (*stack, path))
        selected.add(path)

    # Validation uses this schema even when the requested config is self-contained.
    visit(base / "base.json")
    for name in configs:
        path = Path(name)
        if not path.is_absolute():
            path = root / EXPERIMENTS / path
        visit(path)
    return sorted(selected)


def portable_config_names(root, configs):
    """Resolve inputs once, then refer only to paths present inside the bundle."""
    experiments = (root / EXPERIMENTS).resolve()
    names = []
    for name in configs:
        path = Path(name)
        path = path.resolve() if path.is_absolute() else (experiments / path).resolve()
        if not path.is_relative_to(experiments / "configs"):
            raise ValueError(f"Configuration must stay within configs/: {path}")
        names.append(path.relative_to(experiments).as_posix())
    return names


def collect(root, configs):
    """Use a whitelist: do not recursively copy the whole repository."""
    root = root.resolve()
    paths = [root / "project.py", root / "local_paths.example.json"]
    for name in ("run.py", "preflight.py", "train_cached.py", "requirements.txt",
                 "requirements-traditional.txt", "requirements-traditional-lock.txt"):
        path = root / EXPERIMENTS / name
        if path.is_file():
            paths.append(path)
    # Keep the analyze-data CLI functional, without copying its historical outputs.
    for name in ("analyze_dataset.py", "analyze_traditional.py", "requirements.txt"):
        path = root / "scripts" / name
        if path.is_file():
            paths.append(path)
    for module in MODULES:
        paths.extend(sorted((root / EXPERIMENTS / module).glob("*.py")))
    paths.extend(config_files(root, configs))
    files = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or path.is_symlink():
            raise ValueError(f"External source or symlink not allowed: {path}")
        name = path.relative_to(root).as_posix()
        content = path.read_bytes()
        if path.suffix == ".json" and "configs" in path.parts:
            config = read_json(path)
            # A selected run path is provenance, not a portable execution input.
            if isinstance(config.get("selection"), dict):
                config["selection"].pop("run", None)
            for key in list(config):
                if key.startswith("_"):
                    config.pop(key)
            content = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
        files[name] = content
    return files


def predictions_bytes(path):
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if reader.fieldnames != ["imageid", "price"]:
        raise ValueError("Predictions must have exactly imageid,price columns")
    rows = list(reader)
    if len(rows) != 3000 or len({r["imageid"] for r in rows}) != 3000:
        raise ValueError("Expected 3000 unique test predictions")
    for row in rows:
        price = float(row["price"])
        if not row["imageid"] or not math.isfinite(price) or price < 0:
            raise ValueError("Invalid image ID or predicted price")
    return content


def make_bundle(root, configs, split=None, predictions=None, final=False):
    if final and (split is None or predictions is None):
        raise ValueError("Final export requires --split and --predictions; otherwise use preview mode")
    files = collect(root, configs)
    configs = portable_config_names(root, configs)
    if split is not None:
        meta = split.with_suffix(".json")
        # Preserve the exact frozen split pair instead of creating a new split.
        files[".local/artifacts/split_v1.csv"] = split.read_bytes()
        files[".local/artifacts/split_v1.json"] = meta.read_bytes()
    if predictions is not None:
        files["predictions.csv"] = predictions_bytes(predictions)
    commands = "\n".join(
        f"python project.py train {json.dumps(name, ensure_ascii=False)} --seed {seed}"
        for name in configs for seed in (2026, 2027, 2028))
    guide = f"""HOUSE PRICE PREDICTION - CODE BUNDLE
Export mode: {'FINAL CANDIDATE (requires human review)' if final else 'DEVELOPMENT PREVIEW - NOT A FINAL SUBMISSION'}

1. Python 3.11. For neural models install:
   python -m pip install -r scripts/experiments/requirements.txt
   For traditional CPU models only:
   python -m pip install -r scripts/experiments/requirements-traditional-lock.txt
2. Copy local_paths.example.json to local_paths.json. Set data_root to the
   supplied dataset directory containing train/, test/, train.csv and test.csv.
   Provided images are NOT included in this bundle.
3. split_path must point to the team's confirmed frozen split_v1.csv and its
   same-named .json file. Do not regenerate a split for historical comparisons.
   Frozen split included: {split is not None}.
4. Inspect paths: python project.py paths
5. Reproduce the explicitly included configurations (three separate runs):
{commands}
6. Outputs and models are created under .local/experiment_record by default.
   For the chosen final configuration, retrain for the epoch count selected
   from development results, then export Kaggle predictions:
   python project.py retrain <config> --epochs <chosen_count> --seed <chosen_seed>
   python project.py predict <completed_full_retrain_directory> --output predictions.csv
   Test prediction prices use $1000 USD. Predictions included: {predictions is not None}.

The formal report PDF is submitted separately. Internal design/, archive/,
experiment reports, development outputs, local paths, caches and environments
are intentionally excluded. Only requested JSON configs and their parents
are copied. This exporter does not verify scientific completeness or approve
the final model; both teammates must review the final candidate.
"""
    files["README.txt"] = guide.encode("utf-8")
    manifest = {
        "mode": "final_candidate" if final else "preview",
        "configs": configs,
        "predictions_included": predictions is not None,
        "frozen_split_included": split is not None,
        "sha256": {name: hashlib.sha256(data).hexdigest()
                   for name, data in sorted(files.items())},
    }
    files["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    return files


def export_zip(files, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting any existing submission.
    with destination.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise ValueError("ZIP verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True,
                        help="Path relative to scripts/experiments, e.g. configs/01_baselines/sift_ridge.json")
    parser.add_argument("--split", type=Path, help="Confirmed frozen split CSV; same-named JSON required")
    parser.add_argument("--predictions", type=Path, help="Final model's Kaggle-format CSV")
    parser.add_argument("--final", action="store_true", help="Require split and predictions; still needs human review")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files = make_bundle(ROOT, args.config, args.split, args.predictions, args.final)
    for name in sorted(files):
        print(name)
    if not args.dry_run:
        export_zip(files, args.output.resolve())
        print(f"Verified {len(files)} files: {args.output.resolve()}")


if __name__ == "__main__":
    main()
