"""Contract tests: leakage, augmentation RNG, module interfaces and end-to-end export."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import resolve, validate, write_json, read_json, differences, stage_directory
from common.metrics import evaluate
from preparation.split import group_split, prepare, load_split
from preparation.images import augment, geometry, HouseImages, filtering
from models.alexnet import AlexNetRegressor, initialize, DeterministicAdaptivePool
from training.engine import EarlyStop, train_epoch
from traditional.pipeline import appearance, TraditionalModel
from run import experiment, submission, select_run
from evaluation.report import compare_runs


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        return resolve(ROOT / "configs/base.json")

    def test_stage_output_routing(self):
        c = self.config()
        c["output_root"] = str(ROOT / "test_records")
        for stage, folder in [("00_preparation", "01_preprocessing"), ("01_baselines", "02_baselines"),
                              ("02_training", "04_training"), ("03_augmentation", "05_augmentation"),
                              ("04_architecture", "06_architecture"), ("05_encoder", "07_encoder")]:
            c["experiment"]["stage"] = stage
            self.assertEqual(stage_directory(c), ROOT / "test_records" / folder)
            self.assertEqual(stage_directory(c, "diagnostic").name, "00_preparation")
            self.assertEqual(stage_directory(c, "full_retrain").name, "08_final")
        c["output_root"] = str(ROOT / "test_records/runs")
        self.assertEqual(stage_directory(c, "smoke"), ROOT / "test_records/00_preparation")

    def test_metrics_units_clipping_tail(self):
        m = evaluate([100, 200, 500], [-10, 210, 450], 400)
        self.assertAlmostEqual(m["mse"], (10000+100+2500)/3)
        self.assertAlmostEqual(m["mae"], 160/3)
        self.assertEqual(m["tail_mse"], 2500)
        self.assertIsNone(evaluate([1], [1], 10)["tail_mse"])
        with self.assertRaises(ValueError):evaluate([1], [float("nan")], 0)

    def test_deterministic_adaptive_pool_matches_cpu_reference(self):
        torch.use_deterministic_algorithms(True)
        for h, w in ((6, 6), (9, 9), (12, 12), (2, 3)):
            reference = torch.randn(2, 3, h, w, requires_grad=True)
            y = torch.nn.functional.adaptive_avg_pool2d(reference, (6, 6))
            y.square().sum().backward()
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                x = reference.detach().to(device).requires_grad_(True)
                actual = DeterministicAdaptivePool((6, 6))(x)
                actual.square().sum().backward()
                torch.testing.assert_close(actual.cpu(), y)
                torch.testing.assert_close(x.grad.cpu(), reference.grad)

    def test_transitive_duplicate_groups(self):
        f = pd.DataFrame({"imageid": [f"{i}.jpg" for i in range(8)], "price": np.arange(8)+100.})
        split = group_split(f, ["same", "same", "other", "x", "y", "z", "a", "b"], [["1.jpg", "2.jpg"]], 2026, .2)
        self.assertEqual(split.iloc[:3].group_id.nunique(), 1)
        self.assertEqual(split.groupby("group_id").partition.nunique().max(), 1)
        self.assertEqual(set(split.partition), {"train", "validation"})

    def test_rng_isolation_and_validation(self):
        x = torch.linspace(0, 1, 3*40*50).reshape(3, 40, 50)
        noise = {"noise": {"probability": 1., "std": .01}}
        a, e1 = augment(x, noise, 1, 2, "1.jpg")
        b, e2 = augment(x, {"flip": {"probability": 0.}, **noise}, 1, 2, "1.jpg")
        torch.testing.assert_close(a, b)
        c, _ = augment(x, noise, 1, 3, "1.jpg")
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(e1["noise"], e2["noise"])
        cfg = self.config()["preprocess"]
        out = geometry(torch.ones(3, 20, 40), cfg)
        self.assertEqual(tuple(out.shape), (3, 224, 224))
        self.assertEqual(float(out[:, 0].mean()), .5)
        for mode in ("spatial", "frequency"):
            z = filtering(x, mode)
            self.assertEqual(z.shape, x.shape)
            self.assertTrue(torch.isfinite(z).all())

    def test_early_stopping_does_not_reset_on_tiny_gain(self):
        stopper = EarlyStop({"enabled": True, "relative_improvement": .001, "patience": 2, "min_epochs": 3})
        self.assertFalse(stopper.update(100., 1))
        self.assertFalse(stopper.update(99.99, 2))
        self.assertTrue(stopper.update(99.98, 3))

    def test_architecture_interfaces_and_shared_initialization(self):
        cfg = self.config()["model"]
        cfg["head"] = "mean"
        model = AlexNetRegressor(cfg)
        initialize(model, 2026, .5)
        weight = model.conv1.weight.detach().clone()
        for kind in ("small_kernels", "inception", "residual", "encoder"):
            other = deepcopy(cfg)
            if kind == "encoder":other["head"] = "encoder"
            else:other[kind] = True
            candidate = AlexNetRegressor(other)
            initialize(candidate, 2026, .5)
            torch.testing.assert_close(weight, candidate.conv1.weight)
            candidate.eval()
            with torch.no_grad():pred = candidate(torch.randn(2, 3, 224, 224))
            self.assertEqual(tuple(pred.shape), (2,))
            self.assertTrue(torch.isfinite(pred).all())
        model = AlexNetRegressor(self.config()["model"])
        self.assertEqual(model.output.in_features, 4096)
        with torch.no_grad():self.assertEqual(tuple(model.eval()(torch.zeros(1, 3, 224, 224)).shape), (1,))

    def test_accumulation_matches_actual_batch_with_regularization(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(1, 1)
            def forward(self, x):return self.fc(x).squeeze(-1)
        class Data(torch.utils.data.Dataset):
            epoch = 1
            def __len__(self):return 5
            def __getitem__(self, i):return torch.tensor([float(i)]), float(i*2+1), str(i)
        c = self.config()
        c["target"]["scale"] = 1
        c["training"].update(batch_size=2, accumulation=3, workers=0, penalty="l2", penalty_coefficient=.01)
        model = Net()
        other = deepcopy(model)
        train_epoch(model, Data(), c, torch.optim.SGD(model.parameters(), lr=.01), torch.device("cpu"))
        c["training"].update(batch_size=5, accumulation=1)
        train_epoch(other, Data(), c, torch.optim.SGD(other.parameters(), lr=.01), torch.device("cpu"))
        for a, b in zip(model.parameters(), other.parameters()):torch.testing.assert_close(a, b)

    def test_appearance_dimensions(self):
        rgb = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        self.assertEqual(len(appearance(rgb, "color")), 30)
        self.assertEqual(len(appearance(rgb, "shading")), 20)
        self.assertEqual(len(appearance(rgb, "gradient")), 45)

    def test_all_augmentations_and_encoder_gradients(self):
        settings = {}
        for path in (ROOT / "configs/03_augmentation").glob("augment_*.json"):
            if "combination" in path.name:continue
            settings.update(read_json(path)["augmentation"])
        for op in settings.values():op["probability"] = 1.
        image = torch.rand(3, 80, 100)
        transformed, events = augment(image, settings, 2026, 1, "x")
        self.assertEqual(len(events), 10)
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertGreaterEqual(float(transformed.min()), 0.)
        self.assertLessEqual(float(transformed.max()), 1.)
        cfg = self.config()["model"]
        cfg.update(head="encoder", encoder_layers=2)
        model = AlexNetRegressor(cfg)
        initialize(model, 2026, .5)
        model(torch.rand(2, 3, 224, 224)).square().mean().backward()
        self.assertIsNotNone(model.regressor.position.grad)
        for layer in model.regressor.layers:
            self.assertTrue(torch.isfinite(layer.self_attn.in_proj_weight.grad).all())
        self.assertFalse(torch.equal(model.regressor.layers[0].self_attn.in_proj_weight,
                                     model.regressor.layers[1].self_attn.in_proj_weight))

    def test_config_rejects_typos_and_duplicate_penalties(self):
        c = self.config()
        c["training"]["learning_rate"] = .001
        with self.assertRaises(ValueError):validate(c)

    def test_comparison_guards_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "control", root / "candidate"]
            for i, path in enumerate(paths):
                cfg = self.config()
                cfg["experiment"].update(name=f"model{i}", changed_fields=["model.residual"])
                cfg["model"]["residual"] = bool(i)
                write_json(path / "config.json", cfg)
                write_json(path / "status.json", {"status": "completed", "kind": "development"})
                write_json(path / "data_manifest.json", {"split_sha256": "same", "tail_threshold": 10})
                write_json(path / "metrics.json", {"mse": 4-i, "mae": 2., "tail_mse": None, "n": 2, "tail_n": 0})
                pd.DataFrame({"imageid": ["1.jpg", "2.jpg"], "price": [1., 2.], "group_id": ["a", "b"], "squared_error": [4-i, 4-i]}).to_csv(path / "predictions.csv", index=False)
            compare_runs(paths, root / "ok", paired=True)
            self.assertEqual(read_json(root / "ok/paired_bootstrap.json")["interval_95"], [-1., -1.])
            write_json(paths[0] / "environment.json", {"source_hashes": {"run.py": "old"}})
            write_json(paths[1] / "environment.json", {"source_hashes": {"run.py": "new"}})
            with self.assertRaises(ValueError):compare_runs(paths, root / "changed_code", paired=True)
            write_json(paths[1] / "status.json", {"status": "completed", "kind": "smoke"})
            with self.assertRaises(ValueError):compare_runs(paths, root / "bad")
            write_json(paths[1] / "status.json", {"status": "completed", "kind": "diagnostic"})
            with self.assertRaises(ValueError):compare_runs(paths, root / "diagnostic")
            with self.assertRaises(ValueError):select_run(paths[1], "input", "must reject diagnostic")
        c = self.config()
        c["training"]["penalty"] = "l1"
        with self.assertRaises(ValueError):validate(c)

    def test_all_catalog_configs_with_explicit_mock_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "configs"
            shutil.copytree(ROOT / "configs", dst)
            base = self.config()
            for slot in ("input", "optimizer", "schedule", "batch", "dropout", "regularization", "training", "augmentation", "small", "multi", "convolution", "head", "encoder"):
                selected = deepcopy(base)
                if slot in {"head", "encoder"}:selected["model"]["head"] = "mean" if slot == "head" else "encoder"
                write_json(dst / "selected" / f"{slot}.json", selected)
            n = 0
            for path in dst.glob("*/*.json"):
                if path.parent.name == "selected":continue
                c = resolve(path)
                if c["experiment"]["status"] == "design_required":
                    with self.assertRaises(ValueError):validate(c)
                else:validate(c)
                n += 1
            self.assertGreaterEqual(n, 56)

    def test_end_to_end_neural_traditional_and_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in ("train", "test"):(root / part).mkdir()
            rng = np.random.default_rng(5)
            for i in range(18):
                Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)).save(root / ("train" if i < 16 else "test") / f"{i}.jpg")
            pd.DataFrame({"imageid": [f"{i}.jpg" for i in range(16)], "price": np.arange(16)*10+100.}).to_csv(root / "train.csv", index=False)
            pd.DataFrame({"imageid": ["16.jpg", "17.jpg"]}).to_csv(root / "test.csv", index=False)
            c = self.config()
            c["data"].update(root=str(root), split=str(root / "split.csv"), expected_train=16, expected_test=2, near_pairs=[])
            c["output_root"] = str(root / "results")
            c["training"].update(epochs=1, workers=0, cpu_threads=2)
            c["model"]["head"] = "mean"
            c["traditional"].update(words=4, max_keypoints=32, max_descriptors=256)
            path = root / "config.json"
            write_json(path, c)
            prepare(c)
            frame, meta = load_split(c)
            self.assertAlmostEqual(meta["tail_threshold"], frame.loc[frame.partition == "train", "price"].quantile(.9))
            changed = deepcopy(c)
            changed["data"]["split_seed"] += 1
            with self.assertRaises(ValueError):load_split(changed)
            changed = deepcopy(c)
            changed["data"]["near_pairs"] = [["0.jpg", "1.jpg"]]
            with self.assertRaises(ValueError):load_split(changed)
            changed = deepcopy(c)
            changed["data"]["validation_fraction"] = .4
            with self.assertRaises(ValueError):load_split(changed)
            altered_meta = deepcopy(meta)
            altered_meta["tail_threshold"] += 1
            write_json(root / "split.json", altered_meta)
            with self.assertRaises(ValueError):load_split(c)
            write_json(root / "split.json", meta)
            original_test = (root / "test.csv").read_bytes()
            pd.DataFrame({"imageid": ["17.jpg", "16.jpg"]}).to_csv(root / "test.csv", index=False)
            with self.assertRaises(ValueError):load_split(c)
            (root / "test.csv").write_bytes(original_test)
            for method in ("mean", "traditional", "neural"):
                c["method"] = method
                write_json(path, c)
                run = experiment(path)
                self.assertTrue(np.isfinite(read_json(run / "metrics.json")["mse"]))
                full = experiment(path, full_epochs=1)
                output = root / f"{method}.csv"
                submission(full, output)
                result = pd.read_csv(output)
                self.assertEqual(list(result.imageid), ["16.jpg", "17.jpg"])
                self.assertTrue(np.isfinite(result.price).all())
            c["method"] = "mean"
            write_json(path, c)
            smoke = experiment(path, smoke=True)
            self.assertTrue((root / "results/00_preparation/README.md").exists())
            with self.assertRaises(ValueError):select_run(smoke, "test_reject", "must reject")
            with self.assertRaises(ValueError):submission(smoke, root / "invalid.csv")


if __name__ == "__main__":
    unittest.main()
