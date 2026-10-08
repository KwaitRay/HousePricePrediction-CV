"""CPU-only stage-03 checks; no neural dependencies and no course test labels."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import importlib.util
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import new_run, resolve, stage_directory, write_json
from common.metrics import evaluate, inverse_target, transform_target
from traditional.pipeline import TraditionalModel, extract, histogram

spec = importlib.util.spec_from_file_location("analyze_traditional", ROOT.parent / "analyze_traditional.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class TraditionalChecks(unittest.TestCase):
    def config(self):
        return resolve(ROOT / "configs/01_baselines/sift_ridge.json")

    def test_histogram_and_empty_descriptor_image(self):
        np.testing.assert_allclose(histogram(np.array([0, 0, 1]), 3),
                                   np.sqrt([2/3, 1/3, 0]))
        np.testing.assert_array_equal(histogram(np.array([], dtype=int), 3), np.zeros(3))
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "blank.png"
            Image.new("RGB", (120, 90), (128, 128, 128)).save(image)
            desc, coords, extra = extract(image, self.config()["traditional"])
            self.assertEqual(desc.shape, (0, 128))
            self.assertEqual(coords.shape, (0, 2))
            self.assertEqual(extra.shape, (0,))

    def test_cache_equivalence_image_invalidation_and_seed_reuse(self):
        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            rng = np.random.default_rng(5)
            pixels = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
            image = root / "train/1.jpg"
            Image.fromarray(pixels).save(image)
            cfg = self.config()
            with patch.dict(os.environ, {"CV_CACHE_ROOT": str(root / "cache")}):
                model = TraditionalModel(cfg["traditional"], cfg["target"], 2026)
                fresh = model.records(root, ["1.jpg"])[0]
                self.assertEqual(model.last_extraction_cache_["misses"], 1)
                other = TraditionalModel(cfg["traditional"], cfg["target"], 2027)
                cached = other.records(root, ["1.jpg"])[0]
                self.assertEqual(other.last_extraction_cache_["hits"], 1)
                for a, b in zip(fresh, cached):
                    np.testing.assert_array_equal(a, b)
                self.assertLessEqual(len(fresh[0]), 256)
                self.assertGreater(len(fresh[0]), 0)
                np.testing.assert_allclose(np.linalg.norm(fresh[0], axis=1), 1, atol=2e-6)
                self.assertTrue(((fresh[1] >= 0) & (fresh[1] < 1)).all())
                Image.fromarray(255 - pixels).save(image)
                other.records(root, ["1.jpg"])
                self.assertEqual(other.last_extraction_cache_["misses"], 1)
                changed = deepcopy(cfg["traditional"])
                changed["long_edge"] = 160
                different = TraditionalModel(changed, cfg["target"], 2027)
                different.records(root, ["1.jpg"])
                self.assertEqual(different.last_extraction_cache_["misses"], 1)

    def test_zero_records_feature_shape_and_pyramid(self):
        cfg = self.config()
        record = (np.empty((0, 128)), np.empty((0, 2)), np.empty(0))
        model = TraditionalModel(cfg["traditional"], cfg["target"], 2026)
        np.testing.assert_array_equal(model.features([record]), np.zeros((1, 256)))
        model.cfg["spatial_pyramid"] = True
        np.testing.assert_array_equal(model.features([record]), np.zeros((1, 1280)))

    def test_target_units_and_validation_clipping(self):
        target = self.config()["target"]
        np.testing.assert_allclose(inverse_target(transform_target(np.array([100, 1300]), target), target), [100, 1300])
        score = evaluate([100, 2000], [-100, 1900], 1300)
        self.assertEqual(score["mse"], 10000)
        self.assertEqual(score["tail_n"], 1)

    def test_only_training_records_fit_dictionary_scaler_and_regressor(self):
        rng = np.random.default_rng(2026)
        cfg = self.config()
        cfg["traditional"].update(words=4, max_descriptors=64)
        train = pd.DataFrame({"imageid": ["1.jpg", "2.jpg", "3.jpg"],
                              "pixel_hash": ["a", "b", "c"], "price": [100., 200., 500.]})
        records = []
        for _ in range(3):
            desc = rng.normal(size=(20, 128)).astype(np.float32)
            desc /= np.linalg.norm(desc, axis=1, keepdims=True)
            records.append((desc, rng.random((20, 2)), np.empty(0)))
        model = TraditionalModel(cfg["traditional"], cfg["target"], 2026)
        model.last_extraction_cache_ = {"images": 3, "hits": 0, "misses": 3}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CV_CACHE_ROOT": tmp}):
            with patch.object(model, "records", return_value=records) as reads:
                info = model.fit(Path(tmp), train)
                reads.assert_called_once()
                self.assertEqual(list(reads.call_args.args[1]), list(train.imageid))
            self.assertEqual(model.fit_ids, list(train.imageid))
            self.assertEqual(int(model.scaler.n_samples_seen_), len(train))
            self.assertEqual(info["vocabulary_descriptors"], 60)
            self.assertEqual(info["train_metrics"]["n"], 3)
            # Prediction cannot change any fitted component, even for an empty-keypoint image.
            fitted_mean = model.scaler.mean_.copy()
            centres = model.vocabulary.cluster_centers_.copy()
            coefficients = model.regressor.coef_.copy()
            held_out = pd.DataFrame({"imageid": ["4.jpg"], "price": [10000.]})
            empty = [(np.empty((0, 128)), np.empty((0, 2)), np.empty(0))]
            with patch.object(model, "records", return_value=empty):
                prediction = model.predict(tmp, held_out)
            self.assertTrue(np.isfinite(prediction).all())
            np.testing.assert_array_equal(fitted_mean, model.scaler.mean_)
            np.testing.assert_array_equal(centres, model.vocabulary.cluster_centers_)
            np.testing.assert_array_equal(coefficients, model.regressor.coef_)

    def test_traditional_route_and_environment_without_torch(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["output_root"] = tmp
            self.assertEqual(stage_directory(cfg).name, "03_traditional")
            with patch.dict(sys.modules, {"torch": None}):
                out = new_run(cfg)
            env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
            self.assertIsNone(env["cuda"])
            self.assertIsNone(env["gpu"])

    def test_analysis_keypoint_coverage_includes_blank_images(self):
        summary = analysis.keypoint_summary([0, 128, 256])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["zero_count"], 1)
        self.assertAlmostEqual(summary["zero_fraction"], 1/3)
        self.assertEqual(summary["quantiles"]["median"], 128)

    def test_raw_image_preview_is_local_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            Image.new("RGB", (24, 24), "white").save(root / "train/1.jpg")
            shared = root / "output/experiment_record/03_traditional/new_summary"
            train = pd.DataFrame({"imageid": ["1.jpg"]})
            empty = (np.empty((0, 128)), np.empty((0, 2)), np.empty(0))
            with patch.object(analysis, "ROOT", root), patch.object(analysis, "path_at_root", return_value=root), patch.object(analysis, "extract", return_value=empty):
                preview = analysis.save_keypoint_examples(train, [0], self.config(), shared)
                self.assertTrue(preview.is_file())
                self.assertTrue(preview.is_relative_to(root / ".local/traditional_previews"))
                self.assertFalse((shared / "keypoint_examples.png").exists())
                with self.assertRaises(FileExistsError):
                    analysis.save_keypoint_examples(train, [0], self.config(), shared)

    def test_long_cache_path_uses_short_temp_and_preserves_cached_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            Image.new("RGB", (24, 24), "white").save(root / "train/1.jpg")
            # The old temp suffix would exceed 260, while the final file and
            # new short sibling temp both fit classic Windows path limits.
            suffix_length = len("/sift_records/") + 64 + 1 + 68
            cache_root = root / ("x" * (245 - suffix_length - len(str(root)) - 1))
            cfg = self.config()
            with patch.dict(os.environ, {"CV_CACHE_ROOT": str(cache_root)}), patch("traditional.pipeline.np.savez", wraps=np.savez) as save:
                model = TraditionalModel(cfg["traditional"], cfg["target"], 2026)
                fresh = model.records(root, ["1.jpg"])[0]
                temporary_name = save.call_args.args[0].name
                final = next(cache_root.rglob("*.npz"))
                self.assertGreater(len(str(final)) + 37, 260)
                self.assertLess(len(str(final)), 260)
                self.assertLess(len(str(temporary_name)), len(str(final)))
                self.assertEqual(list(cache_root.rglob("*.tmp")), [])
                cached = model.records(root, ["1.jpg"])[0]
                self.assertEqual(model.last_extraction_cache_["hits"], 1)
                for a, b in zip(fresh, cached):
                    np.testing.assert_array_equal(a, b)

    def test_failed_cache_write_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            Image.new("RGB", (24, 24), "white").save(root / "train/1.jpg")
            with patch.dict(os.environ, {"CV_CACHE_ROOT": str(root / "cache")}), patch("traditional.pipeline.np.savez", side_effect=OSError("simulated write failure")):
                cfg = self.config()
                model = TraditionalModel(cfg["traditional"], cfg["target"], 2026)
                with self.assertRaises(OSError):
                    model.records(root, ["1.jpg"])
            self.assertEqual(list((root / "cache").rglob("*.tmp")), [])
            self.assertEqual(list((root / "cache").rglob("*.npz")), [])

    def test_environment_preserves_torch_present_metadata(self):
        cfg = self.config()
        fake_torch = SimpleNamespace(version=SimpleNamespace(cuda="test-cuda"),
                                     backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 123)),
                                     cuda=SimpleNamespace(is_available=lambda: False))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": fake_torch}):
            cfg["output_root"] = tmp
            out = new_run(cfg)
            env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(env["cuda"], "test-cuda")
            self.assertEqual(env["cudnn"], 123)
            self.assertIsNone(env["gpu"])

    def test_neural_environment_does_not_silently_ignore_missing_torch(self):
        cfg = self.config()
        cfg["method"] = "neural"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": None}):
            cfg["output_root"] = tmp
            with self.assertRaises(ModuleNotFoundError):
                new_run(cfg)

    def test_analysis_rejects_smoke_instead_of_reporting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, mean = root / "smoke", root / "mean"
            write_json(run / "config.json", self.config())
            write_json(run / "status.json", {"status": "completed", "kind": "smoke"})
            write_json(mean / "status.json", {"status": "completed", "kind": "development"})
            write_json(mean / "data_manifest.json", {"split_sha256": "test"})
            frame = pd.DataFrame({"imageid": ["1.jpg", "2.jpg"], "price": [100., 2000.],
                                  "partition": ["train", "validation"]})
            with patch.object(analysis, "load_split", return_value=(frame, {"split_sha256": "test"})):
                with self.assertRaises(ValueError):
                    analysis.analyze([run], root / "analysis", mean)
            self.assertFalse((root / "analysis").exists())


if __name__ == "__main__":
    unittest.main()
