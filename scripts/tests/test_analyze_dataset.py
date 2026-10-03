"""Tests for scripts/analyze_dataset.py; synthetic data only, no network."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_dataset as analysis


def fixture(root: Path):
    for split in ["train", "test"]:
        (root / split).mkdir(parents=True)
    rng = np.random.default_rng(17)
    for index in range(1, 16):
        split = "train" if index <= 12 else "test"
        array = rng.integers(0, 256, (30 + index, 45, 3), dtype=np.uint8)
        Image.fromarray(array).save(root / split / f"{index}.jpg")
    # Exact duplicates inside training (with conflicting labels) and across splits.
    shutil.copyfile(root / "train" / "1.jpg", root / "train" / "12.jpg")
    shutil.copyfile(root / "train" / "1.jpg", root / "test" / "13.jpg")
    pd.DataFrame({"imageid": [f"{i}.jpg" for i in range(1, 13)],
                  "price": [200 + 50 * i for i in range(12)]}).to_csv(root / "train.csv", index=False)
    test = pd.DataFrame({"imageid": [f"{i}.jpg" for i in range(13, 16)]})
    test.to_csv(root / "test.csv", index=False)
    test.assign(price=500).to_csv(root / "sample_solution.csv", index=False)


class AnalysisTests(unittest.TestCase):
    def test_bktree_matches_exhaustive_search(self):
        values = [0, 1, 3, 7, 15, 255, 255, 0xFFFF, 0x123456789ABCDEF0]
        tree = analysis.BKTree()
        for i, value in enumerate(values):
            tree.add(value, i)
        for query in [0, 1, 254, 0x123456789ABCDEFF]:
            for radius in [0, 1, 4, 8]:
                expected = sorted((i, (query ^ v).bit_count()) for i, v in enumerate(values)
                                  if (query ^ v).bit_count() <= radius)
                self.assertEqual(sorted(tree.query(query, radius)), expected)

    def test_exact_groups_stay_together_and_split_is_reproducible(self):
        train = pd.DataFrame({"imageid": [f"{i}.jpg" for i in range(30)],
                              "price": [200.] * 30,
                              "pixel_sha256": [f"g{i // 2:02}" for i in range(30)]})
        first = analysis.make_split(train, .2, 2026)
        second = analysis.make_split(train.sample(frac=1, random_state=42), .2, 2026)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.groupby("exact_group").partition.nunique().max(), 1)
        self.assertEqual(set(first.partition), {"train", "validation"})
        self.assertEqual(set(first.imageid), set(train.imageid))

    def test_baseline_does_not_fit_validation_labels(self):
        manifest = pd.DataFrame({"imageid": ["1.jpg", "2.jpg", "3.jpg"],
                                 "price": [1., 3., 100.],
                                 "partition": ["train", "train", "validation"]})
        metrics, _ = analysis.baseline_metrics(manifest)
        self.assertEqual(metrics.iloc[0].constant_prediction, 2.)
        self.assertEqual(metrics.iloc[0].mse, 98. ** 2)
        self.assertEqual(metrics.iloc[0].mae, 98.)

    def test_near_duplicate_cap_and_exact_pair_exclusion(self):
        frame = pd.DataFrame({"imageid": ["1.jpg", "2.jpg", "3.jpg"],
                              "price": [100., 200., 300.], "dhash": ["0", "0", "0"],
                              "pixel_sha256": ["same", "same", "different"]})
        pairs, truncated = analysis.near_duplicate_pairs(frame, 0, 1)
        self.assertTrue(truncated)
        self.assertEqual(len(pairs), 1)
        self.assertNotEqual(set(pairs.iloc[0][["imageid_a", "imageid_b"]]), {"1.jpg", "2.jpg"})

    def test_invalid_schema_and_traversal_fail_before_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            frame = pd.read_csv(root / "train.csv")
            frame.loc[0, "imageid"] = "../outside.jpg"
            frame.loc[1, "price"] = float("nan")
            frame.to_csv(root / "train.csv", index=False)
            _, audit = analysis.audit_tables(root, 12, 3)
            self.assertTrue(any("unsafe" in e for e in audit["errors"]))
            self.assertTrue(any("prices" in e for e in audit["errors"]))

    def test_decode_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.jpg"
            path.write_bytes(b"not an image")
            row = analysis.inspect_image(("train", path.name, path))
            self.assertTrue(row["error"])

    def test_full_and_partial_pipeline_on_synthetic_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            fixture(root)
            original = {str(p.relative_to(root)): analysis.file_sha256(p)
                        for p in root.rglob("*") if p.is_file()}
            output = Path(temporary) / "results"
            common = ["--data-root", str(root), "--output-root", str(output),
                      "--expected-train", "12", "--expected-test", "3", "--workers", "2"]
            self.assertEqual(analysis.main(common), 0)
            full = next(output.iterdir())
            summary = json.loads((full / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["full_scan"])
            self.assertEqual(summary["duplicates"]["train_pixel_groups_with_label_conflicts"], 1)
            self.assertEqual(summary["duplicates"]["groups_by_kind_and_scope"]["pixel_sha256:cross_train_test"], 1)
            split = pd.read_csv(full / "tables" / "split_candidate.csv")
            self.assertEqual(split[split.imageid.isin(["1.jpg", "12.jpg"])].partition.nunique(), 1)
            self.assertEqual(len(split), 12)
            self.assertTrue((full / "report.md").is_file())
            inventory = pd.read_csv(full / "tables" / "image_inventory.csv")
            self.assertTrue(inventory.loc[inventory.split.eq("test"), "brightness"].isna().all())
            for png in (full / "figures").glob("*.png"):
                with Image.open(png) as image:
                    image.load()
                    self.assertGreater(image.width, 100)
            self.assertEqual(analysis.main(common + ["--max-images", "2"]), 0)
            partial = next(p for p in output.iterdir() if p != full)
            partial_summary = json.loads((partial / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(partial_summary["full_scan"])
            self.assertFalse((partial / "tables" / "split_candidate.csv").exists())
            after = {str(p.relative_to(root)): analysis.file_sha256(p)
                     for p in root.rglob("*") if p.is_file()}
            self.assertEqual(original, after)


if __name__ == "__main__":
    unittest.main()
