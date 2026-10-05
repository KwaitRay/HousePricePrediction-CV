"""Standard-library checks for the clean submission boundary."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "package_submission.py"
spec = importlib.util.spec_from_file_location("package_submission", SCRIPT)
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class SubmissionPackaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, data in {
            "project.py": "pass", "local_paths.example.json": "{}",
            "scripts/experiments/run.py": "pass",
            "scripts/experiments/common/config.py": "pass",
            "scripts/experiments/configs/base.json": "{}",
            "scripts/experiments/configs/a.json": '{"extends":"base.json","selection":{"run":"D:/private/run"}}',
            "scripts/experiments/configs/unused.json": "{}",
            "design/plan.md": "internal", "output/experiment_record/report.md": "internal",
            "archive/old.py": "pass", "local_paths.json": "private",
            "scripts/experiments/tests/test_dummy.py": "pass",
            "scripts/experiments/common/__pycache__/config.pyc": "cache",
        }.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(data, encoding="utf-8")

    def bundle(self, **kwargs):
        return pack.make_bundle(self.root, ["configs/a.json"], **kwargs)

    def test_whitelist_and_dependency_closure(self):
        files = self.bundle()
        self.assertIn("scripts/experiments/configs/base.json", files)
        self.assertIn("scripts/experiments/common/config.py", files)
        for name in files:
            self.assertFalse(name.endswith(".md"))
            self.assertFalse(name.startswith(("design/", "archive/", "output/")))
            self.assertNotIn("tests/", name)
            self.assertNotIn("__pycache__", name)
        self.assertNotIn("local_paths.json", files)
        self.assertNotIn("scripts/experiments/configs/unused.json", files)
        selected = json.loads(files["scripts/experiments/configs/a.json"])
        self.assertNotIn("run", selected["selection"])

    def test_missing_final_assets_fail(self):
        with self.assertRaises(ValueError):
            self.bundle(final=True)

    def test_escape_and_cycle_fail(self):
        config = self.root / "scripts/experiments/configs/a.json"
        config.write_text('{"extends":"../../../local_paths.example.json"}')
        with self.assertRaises(ValueError):
            self.bundle()
        config.write_text('{"extends":"a.json"}')
        with self.assertRaises(ValueError):
            self.bundle()

    def test_zip_readback_and_no_overwrite(self):
        output = self.root / "bundle.zip"
        files = self.bundle()
        pack.export_zip(files, output)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), set(files))
            self.assertIsNone(archive.testzip())
        with self.assertRaises(FileExistsError):
            pack.export_zip(files, output)

    def test_frozen_split_pair_preserved(self):
        split = self.root / "split.csv"
        split.write_bytes(b"imageid,partition\n1.jpg,train\n")
        split.with_suffix(".json").write_bytes(b'{"hash":"frozen"}')
        files = self.bundle(split=split)
        self.assertEqual(files[".local/artifacts/split_v1.csv"], split.read_bytes())
        self.assertEqual(files[".local/artifacts/split_v1.json"], split.with_suffix(".json").read_bytes())

    def test_bad_predictions_fail(self):
        predictions = self.root / "bad.csv"
        predictions.write_text("imageid,price\n1.jpg,nan\n")
        with self.assertRaises(ValueError):
            self.bundle(predictions=predictions)


if __name__ == "__main__":
    unittest.main()
