"""Train-only SIFT vocabulary, optional spatial/color features, regression."""
from pathlib import Path
import time
import hashlib
import json
import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from common.config import stable_seed, write_json, ROOT, digest
from common.metrics import transform_target, inverse_target


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def histogram(labels, size):
    h = np.bincount(labels, minlength=size).astype(np.float64)
    return unit(np.sqrt(h / max(h.sum(), 1)))


def regions(arr):
    h, w = arr.shape[:2]
    return [arr, arr[:h//2, :w//2], arr[:h//2, w//2:], arr[h//2:, :w//2], arr[h//2:, w//2:]]


def appearance(rgb, kind):
    if kind == "none":
        return np.empty(0)
    if kind == "color":
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        return np.concatenate([np.r_[part.mean((0, 1)), part.std((0, 1))] for part in regions(lab)])
    x = rgb.astype(np.float64) / 255
    y = x @ np.array([.299, .587, .114])
    if kind == "shading":
        smooth = cv2.GaussianBlur(y, (9, 9), 2, borderType=cv2.BORDER_REFLECT_101)
        residual = abs(y - smooth)
        return np.array([v for a, b in zip(regions(smooth), regions(residual))
                         for v in [a.std(), np.quantile(a, .9) - np.quantile(a, .1), b.mean(), np.quantile(b, .9)]])
    if kind != "gradient":
        raise ValueError(f"未知附加特征: {kind}")
    dx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3, scale=1/8, borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3, scale=1/8, borderType=cv2.BORDER_REFLECT_101)
    mag, angle = np.hypot(dx, dy), np.mod(np.arctan2(dy, dx), np.pi)
    values = []
    for m, a in zip(regions(mag), regions(angle)):
        hist, _ = np.histogram(a, bins=8, range=(0, np.pi), weights=m)
        values.extend(np.r_[hist / max(hist.sum(), 1e-12), m.mean()])
    return np.array(values)


def extract(path, cfg):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        scale = cfg["long_edge"] / max(h, w)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
        rgb = np.array(image)
    sift = cv2.SIFT_create(nfeatures=cfg["max_keypoints"], nOctaveLayers=3, contrastThreshold=.04, edgeThreshold=10, sigma=1.6)
    points, desc = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    if desc is None:
        desc, coords = np.empty((0, 128), np.float32), np.empty((0, 2), np.float32)
    else:
        order = sorted(range(len(points)), key=lambda i: (-points[i].response, *points[i].pt, points[i].size, points[i].angle))[:cfg["max_keypoints"]]
        desc = desc[order].astype(np.float32)
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
        coords = np.array([[points[i].pt[0] / rgb.shape[1], points[i].pt[1] / rgb.shape[0]] for i in order])
    return desc, coords, appearance(rgb, cfg["appearance"])


class TraditionalModel:
    def __init__(self, cfg, target, seed):
        self.cfg, self.target, self.seed = cfg, target, seed

    def records(self, root, ids, partition="train"):
        return [extract(Path(root) / partition / name, self.cfg) for name in ids]

    def features(self, records):
        result = []
        k = self.cfg["words"]
        for desc, coords, extra in records:
            labels = self.vocabulary.predict(desc) if len(desc) else np.empty(0, dtype=int)
            global_hist = histogram(labels, k)
            if self.cfg["spatial_pyramid"]:
                cells = np.minimum((coords * 2).astype(int), 1)
                pieces = [global_hist / np.sqrt(2)]
                for y in range(2):
                    for x in range(2):
                        mask = (cells[:, 0] == x) & (cells[:, 1] == y)
                        pieces.append(histogram(labels[mask], k) / np.sqrt(8))
                global_hist = unit(np.concatenate(pieces))
            result.append(np.r_[global_hist, extra])
        return np.asarray(result)

    def fit(self, root, train):
        cv2.setNumThreads(1)
        rec = self.records(root, train.imageid)
        pool = []
        for name, (desc, _, _) in zip(train.imageid, rec):
            rng = np.random.default_rng(stable_seed(self.seed, "descriptor", name))
            pool.append(desc[rng.choice(len(desc), min(len(desc), self.cfg["per_image_sample"]), replace=False)])
        pool = np.concatenate(pool)
        if len(pool) < self.cfg["words"]:
            raise ValueError("训练侧描述子少于视觉词数量；不能借用验证描述子，请另登记更小词典")
        rng = np.random.default_rng(self.seed)
        pool = pool[rng.choice(len(pool), min(len(pool), self.cfg["max_descriptors"]), replace=False)]
        # Reuse the exact training-side vocabulary across regressor, spatial and appearance controls.
        # A new fold, full-data refit, seed, dictionary size, image or extraction code gets a new key.
        from importlib.metadata import version
        key_data = {"images": train[["imageid", "pixel_hash"]].to_dict("records"), "seed": self.seed,
                    "extraction": {k: self.cfg[k] for k in ("long_edge", "max_keypoints", "words", "per_image_sample", "max_descriptors")},
                    "source": digest(__file__), "opencv": cv2.__version__, "sklearn": version("scikit-learn")}
        cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        from common.local_paths import local_path
        cache = local_path("cache_root") / "vocabulary_cache" / f"{cache_key}.joblib"
        reused = cache.exists()
        if reused:
            self.vocabulary = joblib.load(cache)
        else:
            self.vocabulary = MiniBatchKMeans(n_clusters=self.cfg["words"], batch_size=2048, n_init=3, max_iter=100, random_state=self.seed).fit(pool)
            cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vocabulary, cache)
        features = self.features(rec)
        self.scaler = StandardScaler().fit(features)
        self.regressor = (Ridge(alpha=self.cfg["ridge_alpha"], fit_intercept=True, solver="svd")
                          if self.cfg["regressor"] == "ridge" else
                          SVR(C=10, epsilon=.05, gamma="scale", tol=.001, cache_size=512, max_iter=-1))
        self.regressor.fit(self.scaler.transform(features), transform_target(train.price.to_numpy(), self.target))
        self.fit_ids = list(train.imageid)
        return {"keypoints": [len(r[0]) for r in rec], "zero_keypoint_count": sum(len(r[0]) == 0 for r in rec),
                "feature_dimension": features.shape[1], "vocabulary_descriptors": len(pool),
                "vocabulary_cache_key": cache_key, "vocabulary_reused": reused}

    def predict(self, root, frame, partition="train"):
        features = self.features(self.records(root, frame.imageid, partition))
        return inverse_target(self.regressor.predict(self.scaler.transform(features)), self.target)


def fit_traditional(c, root, train, validation, out):
    start = time.perf_counter()
    model = TraditionalModel(c["traditional"], c["target"], c["seed"])
    info = model.fit(root, train)
    train_seconds = time.perf_counter() - start
    joblib.dump(model, out / "traditional.joblib")
    write_json(out / "features.json", info)
    start = time.perf_counter()
    pred = model.predict(root, validation) if len(validation) else None
    write_json(out / "cost.json", {"train_seconds": train_seconds, "predict_seconds": time.perf_counter()-start})
    return pred
