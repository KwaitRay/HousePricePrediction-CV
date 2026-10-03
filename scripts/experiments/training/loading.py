"""Reusable loaders with explicit epoch delivery and legacy sample ordering."""
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from common.config import stable_seed


class EpochDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        # Each worker processes items serially. The epoch travels with the task,
        # so persistent workers never rely on a stale main-process attribute.
        self.dataset.epoch = epoch
        return self.dataset[index]


class EpochSampler(Sampler):
    def __init__(self, dataset, seed, training):
        self.dataset, self.seed, self.training = dataset, seed, training

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        epoch = self.dataset.epoch
        if self.training:
            generator = torch.Generator().manual_seed(stable_seed(self.seed, "shuffle", epoch))
            # The old fresh DataLoader consumes one int64 for its worker base
            # seed before RandomSampler consumes the same generator.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            indices = RandomSampler(self.dataset, generator=generator)
        else:
            indices = range(len(self.dataset))
        return iter((epoch, index) for index in indices)


def reusable_loader(dataset, config, training, worker_init_fn):
    workers = config["training"]["workers"]
    return DataLoader(EpochDataset(dataset), batch_size=config["training"]["batch_size"],
                      sampler=EpochSampler(dataset, config["seed"], training),
                      num_workers=workers, drop_last=False, persistent_workers=workers > 0,
                      worker_init_fn=worker_init_fn,
                      generator=torch.Generator().manual_seed(stable_seed(config["seed"], "loader_workers", training)))


class LoaderPool:
    """One run owns its loaders; small deterministic probes use cached batches."""
    def __init__(self, config, legacy_loader, worker_init_fn):
        self.config, self.legacy_loader, self.worker_init_fn = config, legacy_loader, worker_init_fn
        self.optimized = config["training"].get("loader_mode", "legacy") == "persistent"
        self.loaders, self.probes = {}, {}

    def get(self, dataset, training=False, probe=False):
        if not self.optimized:
            return self.legacy_loader(dataset, self.config, training)
        key = (dataset, training)
        if probe:
            if dataset.training:
                raise ValueError("Only unaugmented evaluation probes may be cached")
            if dataset not in self.probes:
                cfg = {**self.config, "training": {**self.config["training"], "workers": 0}}
                self.probes[dataset] = list(self.legacy_loader(dataset, cfg))
            return self.probes[dataset]
        if key not in self.loaders:
            self.loaders[key] = reusable_loader(dataset, self.config, training, self.worker_init_fn)
        return self.loaders[key]

    def close(self):
        # PyTorch has no public explicit close for persistent DataLoader workers.
        # Keep the private shutdown use isolated and test process cleanup.
        for batches in self.loaders.values():
            iterator = getattr(batches, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                batches._iterator = None
        self.loaders.clear()
        self.probes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
