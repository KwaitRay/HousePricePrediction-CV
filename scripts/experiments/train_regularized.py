"""Cached training with read-only regularization diagnostics after each epoch."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import torch
import training.engine as engine
from common.config import digest, write_json
from models.alexnet import penalty_parameters


@torch.no_grad()
def weight_statistics(model):
    weights = [p for _, p in penalty_parameters(model)]
    l1 = sum(float(p.abs().sum()) for p in weights)
    squared = sum(float(p.square().sum()) for p in weights)
    return {'weight_l1': l1, 'weight_l2': squared ** .5}


def main():
    original = engine.train_epoch
    def measured(model, ds, c, optimizer, device, batches=None):
        row = original(model, ds, c, optimizer, device, batches)
        row.update(weight_statistics(model))
        # Monitoring sum: sample-mean data loss plus update-mean penalty.
        row['total_loss_monitor'] = row['data_loss'] + row['penalty']
        return row
    engine.train_epoch = measured
    import run
    original_new_run = run.new_run
    def recorded(c, kind='development'):
        out = original_new_run(c, kind)
        write_json(out / 'regularization_runtime.json', {
            'adapter_sha256': digest(Path(__file__)),
            'weight_norms': 'Eligible weights only; read after epoch with no grad and no RNG consumption',
            'total_loss_monitor': 'Sample-mean data loss plus update-mean penalty; diagnostic only, not selection metric',
            'selection_metric': 'Validation original-price MSE, excludes parameter penalties'})
        return out
    run.new_run = recorded
    import train_cached
    train_cached.main()


if __name__ == '__main__': main()
