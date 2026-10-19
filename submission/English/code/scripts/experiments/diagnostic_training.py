"""Optional train-only mild price weighting; unchanged unweighted evaluation."""
import sys
import torch
import training.engine as engine
from common.config import write_json


def weighted_sum(prediction, target, weights):
    if prediction.shape != target.shape or weights.shape != target.shape:
        raise ValueError('Weighted loss shapes differ')
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid sample weights')
    return ((prediction - target).square() * weights).sum()


def main():
    original_epoch = engine.train_epoch
    def epoch(model, ds, c, optimizer, device, batches=None):
        spec = c.get('_train_price_weights')
        if not spec:
            return original_epoch(model, ds, c, optimizer, device, batches)
        original_loss = engine.nn.functional.mse_loss
        def loss(prediction, target, reduction='mean'):
            if reduction != 'sum' or c['target']['transform'] != 'scale':
                raise ValueError('This adapter requires scaled-target sum MSE')
            bands = torch.bucketize(target * c['target']['scale'], target.new_tensor([300., 700., 1300.]), right=False)
            weights = target.new_tensor(spec['weights'])[bands]
            return weighted_sum(prediction, target, weights)
        engine.nn.functional.mse_loss = loss
        try:
            return original_epoch(model, ds, c, optimizer, device, batches)
        finally:
            engine.nn.functional.mse_loss = original_loss
    engine.train_epoch = epoch
    import run
    original_new = run.new_run
    def recorded(c, kind='development'):
        out = original_new(c, kind)
        write_json(out / 'diagnostic_training.json', {'weighting': c.get('_train_price_weights'),
                   'evaluation': 'Unweighted original-price MSE; original frozen validation population',
                   'note': 'Price-band coverage probe, not full-population semantic balancing'})
        return out
    run.new_run = recorded
    run.main(sys.argv[1:])


if __name__ == '__main__':
    main()
