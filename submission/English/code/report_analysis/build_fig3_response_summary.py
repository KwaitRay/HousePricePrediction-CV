"""Summarize archived Fig. 3 blur responses without encoder inference."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'figures/analysis_sources/occlusion/per_region_predictions.csv'
OUT = ROOT / 'figures/analysis_sources/report_fig3_spatial_response_comparison.csv'


def main():
    table = pd.read_csv(SOURCE)
    rows = []
    for imageid, group in table[table['mode'] == 'blur'].groupby('imageid'):
        responses = {}
        for model in ['Single-view KRR', 'Two-view KRR', 'Two-view PCA KRR']:
            selected = group[group.model == model].sort_values('cell')
            assert selected.cell.tolist() == list(range(16))
            responses[model] = selected.delta_price.to_numpy()
        single, whole, pca = responses.values()
        rows.append(dict(imageid=imageid, grid_cells=16,
                         single_view_peak_drop=single.max(),
                         two_view_peak_drop=whole.max(),
                         two_view_pca_peak_drop=pca.max(),
                         blur_response_correlation=np.corrcoef(whole, pca)[0, 1],
                         source='occlusion/per_region_predictions.csv',
                         source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == '__main__':
    main()
