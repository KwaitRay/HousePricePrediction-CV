"""Machine-local paths, resolved against repository root, independent of cwd."""
import json
import os
from pathlib import Path

REPOSITORY=Path(__file__).resolve().parents[3]
DEFAULTS={
    'data_root':'.local/house_dataset',
    'split_path':'.local/artifacts/split_v1.csv',
    'records_root':'.local/experiment_record',
    'cache_root':'.local/cache',
    'eda_output_root':'.local/analysis_outputs',
}

def local_path(key):
    if key not in DEFAULTS:raise KeyError(key)
    config=REPOSITORY/'local_paths.json'
    settings=json.loads(config.read_text(encoding='utf-8-sig')) if config.exists() else {}
    unknown=set(settings)-set(DEFAULTS)
    if unknown:raise ValueError(f'Unknown local path settings: {sorted(unknown)}')
    value=os.environ.get('CV_'+key.upper(),settings.get(key,DEFAULTS[key]))
    result=Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (REPOSITORY/result).resolve()
