"""CPU integration check with existing private heads/features, outside the release.

No encoder execution or fitting; artifacts stay in .local/portable_adapter_check.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'submission/English/code'
TARGET=ROOT/'.local/portable_adapter_check'


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def main():
    if (TARGET/'comparison.json').exists():raise FileExistsError(TARGET)
    recipe=json.loads((CODE/'final_recipe.json').read_text(encoding='utf-8'))
    folder=ROOT/'.local/experiment_record'/recipe['run']
    (TARGET/'fitted').mkdir(parents=True,exist_ok=True)
    write(TARGET/'fitted/recipe.json',recipe)
    for i,info in enumerate(recipe['models']):
        shutil.copyfile(folder/f'component_{i}_full_head.npz',TARGET/'fitted'/f'component_{i}_head.npz')
        if i==1:shutil.copyfile(folder/'component_1_full_metric.joblib',TARGET/'fitted/component_1_metric.joblib')
        if i==2:shutil.copyfile(folder/'component_2_full_pca.npz',TARGET/'fitted/component_2_pca.npz')
        cache=ROOT/info['test_cache'];m=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
        batch=m['batch_size'];parts=[]
        for offset in range(0,3000,batch):parts.append(np.load(cache/f'{offset:05d}_{min(batch,3000-offset)}.npy',allow_pickle=False))
        dest=TARGET/'features'/f'component_{i}_test.npy';dest.parent.mkdir(exist_ok=True)
        np.save(dest,np.concatenate(parts))
        write(dest.with_suffix('.json'),{'imageids':pd.read_csv(CODE/'inputs/test.csv').imageid.tolist(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/submission_support/reproduce_final.py'),'predict','--output',str(TARGET)],check=True)
    check=json.loads((TARGET/'comparison.json').read_text(encoding='utf-8'))
    assert check['n']==3000 and check['max_abs_difference_from_archived_submission']<1e-7
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
