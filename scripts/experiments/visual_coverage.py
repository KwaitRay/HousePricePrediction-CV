"""Training-only stratified visual audit; annotations are explicit reviewed observations."""
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from common.config import write_json, read_json, digest
from common.local_paths import local_path, REPOSITORY

WORK=local_path('records_root')/'05_augmentation/visual_coverage_20261005'
PUBLIC=REPOSITORY/'output/experiment_record/05_augmentation/analyses/visual_coverage_20261005'
BANDS=['low','middle','upper','tail']
TYPES={'F':'地面外观','A':'航拍或高位全景','D':'局部或环境特写','I':'室内','R':'建筑效果图','U':'无法判断'}
FLAGS={'P':'泳池','T':'黄昏夜景','O':'明显植被遮挡','W':'木质山林小屋'}


def sheets(frame,prefix,batch=20):
    for offset in range(0,len(frame),batch):
        chunk=frame.iloc[offset:offset+batch];canvas=Image.new('RGB',(1600,5*224),'white');draw=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(chunk.iterrows()):
            with Image.open(local_path('data_root')/'train'/r.imageid) as im:
                im=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(316,197))
                x=(j%5)*320;y=(j//5)*224;canvas.paste(im,(x,y));draw.text((x+5,y+199),r.code,fill='black')
        canvas.crop((0,0,1600,math.ceil(len(chunk)/5)*224)).save(WORK/f'{prefix}_{offset//batch+1:02d}.jpg',quality=95)


def prepare():
    WORK.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    if (WORK/'sample.csv').exists():raise ValueError('Frozen sample already exists')
    f=pd.read_csv(local_path('split_path'));f=f[f.partition=='train'].copy()
    f['band']=pd.cut(f.price,[0,300,700,1300,np.inf],labels=BANDS).astype(str)
    chosen=pd.concat([f[f.band==b].sample(100,random_state=20261005+i) for i,b in enumerate(BANDS)])
    chosen=chosen.sample(frac=1,random_state=20261005).reset_index(drop=True)
    chosen['code']=[f'V{i+1:03d}' for i in range(len(chosen))]
    assert len(chosen)==400 and chosen.imageid.nunique()==400 and chosen.partition.eq('train').all()
    chosen.to_csv(WORK/'sample.csv',index=False,encoding='utf-8-sig');sheets(chosen,'sheet')
    qa=chosen.sample(40,random_state=20261006).copy();qa['original_code']=qa.code;qa['code']=[f'Q{i+1:03d}' for i in range(len(qa))]
    qa.to_csv(WORK/'qa_sample.csv',index=False,encoding='utf-8-sig');sheets(qa,'qa')
    registration={'split_sha256':digest(local_path('split_path')),'sample_sha256':digest(WORK/'sample.csv'),
                  'train_population':{b:int((f.band==b).sum()) for b in BANDS},'per_band_sample':100,
                  'band_seeds':{b:20261005+i for i,b in enumerate(BANDS)},'presentation_seed':20261005,
                  'types':TYPES,'flags':FLAGS,'annotation_source':'assistant visual review; not independent human labels',
                  'sample_selection':'training price strata only; no predictions or errors used','qa_seed':20261006}
    write_json(WORK/'registration.json',registration);write_json(PUBLIC/'registration.json',registration)
    print('Prepared 400 training images, 20 blind contact sheets and 40-image same-reviewer QA')


def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]


def summarize():
    f=pd.read_csv(WORK/'sample.csv');ann=read_json(WORK/'annotations.json')
    assert set(ann)==set(f.code), 'All sampled images must have explicit labels'
    reg=read_json(WORK/'registration.json');assert digest(WORK/'sample.csv')==reg['sample_sha256']
    unknown=read_json(WORK/'uncertain_flags.json') if (WORK/'uncertain_flags.json').exists() else {}
    rows=[]
    for _,r in f.iterrows():
        typ,flags=ann[r.code].split(':');assert typ in TYPES
        assert set(flags)-set(FLAGS)==set()
        rows.append({'code':r.code,'band':r.band,'type':typ,**{k:None if k in unknown.get(r.code,[]) else k in flags for k in FLAGS}})
    labeled=pd.DataFrame(rows)
    cells=[]
    for band in BANDS:
        a=labeled[labeled.band==band];N=reg['train_population'][band]
        for label,name in {**TYPES,**FLAGS}.items():
            k=int((a.type==label).sum() if label in TYPES else a[label].sum());n=len(a)
            u=int((a.type=='U').sum()) if label in TYPES and label!='U' else (int(a[label].isna().sum()) if label in FLAGS else 0)
            ci=[wilson(k,n)[0],wilson(k+u,n)[1]]
            cells.append({'band':band,'label':label,'name':name,'sample_n':n,'positive':k,'unknown':u,'fraction':k/n,'possible_fraction_upper':(k+u)/n,'ci95':ci,
                          'population':N,'estimated_count':N*k/n,'estimated_count_ci95':[N*x for x in ci],
                          'sample_rare':k<=5,'low_coverage':k<=5 and ci[1]<.1,'absolute_coverage_candidate':ci[1]*N<50})
    totals=[]
    for label,name in {**TYPES,**FLAGS}.items():
        c=[x for x in cells if x['label']==label]
        totals.append({'label':label,'name':name,'weighted_fraction':sum(x['estimated_count'] for x in c)/6399,
                       'estimated_count':sum(x['estimated_count'] for x in c)})
    write_json(PUBLIC/'coverage.json',{'cells':cells,'weighted_totals':totals,'annotated':400,'unannotated':5999,
                                     'caveat':'Sampling estimates, pointwise Wilson intervals without annotation-error adjustment; no full-population visual labels.'})
    labeled.to_csv(WORK/'labels_joined.csv',index=False,encoding='utf-8-sig')
    first=read_json(WORK/'annotations_first_pass.json');qa=read_json(WORK/'qa_annotations.json')
    mapping=pd.read_csv(WORK/'qa_sample.csv');disagreements=[];type_match=0;exact_match=0
    flag_matches={k:0 for k in FLAGS}
    for _,row in mapping.iterrows():
        x,y=first[row.original_code],qa[row.code];a,b=x.split(':'),y.split(':')
        type_match+=a[0]==b[0];exact_match+=x==y
        for k in FLAGS:flag_matches[k]+=(k in a[1])==(k in b[1])
        if x!=y:disagreements.append({'code':row.original_code,'first':x,'second':y})
    write_json(PUBLIC/'qa.json',{'n':40,'type_matches':type_match,'all_fields_matches':exact_match,'flag_matches':flag_matches,'disagreements':disagreements,
                                'scope':'Same assistant repeated review in the same session, not independent rater agreement; memory effects possible.'})
    print('Coverage computed for 400 assistant-reviewed training images')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','summarize']);args=parser.parse_args()
    prepare() if args.command=='prepare' else summarize()
