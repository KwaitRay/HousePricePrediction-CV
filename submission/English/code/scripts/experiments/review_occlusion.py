"""Exploratory inference-only occlusion of six preselected validation examples."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from continue_training import *
from models.alexnet import AlexNetRegressor
from preparation.images import HouseImages


def main():
    work=local_path('records_root')/'05_augmentation/error_review_20261005'
    state=read_json(local_path('records_root')/'05_augmentation/geometry_20261004/state.json')
    run=Path(state['runs']['augment_01_flip_2026'])
    c=read_json(run/'config.json');torch.set_num_threads(c['training']['cpu_threads'])
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model=AlexNetRegressor(c['model']);model.load_state_dict(ck['model']);model.eval().to('cuda')
    ids=['2138.jpg','584.jpg','192.jpg','5767.jpg','6972.jpg','4471.jpg']
    split=pd.read_csv(local_path('split_path'));sample=split.set_index('imageid').loc[ids].reset_index()
    assert sample.partition.eq('validation').all()
    ds=HouseImages(local_path('data_root'),sample,c,ck['stats'])
    pred=pd.read_csv(run/'predictions.csv').set_index('imageid')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(6,2,figsize=(8,19))
    records=[]
    with torch.inference_mode():
        for i,name in enumerate(ids):
            x=ds[i][0];original=float(model(x[None].cuda()).item()*1000)
            if not np.isclose(original,pred.loc[name,'prediction'],rtol=1e-5,atol=.05):raise ValueError('Checkpoint prediction mismatch')
            altered=[]
            for y in range(7):
                for z in range(7):
                    occluded=x.clone();occluded[:,y*32:(y+1)*32,z*32:(z+1)*32]=0.
                    altered.append(occluded)
            values=[]
            for j in range(0,len(altered),4):values.extend((model(torch.stack(altered[j:j+4]).cuda())*1000).cpu().tolist())
            change=np.array(values).reshape(7,7)-original
            raw=(x*.5+.5).permute(1,2,0).numpy().clip(0,1)
            axes[i,0].imshow(raw);axes[i,0].set_title(f'{name} 真值{sample.iloc[i].price:.0f} 预测{original:.0f}');axes[i,0].axis('off')
            vmax=max(abs(change.min()),abs(change.max()))
            axes[i,1].imshow(raw);h=axes[i,1].imshow(change,extent=[0,224,224,0],cmap='coolwarm',vmin=-vmax,vmax=vmax,alpha=.7)
            axes[i,1].set_title('遮挡后预测变化：红升、蓝降');axes[i,1].axis('off');fig.colorbar(h,ax=axes[i,1],fraction=.045)
            top=np.unravel_index(np.abs(change).argmax(),change.shape)
            records.append({'imageid':name,'seed':2026,'checkpoint_epoch':ck['epoch'],'prediction':original,
                            'price':float(sample.iloc[i].price),'occlusion_delta':change.tolist(),
                            'largest_absolute_change_cell_row_col':[int(k) for k in top],'largest_change':float(change[top])})
    fig.tight_layout();fig.savefig(work/'occlusion_six.png',dpi=130);plt.close(fig)
    write_json(work/'occlusion_six.json',{'scope':'6 purposively selected examples, seed2026 best checkpoint; 32x32 gray occlusion; exploratory sensitivity not causal attention attribution','records':records})
    print([(r['imageid'],r['largest_absolute_change_cell_row_col'],r['largest_change']) for r in records])


if __name__=='__main__':main()
