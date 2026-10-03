import torch
from models.alexnet import penalty_parameters

class HybridMuon(torch.optim.Optimizer):
    def __init__(self, matrix, other, cfg):
        self.muon=torch.optim.Muon(matrix,lr=cfg['lr'],weight_decay=cfg['weight_decay'],
            momentum=.95,nesterov=True,ns_steps=5,eps=1e-7,adjust_lr_fn='original')
        self.other=torch.optim.AdamW(other,lr=1e-4,betas=(.9,.999),eps=1e-8)
        super().__init__([p for g in self.muon.param_groups+self.other.param_groups for p in g['params']],{})
        self.param_groups=self.muon.param_groups+self.other.param_groups
    def step(self,closure=None):
        if closure is not None:raise ValueError('Closure not supported for this registered experiment')
        self.muon.step();self.other.step()
    def state_dict(self):return {'muon':self.muon.state_dict(),'other':self.other.state_dict()}
    def load_state_dict(self,state):
        self.muon.load_state_dict(state['muon']);self.other.load_state_dict(state['other'])
        self.param_groups=self.muon.param_groups+self.other.param_groups

def optimizer_for(model,cfg):
    name=cfg['optimizer']
    if name in {'sgd','adamw'}:raise ValueError('Use the core optimizer factory for SGD/AdamW')
    penalized={n for n,p in penalty_parameters(model)}
    params=list(model.named_parameters())
    groups=[{'params':[p for n,p in params if n in penalized],'weight_decay':cfg['weight_decay']},
            {'params':[p for n,p in params if n not in penalized],'weight_decay':0.}]
    if name=='adam':return torch.optim.Adam(groups,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,decoupled_weight_decay=False)
    if name=='adagrad':return torch.optim.Adagrad(groups,lr=cfg['lr'],lr_decay=0,initial_accumulator_value=0,eps=1e-10)
    if name!='muon':raise ValueError(name)
    matrix=[(n,p) for n,p in params if n.startswith('regressor.') and p.ndim==2]
    assert {n for n,p in matrix}=={'regressor.1.weight','regressor.4.weight'}
    ids={id(p) for n,p in matrix}
    rest=[(n,p) for n,p in params if id(p) not in ids]
    other=[{'params':[p for n,p in rest if n in penalized],'weight_decay':cfg['weight_decay']},
           {'params':[p for n,p in rest if n not in penalized],'weight_decay':0.}]
    assert len(ids)+len(rest)==len(params)
    return HybridMuon([p for n,p in matrix],other,cfg)
