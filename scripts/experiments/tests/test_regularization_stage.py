from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.config import resolve,validate,differences
from regularization import FILES,ALLOWED
from training.engine import regularizer,optimizer_for
from train_regularized import weight_statistics


class RegularizationContracts(unittest.TestCase):
    def test_candidates_only_change_regularization(self):
        base=resolve(ROOT/'configs/selected/dropout.json')
        for name in FILES:
            c=validate(resolve(ROOT/'configs/02_training'/name))
            self.assertLessEqual(set(differences(base,c)),ALLOWED)
            cfg=c['training']
            if cfg['penalty']!='none':self.assertEqual(cfg['weight_decay'],0)
            else:self.assertEqual(cfg['penalty_coefficient'],0)
        self.assertEqual(len(FILES),7)

    def test_penalty_gradients_exclude_bias_and_no_double_decay(self):
        for name,expected in [('l1',torch.tensor([[.25,-.25]])),('l2',torch.tensor([[.5,-.75]]))]:
            model=torch.nn.Linear(2,1)
            with torch.no_grad():model.weight.copy_(torch.tensor([[2.,-3.]]))
            cfg={'optimizer':'adamw','lr':.0001,'weight_decay':0.,'penalty':name,'penalty_coefficient':.25}
            loss=regularizer([model.weight],cfg);loss.backward()
            self.assertTrue(torch.equal(model.weight.grad,expected));self.assertIsNone(model.bias.grad)
            opt=optimizer_for(model,cfg);self.assertTrue(all(g['weight_decay']==0 for g in opt.param_groups))

    def test_monitor_does_not_modify_parameters_gradients_or_rng(self):
        model=torch.nn.Linear(2,1)
        model(torch.ones(1,2)).sum().backward()
        params=[p.clone() for p in model.parameters()];grads=[p.grad.clone() for p in model.parameters()]
        rng=torch.get_rng_state().clone();statistics=weight_statistics(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertGreater(statistics['weight_l2'],0)
        for p,old,grad in zip(model.parameters(),params,grads):
            self.assertTrue(torch.equal(p,old));self.assertTrue(torch.equal(p.grad,grad))


if __name__=='__main__':unittest.main()
