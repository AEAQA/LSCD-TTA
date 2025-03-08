import os
import copy
import functools
import warnings
import random
import math

from pathlib import Path
import torch.nn.functional as F

from framework.ERM import ERM
from framework.loss_and_acc import *
from framework.registry import EvalFuncs, Models

from models.AdaptorHeads import RotationHead, NormHead, NoneHead, Head, JigsawHead
from models.AdaptorHelper import get_new_optimizers, convert_to_target
from models.LAME import laplacian_optimization, kNN_affinity

from utils.tensor_utils import to, AverageMeterDict, zero_and_update
from utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from utils.utils import set_named_submodule, get_named_submodule
from utils.custom_transforms import get_tta_transforms

from .cmp_losses import Losses

warnings.filterwarnings("ignore")
np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "{:.4f},  ".format(x)))


class AdaMixBN(nn.BatchNorm2d):
    """
    AdaMixBN in DomainAdaptor (https://github.com/koncle/DomainAdaptor)
    """
    # AdaMixBn cannot be applied in an online manner.
    def __init__(self, in_ch, lambd=None, transform=True, mix=True, idx=0):
        super(AdaMixBN, self).__init__(in_ch)
        self.lambd = lambd
        self.rectified_params = None
        self.transform = transform
        self.layer_idx = idx
        self.mix = mix

    def get_retified_gamma_beta(self, lambd, src_mu, src_var, cur_mu, cur_var):
        C = src_mu.shape[1]
        new_gamma = (cur_var + self.eps).sqrt() / (lambd * src_var + (1 - lambd) * cur_var + self.eps).sqrt() * self.weight.view(1, C, 1, 1)
        new_beta = lambd * (cur_mu - src_mu) / (cur_var + self.eps).sqrt() * new_gamma + self.bias.view(1, C, 1, 1)
        return new_gamma.view(-1), new_beta.view(-1)

    def get_lambd(self, x, src_mu, src_var, cur_mu, cur_var):
        instance_mu = x.mean((2, 3), keepdims=True)
        instance_std = x.std((2, 3), keepdims=True)

        it_dist = ((instance_mu - cur_mu) ** 2).mean(1, keepdims=True) + ((instance_std - cur_var.sqrt()) ** 2).mean(1, keepdims=True)
        is_dist = ((instance_mu - src_mu) ** 2).mean(1, keepdims=True) + ((instance_std - src_var.sqrt()) ** 2).mean(1, keepdims=True)
        st_dist = ((cur_mu - src_mu) ** 2).mean(1)[None] + ((cur_var.sqrt() - src_var.sqrt()) ** 2).mean(1)[None]

        src_lambd = 1 - (st_dist) / (st_dist + is_dist + it_dist)

        src_lambd = torch.clip(src_lambd, min=0, max=1)
        return src_lambd

    def get_mu_var(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)

        lambd = self.get_lambd(x, src_mu, src_var, cur_mu, cur_var).mean(0, keepdims=True)

        if self.lambd is not None:
            lambd = self.lambd

        if self.transform:
            if self.rectified_params is None:
                new_gamma, new_beta = self.get_retified_gamma_beta(lambd, src_mu, src_var, cur_mu, cur_var)
                # self.test(x, lambd, src_mu, src_var, cur_mu, cur_var, new_gamma, new_beta)
                self.weight.data = new_gamma.data
                self.bias.data = new_beta.data
                self.rectified_params = new_gamma, new_beta
            return cur_mu, cur_var
        else:
            new_mu = lambd * src_mu + (1 - lambd) * cur_mu
            new_var = lambd * src_var + (1 - lambd) * cur_var
            return new_mu, new_var

    def forward(self, x):
        n, C, H, W = x.shape
        new_mu = x.mean((0, 2, 3), keepdims=True)
        new_var = x.var((0, 2, 3), keepdims=True)

        if self.training:
            if self.mix:
                new_mu, new_var = self.get_mu_var(x)

            # Normalization with new statistics
            inv_std = 1 / (new_var + self.eps).sqrt()
            new_x = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
            return new_x
        else:
            return super(AdaMixBN, self).forward(x)

    def reset(self):
        self.rectified_params = None

    def test_equivalence(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)
        lambd = 0.9

        new_gamma, new_beta = self.get_retified_gamma_beta(x, lambd, src_mu, src_var, cur_mu, cur_var)
        inv_std = 1 / (cur_var + self.eps).sqrt()
        x_1 = (x - cur_mu) * (inv_std * new_gamma.view(1, C, 1, 1)) + new_beta.view(1, C, 1, 1)

        new_mu = lambd * src_mu + (1 - lambd) * cur_mu
        new_var = lambd * src_var + (1 - lambd) * cur_var
        inv_std = 1 / (new_var + self.eps).sqrt()
        x_2 = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
        assert (x_2 - x_1).abs().mean() < 1e-5
        return x_1, x_2


class EntropyMinimizationHead(Head):
    KEY = 'EM'
    ft_steps = 1

    def __init__(self, num_classes, in_ch, args):
        super(EntropyMinimizationHead, self).__init__(num_classes, in_ch, args)
        self.losses = Losses()

    def get_cos_logits(self, feats, backbone):
        w = backbone.fc.weight  # c X C
        w, feats = F.normalize(w, dim=1), F.normalize(feats, dim=1)
        logits = (feats @ w.t())  # / 0.07
        return logits

    def label_rectify(self, feats, logits, thresh=0.95):
        # mask = self.get_confident_mask(logits, thresh=thresh)
        max_prob = logits.softmax(1).max(1)[0]
        normed_feats = feats / feats.norm(dim=1, keepdim=True)
        # N x N
        sim = (normed_feats @ normed_feats.t()) / 0.07
        # sim = feats @ feats.t()
        # select from high confident masks
        selected_sim = sim  # * max_prob[None]
        # N x n @ n x C = N x C
        rectified_feats = (selected_sim.softmax(1) @ feats)
        return rectified_feats + feats

    def do_lame(self, feats, logits):
        prob = logits.softmax(1)
        unary = - torch.log(prob + 1e-10)  # [N, K]

        feats = F.normalize(feats, p=2, dim=-1)  # [N, d]
        kernel = kNN_affinity(5)(feats)  # [N, N]

        kernel = 1 / 2 * (kernel + kernel.t())

        # --- Perform optim ---
        Y = laplacian_optimization(unary, kernel)
        return Y

    def do_ft(self, backbone, x, label, source_logits=None, source_feats=None, source_model=None, target_model=None, 
              loss_name=None, step=0, model=None, history_size=10, **kwargs):
        assert loss_name is not None

        if loss_name.lower() == 'gem-aug':
            with torch.no_grad():
                aug_x = kwargs['tta']
                n, N, C, H, W = aug_x.shape
                aug_x = aug_x.reshape(n * N, C, H, W)
                aug_logits = backbone(aug_x)[-1].view(n, N, -1).mean(1)
        else:
            aug_logits = None

    
        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))
        ret = {
            'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            'logits': logits.detach()
        }

        if loss_name.lower() == 'tribe':
            tribe_loss, source_model, target_model = self.process_tribe(
                x, label, logits=logits, source_logits=source_logits, source_model=source_model, target_model=target_model, 
            )
            ret.update({'tribe': {'loss': tribe_loss}})
            return ret
        
        ret.update(self.losses.get_loss(loss_name, 
                                        logits=logits, feats=feats, source_logits=source_logits, source_feats=source_feats, 
                                        source_model=source_model, target_model=target_model, backbone=backbone, 
                                        step=step, aug_logits=aug_logits))
        
        return ret

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))
        res = {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': logits, 'target': label},
            'logits': logits,
            'feats': feats,
        }
        if self.args.LAME:
            res.update({'LAME': {'acc_type': 'acc', 'pred': self.do_lame(feats, logits), 'target': label}})
        return res

    def setup(self, model, online):
        model.backbone.train()
        lr = self.args.lr
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(model, lr=lr, names=['bn'], opt_type='sgd', momentum=self.args.online),
        ]

    def process_tribe(self, x, label, logits, source_model, target_model):
        def build_ema(model):
            ema_model = copy.deepcopy(model)
            for param in ema_model.parameters():
                param.detach_()
            return ema_model

        def configure_model(model_, num_class):
            eta = 0.01
            gamma = 0
            model = copy.deepcopy(model_)
            model.requires_grad_(False)
            normlayer_names = []

            for name, sub_module in model.named_modules():
                if isinstance(sub_module, nn.BatchNorm2d) or isinstance(sub_module, nn.BatchNorm1d):
                    normlayer_names.append(name)
                    
            for name in normlayer_names:
                bn_layer = get_named_submodule(model, name)
                if isinstance(bn_layer, nn.BatchNorm2d):
                    NewBN = BalancedRobustBN2dV5
                    # NewBN = BalancedRobustBN2dEMA
                elif isinstance(bn_layer, nn.BatchNorm1d):
                    NewBN = BalancedRobustBN1dV5
                else:
                    raise RuntimeError()
                
                momentum_bn = NewBN(bn_layer,
                                    num_class,
                                    eta,
                                    gamma,
                                    )
                momentum_bn.requires_grad_(True)
                set_named_submodule(model, name, momentum_bn)
            
            return model

        
        def self_softmax_entropy(x):
            return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)

        def update_teacher(model, ema_model):
            ema_decay = 0.99
            with torch.no_grad():
                for model_params, ema_model_params in zip(model.parameters(), ema_model.parameters()):
                    ema_model_params.data = ema_decay * ema_model_params.data + (1.0 - ema_decay) * model_params.data
            return ema_model
        
        def set_bn_label(model, label=None):
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, BalancedRobustBN1dV5) or isinstance(sub_module, BalancedRobustBN2dV5) or isinstance(sub_module, BalancedRobustBN2dEMA):
                    sub_module.label = label
            return
    
        def update_tribe_model(real_source_model, target_model, aux_model, batch_data, logits):
            h0 = 0.05
            num_class = logits.shape[1]
            eta = 0.01
            gamma = 0
            weight_lambda = 0.5

            model = copy.copy(target_model)

            p_l = logits.argmax(dim=1)
            aux_model.train()
            model.train()

            # # don't use aug cause did not use in training
            # tta_transformers = get_tta_transforms(n_pixels=batch_data['x'].shape[-1])
            # strong_sup_aug = tta_transformers(batch_data['x'])
            
            set_bn_label(model, p_l)
            set_bn_label(aux_model, p_l)

            # stu_sup_out = model(strong_sup_aug, batch_data['label'], train_mode='test')
            stu_sup_out = model(**batch_data, train_mode='test')
            ema_sup_out = aux_model(**batch_data, train_mode='test')


            entropy = self_softmax_entropy(ema_sup_out['logits'])
            entropy_mask = (entropy < h0 * math.log(num_class))

            l_sup = torch.nn.functional.cross_entropy(stu_sup_out['logits'], ema_sup_out['logits'].argmax(dim=-1), reduction='none')[entropy_mask].mean()

            with torch.no_grad():
                set_bn_label(real_source_model, p_l)
                source_anchor_dict = real_source_model(**batch_data, train_mode='test')
                source_anchor = source_anchor_dict['logits'].detach()
            
            l_reg = weight_lambda* torch.nn.functional.mse_loss(ema_sup_out['logits'], source_anchor, reduction='none')[entropy_mask].mean()

            l = (l_sup + l_reg)

            aux_model = update_teacher(model, aux_model)
            aux_model.eval()
            model.eval()

            return l, aux_model, model
        

        batch_data = {
            'x': x,
            'label': label,
        }
        num_class = logits.shape[1]
        global source_model_bak
        global device

        batch_data = to(batch_data, device)
        
        source_model_ = configure_model(source_model_, num_class)
        target_model_ = configure_model(target_model, num_class)

        # aux_model = build_ema(source_model_)
        # for (name1, param1), (name2, param2) in zip(target_model.named_parameters(), aux_model.named_parameters()):
        #     set_named_submodule(aux_model, name2, param1)
        # real_source_model = build_ema(source_model_bak)

        aux_model = copy.deepcopy(source_model_)
        real_source_model = copy.deepcopy(source_model_bak)

        tribe_loss, source_model_, target_model_ = update_tribe_model(real_source_model, target_model, aux_model, batch_data, logits)

        return tribe_loss, source_model_, target_model_


@Models.register('DomainAdaptor')
class DomainAdaptor(ERM):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(DomainAdaptor, self).__init__(num_classes, pretrained, args)
        heads = {
            'em': EntropyMinimizationHead,
            'rot': RotationHead,
            'norm': NormHead,
            'none': NoneHead,
            'jigsaw': JigsawHead,
        }
        self.head = heads[args.TTA_head.lower()](num_classes, self.in_ch, args)
        if args.AdaMixBN == True:
            self.bns = list(convert_to_target(self.backbone, functools.partial(AdaMixBN, transform=args.Transform, lambd=args.mix_lambda),
                                              verbose=False, start=0, end=5, res50=args.backbone == 'resnet50')[-1].values())

    def step(self, x, label, source_logits=None, source_feats=None, source_model=None, target_model=None, train_mode='test', **kwargs):
        # print(train_mode)
        if train_mode == 'train':
            res = self.head.do_train(self.backbone, x, label, model=self, **kwargs)
        elif train_mode == 'test':
            res = self.head.do_test(self.backbone, x, label, model=self, **kwargs)
        elif train_mode == 'ft':
            res = self.head.do_ft(self.backbone, x, label, source_logits=source_logits, source_feats=source_feats, 
                                  source_model=source_model, target_model=target_model, model=self, **kwargs)
        else:
            raise Exception("Unexpected mode : {}".format(train_mode))
        return res
    
    def finetune(self, data, optimizers, loss_name, running_loss=None, running_corrects=None, source_logits=None, source_feats=None, source_model=None, target_model=None):
        if hasattr(self, 'bns'):
            [bn.reset() for bn in self.bns]

        with torch.enable_grad():
            res = None
            for i in range(self.head.ft_steps):  # ft_steps==1
                o = self.step(**data, source_logits=source_logits, source_feats=source_feats, 
                              source_model=source_model, target_model=target_model, train_mode='ft',
                              step=i, loss_name=loss_name)
                meta_train_loss = get_loss_and_acc(o, running_loss, running_corrects, prefix=f'A{i}_')
                zero_and_update(optimizers, meta_train_loss)
                if i == 0:
                    res = o
            return res

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def setup(self, online):
        return self.head.setup(self, online)


@EvalFuncs.register('tta_ft')
def test_time_adaption(model, eval_data, lr, epoch, args, engine, mode, tSNE_flag=None):
    global device
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()

    model.eval()
    model_to_ft = copy.deepcopy(model)
    
    global source_model_bak
    source_model = copy.deepcopy(model)
    source_model_bak = copy.deepcopy(source_model)
    target_model = model_to_ft

    original_state_dict = model.state_dict()

    online = args.online
    optimizers = model_to_ft.setup(online)

    loss_names = args.loss_names

    with torch.no_grad():
        total_data = int(args.batch_size) * len(eval_data)
        for i, data in enumerate(eval_data):

            data = to(data, device)
            # Normal Test
            out = model(**data, train_mode='test')
            source_logits = out['logits']
            source_feats = out['feats']
            del out['feats']
            get_loss_and_acc(out, running_loss, running_corrects, prefix='original_')
            
            # test-time adaptation to a single batch
            for loss_name in loss_names:
                # recover to the original weight
                # model_to_ft.load_state_dict(original_state_dict) if (not online) else ""

                if not online:
                    model_to_ft.load_state_dict(original_state_dict)

                if online or (loss_name in ['gem-t', 'gem-skd', 'gem-aug']):
                    # adapt to the current batch
                    adapt_out = model_to_ft.finetune(data, optimizers, loss_name, running_loss, running_corrects, 
                                                     source_logits=source_logits, source_feats=source_feats, 
                                                     source_model=source_model, target_model=target_model)

                # get the adapted result
                cur_out = model_to_ft(**data, train_mode='test')
                del cur_out['feats']

                get_loss_and_acc(cur_out, running_loss, running_corrects, prefix=f'{loss_name}_')
                if loss_name == loss_names[-1]:
                    get_loss_and_acc(cur_out, running_loss, running_corrects)  # the last one is recorded as the main result

    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    return acc['main'], (loss, acc)