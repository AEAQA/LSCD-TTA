
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

HISTORY_SIZE=10

class Adaptive_Region_Specific_TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-5, num_region_per_axis=(16, 16), do_bg=True, batch_dice=True, A=0.3, B=0.4, apply_nonlin=True):
        """
        num_region_per_axis: the number of boxes of each axis in (x, y)
        2D num_region_per_axis's axis in (x, y)
        """
        super(Adaptive_Region_Specific_TverskyLoss, self).__init__()
        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.dim = len(num_region_per_axis)
        assert self.dim == 2, "Only 2D case is supported here."
        self.pool = nn.AdaptiveAvgPool2d(num_region_per_axis)

        self.A = A
        self.B = B
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y):
        # 2D: [batchsize, class_num]
        if self.apply_nonlin:
            x = torch.softmax(x, dim=1)

        shp_x, shp_y = x.shape, y.shape
        assert len(shp_x) == 2 and len(shp_y) == 2, "Expected 2D tensors for x and y, got shapes: {} and {}".format(shp_x, shp_y)

        if not self.do_bg:
            x = x[:, 1:]

        with torch.no_grad():
            # Convert y to one-hot encoding if necessary
            pseudo_labels = y.argmax(dim=1)
            y_onehot = torch.zeros_like(y).scatter_(1, pseudo_labels.unsqueeze(1), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # Expand x and y_onehot to 4D to simulate spatial dimensions (1x1)
        x = x.unsqueeze(2).unsqueeze(3)  # [batch_size, class_num, 1, 1]
        y_onehot = y_onehot.unsqueeze(2).unsqueeze(3)  # [batch_size, class_num, 1, 1]

        # Calculate true positives, false positives, and false negatives
        tp = x * y_onehot
        fp = x * (1 - y_onehot)
        fn = (1 - x) * y_onehot

        # Apply region-based pooling
        region_tp = self.pool(tp)
        region_fp = self.pool(fp)
        region_fn = self.pool(fn)

        if self.batch_dice:
            region_tp = region_tp.sum(0)
            region_fp = region_fp.sum(0)
            region_fn = region_fn.sum(0)

        # Tversky Loss calculation
        alpha = self.A + self.B * (region_fp + self.smooth) / (region_fp + region_fn + self.smooth)
        beta = self.A + self.B * (region_fn + self.smooth) / (region_fp + region_fn + self.smooth)

        region_tversky = (region_tp + self.smooth) / (region_tp + alpha * region_fp + beta * region_fn + self.smooth)
        region_tversky = 1 - region_tversky

        region_tversky = region_tversky.mean()

        return region_tversky


class Losses():
    def __init__(self):
        self.history_size = HISTORY_SIZE
        self.history_losses = []
        self.history_diff = []

        self.slr_history_losses = []
        self.em_history_losses = []

        self.slr_history_diff = []
        self.em_history_diff = []

        self.losses = {
            'em': self.em,
            'slr': self.slr,
            'norm': self.norm,
            'gem-t': self.GEM_T,
            'gem-skd': self.GEM_SKD,
            'gem-aug': self.GEM_Aug,
            'conf':self.conf_loss,
            'shot':self.shot,
            'cbst':self.cbst,
            'pseudo':self.pseudo,
            'focal':self.focal,
            'ce':self.ce,
            'kl':self.KL_loss,
            'adda':self.adversarial_loss,
            'smooth_focal':self.smooth_focal,
            'cos':self.cos_sim_loss,
            'dice':self.diceloss,
            'bse':self.balanced_softmax_entropy,
            'sse': self.soft_softmax_entropy,
            'ncse':self.neg_conf_softmax_entropy,
            'lscd':self.lscd,
            'cscada':self.cscada,
            'ars':self.ars,

        }

    def adjust_loss(self, loss):
        mean_loss = torch.tensor(self.history_losses).mean()
        mean_diff = torch.tensor(self.history_diff).mean()
        diff = torch.abs(loss - mean_loss)

        if diff > torch.abs(mean_diff):
            weighted_loss = (loss + mean_loss) / 2
        else:
            weighted_loss = loss

        self.update_history_losses(loss,diff)

        return weighted_loss

    def update_history_losses(self, loss, diff):
        self.history_losses.append(loss)
        self.history_diff.append(diff)
        if len(self.history_losses) > self.history_size:
            self.history_losses.pop(0)
        if len(self.history_diff) > self.history_size:
            self.history_diff.pop(0)


    def get_loss(self, name, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):

        loss = self.losses[name.lower()](logits, feats, source_logits=source_logits, source_feats=source_feats, 
                                         source_model=source_model, target_model=target_model, **kwargs)
        
        # loss = self.adjust_loss(loss)
        return {name: {'loss': loss}}
    
    def GEM_T(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        T = logits.std(1, keepdim=True).detach() * 2
        prob = (logits / T).softmax(1)
        loss = - ((prob * prob.log()).sum(1) * (T ** 2)).mean()
        return loss

    def GEM_SKD(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        T = logits.std(1, keepdim=True).detach() * 2

        original_prob = logits.softmax(1)
        prob = (logits / T).softmax(1)

        loss = - ((original_prob.detach() * prob.log()).sum(1) * (T ** 2)).mean()
        return loss

    def GEM_Aug(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        T = logits.std(1, keepdim=True).detach() * 2
        aug_logits = kwargs['aug_logits']
        loss = - ((aug_logits.softmax(1).detach() * (logits / T).softmax(1).log()).sum(1) * (T ** 2)).mean()
        return loss

    def em(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        prob = (logits).softmax(1)
        
        # prob_s = (source_logits).softmax(1)
        # loss = (- prob_s * prob.log()).sum(1).mean()

        loss = (- prob * prob.log()).sum(1).mean()
        return loss
    
    def focal(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        gamma = 1
        prob = (logits).softmax(1)

        # log_prob = F.log_softmax(logits, dim=1)
        # ce_loss = -log_prob
        ce_loss = - prob * prob.log()
        
        focal_weight = (1 - prob).pow(gamma)
        
        loss = (focal_weight * ce_loss).sum(1).mean()
        return loss
    
    def slr(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        prob = (logits).softmax(1)
        loss = -(prob * (prob / (1 - prob + 1e-8)).log()).sum(1).mean()
        return loss  # * 3 is enough = 82.7
   
    def norm(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        logits = (logits).softmax(1)
        return -logits.norm(dim=1).mean() * 2

    def conf_loss(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        confidence_loss = F.softmax(logits, dim=1).max(dim=1)[0].mean()
        return confidence_loss

    def cbst(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        threshold = 0.1
        prob = (logits).softmax(1)
        ignore_index = -1
        batch_size = prob.size(0) // 2
        logits = logits[batch_size:]
        prob = prob[batch_size:]

        maxpred = torch.argmax(prob.detach(), dim=1)
        mask = (maxpred > threshold)

        label = torch.where(mask, maxpred, torch.ones(1).to(prob.device, dtype=torch.long)*ignore_index)
        
        loss = F.cross_entropy(logits, label, ignore_index=ignore_index)

        return loss

    def shot(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        # (1) entropy
        ent_loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean(0)

        # (2) diversity
        softmax_out = F.softmax(logits, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-8))

        # (3) pseudo label
        # adapt
        py, y_prime = F.softmax(logits, dim=-1).max(1)
        flag = py > 0.1
        clf_loss = F.cross_entropy(logits[flag], y_prime[flag])

        loss = ent_loss + 0.1*clf_loss
        return loss

    def ce(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        prob_value_list, prob_index_list = F.softmax(logits, dim=-1).max(1)
        flag = prob_value_list > 0.1
        clf_loss = F.cross_entropy(logits[flag], prob_index_list[flag])
        loss = clf_loss

        return loss
    
    def pseudo(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        prob_value_list, prob_index_list = F.softmax(logits, dim=-1).max(1)
        loss = prob_value_list.mean()

        return loss

    def KL_loss(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_labels.unsqueeze(1), 1)

        kl_loss = F.kl_div(prob.log(), target_one_hot, reduction='batchmean')
        loss = kl_loss

        return loss


    def adversarial_loss(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        fake_pred = logits
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_labels.unsqueeze(1), 1)

        g_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

        real_pred = target_one_hot

        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))

        d_loss = (real_loss + fake_loss) / 2

        return d_loss


    def cos_sim_loss(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        def compute_class_prototypes(feats, pseudo_labels, num_classes):
            class_prototypes = []
            for i in range(num_classes):
                class_feats = feats[pseudo_labels == i]
                if class_feats.size(0) > 0:
                    class_prototype = class_feats.mean(dim=0)
                else:
                    class_prototype = torch.zeros(feats.size(1)).to(feats.device)
                class_prototypes.append(class_prototype)
            
            return torch.stack(class_prototypes)
        
        label_smoothing_args = 1e-2
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)
        
        num_classes = logits.size(1)
        
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)
        
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_labels.unsqueeze(1), 1)

        class_prototypes = compute_class_prototypes(feats, pseudo_labels, num_classes)

        cos_sim_loss = 0
        for i in range(num_classes):
            class_feats = feats[pseudo_labels == i]
            if class_feats.size(0) > 0:
                cos_sim = F.cosine_similarity(class_feats, class_prototypes[i].unsqueeze(0), dim=1)
                cos_sim_loss += (1 - cos_sim.mean())

        loss = cos_sim_loss
        return loss
    
    
    def sommth_pseudo(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        label_smoothing_args=1e-2
        prob = logits.softmax(dim=1)
        
        pseudo_labels = prob.argmax(dim=1)
        
        num_classes = logits.size(1)
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)

        loss = (-soft_one_hot * prob.log()).sum(dim=1).mean()
        return loss


    def balanced_soft_ce(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        """
        bsce
        """
        label_smoothing_args=1e-2  # better with hard label, however for a better writing
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)
        
        num_classes = logits.size(1)
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)

        balance_soft_one_hot = (1-soft_one_hot) * (1-prob) + soft_one_hot * prob
        loss = -(torch.exp((balance_soft_one_hot)) * soft_one_hot * prob.log()).sum(dim=1).mean()

        return loss

    def balanced_softmax_entropy(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        """
        bse
        """
        label_smoothing_args=1e-2  # better with hard label, however for a better writing
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)
        
        num_classes = logits.size(1)
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)

        balance_soft_one_hot = (1 - soft_one_hot) * (1 - prob) + soft_one_hot * prob
        loss = -(torch.exp((balance_soft_one_hot)) * prob * prob.log()).sum(dim=1).mean()

        return loss

    def smooth_focal(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        label_smoothing_args = 1e-2
        alpha = 2
        gamma = 0.5

        prob = logits.softmax(dim=1)
        prob_value_list, prob_index_list = F.softmax(logits, dim=-1).max(1)
        pseudo_labels = prob.argmax(dim=1)

        # Step 2: Calculate the target one-hot encoding with label smoothing
        num_classes = logits.size(1)
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)
        
        # Step 3: Calculate Focal Loss with pseudo-labels
        p_t = (soft_one_hot * prob) + ((1 - soft_one_hot) * (1 - prob))
        wcce_loss = -alpha * (1-p_t)**gamma * soft_one_hot * prob.log()
        loss = wcce_loss.sum(dim=1).mean()
        
        # Step 4: Return the average loss across the batch
        return loss
    
    def soft_softmax_entropy(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        """
        sse
        """
        label_smoothing_args = 1e-1
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)

        num_classes = logits.size(1)
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)

        balance_soft_one_hot = (1-soft_one_hot) * (1-prob) + soft_one_hot * prob

        sse_loss = -(torch.exp(balance_soft_one_hot) * prob**0.5 * (prob).log()).sum(dim=1).mean()
        
        loss = sse_loss
        return loss

    def diceloss(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        # probs = torch.sigmoid(logits)
        prob = logits.softmax(dim=1)
        pseudo_labels = prob.argmax(dim=1)
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_labels.unsqueeze(1), 1)

        smooth = 0

        intersection = (prob * target_one_hot).sum(dim=1).mean()
        dice = (2 * intersection + smooth) / (prob.sum(dim=1).mean() + target_one_hot.sum(dim=1).mean() + smooth)
        return 1 - dice

    def neg_conf_softmax_entropy(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs): 
        """
        ncse
        """
        label_smoothing_args = 1e-2

        prob = logits.softmax(dim=1)
        
        pseudo_labels = prob.argmax(dim=1)
        num_classes = logits.size(1)
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_labels.unsqueeze(1), 1)

        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_labels.unsqueeze(1), 1.0 - label_smoothing_args)


        beta = 1
        neg_inds = target_one_hot.lt(1).float()
        neg_weights = torch.pow(1 - soft_one_hot, beta)

        neg_conf_prob = neg_inds * neg_weights
        loss = - (torch.exp(neg_conf_prob) * prob**0.5 * (prob).log()).sum(dim=1).mean()
        
        return loss

    def cscada(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        temperature = 0.07
        lambda_unsupervised = 1.0
        ema_decay = 0.99

        def unsupervised_loss(logits, source_logits):
            """
            无监督损失：计算目标域模型与源域模型输出之间的连续性损失（MSE）。
            """
            normalized_logits = F.normalize(logits, p=2, dim=-1)
            normalized_source_logits = F.normalize(source_logits, p=2, dim=-1)
            mse_loss = F.mse_loss(normalized_logits, normalized_source_logits)
            return mse_loss

        def contrastive_loss(source_feats, target_feats):
            """
            计算源域到目标域以及目标域到源域的对比损失（L_ct）。
            """
            global device

            # 归一化处理源域和目标域特征
            normalized_source_feats = F.normalize(source_feats, p=2, dim=-1)
            normalized_target_feats = F.normalize(target_feats, p=2, dim=-1)

            # 计算源域到目标域的对比损失
            
            similarity_source_to_target = torch.matmul(normalized_source_feats, normalized_target_feats.T) / temperature
            labels = torch.arange(source_feats.size(0)).cuda()
            similarity_source_to_target = similarity_source_to_target.to(device)
            labels = labels.to(device)
            contrastive_loss_source_to_target = F.cross_entropy(similarity_source_to_target, labels)

            # 计算目标域到源域的对比损失
            similarity_target_to_source = torch.matmul(normalized_target_feats, normalized_source_feats.T) / temperature
            contrastive_loss_target_to_source = F.cross_entropy(similarity_target_to_source, labels)

            return 0.5 * (contrastive_loss_source_to_target + contrastive_loss_target_to_source)

        def update_teacher(model, ema_model):
            with torch.no_grad():
                for model_params, ema_model_params in zip(model.parameters(), ema_model.parameters()):
                    ema_model_params.data = ema_decay * ema_model_params.data + (1.0 - ema_decay) * model_params.data
            return ema_model

        if source_model is None or target_model is None:
            raise ValueError("source_model and target_model must be initialized globally.")

        ema_model = copy.copy(source_model)  # 浅拷贝一起更新

        supervised_loss = self.ce(logits, feats, source_logits, **kwargs)

        unsup_loss = unsupervised_loss(logits, source_logits)

        contrastive_loss_value = contrastive_loss(feats, source_feats)

        ema_model = update_teacher(target_model, ema_model)

        cscada_loss = supervised_loss + lambda_unsupervised * unsup_loss + 0.1 * contrastive_loss_value
        return cscada_loss

    def ars(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        prob = logits.softmax(dim=1)
        pseudo_inds = prob.argmax(dim=1)
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_inds.unsqueeze(1), 1)

        ars_loss = Adaptive_Region_Specific_TverskyLoss(num_region_per_axis=(16, 16), apply_nonlin=False)
        # prob/target_one_hot shape: [batch_size, num_class]
        cal_ars_loss = ars_loss(prob, target_one_hot)
        
        return cal_ars_loss

    def lscd(self, logits, feats, source_logits=None, source_feats=None, source_model=None, target_model=None, **kwargs):
        label_smoothing_args = 1e-2
        prob = logits.softmax(dim=1)
        pseudo_inds = prob.argmax(dim=1)
        
        num_classes = logits.size(1)
        soft_one_hot = torch.zeros_like(prob)
        soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
        soft_one_hot.scatter_(1, pseudo_inds.unsqueeze(1), 1.0 - label_smoothing_args)
        target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_inds.unsqueeze(1), 1)

        balance_soft_one_hot = (1 - soft_one_hot) * (1 - prob) + soft_one_hot * prob

        # ncse
        beta = 1
        neg_inds = target_one_hot.lt(1).float()
        neg_weights = torch.pow(1 - soft_one_hot, beta)
        neg_conf_prob = neg_inds * neg_weights
        ncse_loss = - (torch.exp(neg_conf_prob) * prob**0.5 * (prob + 1e-8).log()).sum(dim=1).mean()

        # bse
        bse_loss = - (torch.exp((balance_soft_one_hot)) * prob * (prob + 1e-8).log()).sum(dim=1).mean()

        # wcce(not use)
        gamma = 1
        wcce_loss = (((1 - prob).pow(gamma))*(- prob * prob.log())).sum(1).mean(0)

        # div (for stable)
        msoftmax = prob.mean(dim=0)
        diversity_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-8))

        # lsd
        """
        Low Saturation Divesity
        """
        lsd_arg = 1e-3      # alpha
        div_arg = 4     # beta
        lsd_loss = -(prob * (1 / (1 - prob + lsd_arg)).log()).sum(dim=1).mean()
        merge_lsd_loss = lsd_loss + div_arg * diversity_loss

        lscd_loss = 0.25 * ncse_loss + 1.0 * bse_loss + 1.5 * merge_lsd_loss

        return lscd_loss

