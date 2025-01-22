"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.RepulsionLoss import repulsion_loss
from utils.metrics import bbox_inner_iou, shape_iou, wasserstein
from utils.torch_utils import is_parallel
from utils.metrics import bbox_iou, wasserstein_loss


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true

        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(VFLoss, self).__init__()

        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'mean'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (
                true <= 0.0).float()
        loss *= focal_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ComputeLoss:

    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device
        self.model = model
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.slide_ratio = h['slide_ratio']
        if self.slide_ratio > 0:
            BCEcls, BCEobj = SlideLoss(BCEcls), SlideLoss(BCEobj)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]

        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])

        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        iou_ratio = 0.5

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                if self.model.hyp['nwdloss'] > 0:
                    if self.model.hyp['shapeloss'] > 0:
                        nwd = wasserstein(pbox, tbox[i]).squeeze()
                    else:
                        nwd = wasserstein_loss(pbox, tbox[i]).squeeze()
                    lbox += (1 - iou_ratio) * (1.0 - iou).mean() + iou_ratio * (1.0 - nwd).mean()

                    iou = (iou.detach() * (1 - iou_ratio) + nwd.detach() * iou_ratio).clamp(0, 1).type(tobj.dtype)
                else:
                    lbox += (1.0 - iou).mean()
                    iou = iou.detach().clamp(0, 1).type(tobj.dtype)
                if True:
                    sort_id = torch.argsort(iou)

                    b, a, gj, gi, iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], iou[sort_id]

                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou

                auto_iou = iou.mean()

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp

                    if self.slide_ratio > 0:
                        lcls += self.BCEcls(ps[:, 5:], t, auto_iou)
                    else:
                        lcls += self.BCEcls(ps[:, 5:], t)

            if self.slide_ratio > 0 and n:
                obji = self.BCEobj(pi[..., 4], tobj, auto_iou)
            else:
                obji = self.BCEobj(pi[..., 4], tobj)

            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        bs = tobj.shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):

        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],

                            ], device=targets.device).float() * g

        for i in range(self.nl):

            anchors = self.anchors[i]

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets * gain

            if nt:

                r = t[:, :, 4:6] / anchors[:, None]

                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']

                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


import math


class RepLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, sigma=0.5):
        super(RepLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.eps = 1e-7

    def forward(self, gt_boxes, pre_boxes):

        box_iou = self.bbox_iou(gt_boxes, pre_boxes)
        proposal_overlaps = self.bbox_iou(pre_boxes, pre_boxes, xywh=False)

        max_attr, max_attr_index = box_iou.max(dim=0)

        GT_attr = gt_boxes[max_attr_index]

        box_iou[max_attr_index, range(pre_boxes.shape[0])] = 0

        if not box_iou.sum == 0:
            max_rep, max_rep_index = box_iou.max(dim=0)
            GT_rep = gt_boxes[max_rep_index]
            rep_loss = self.Attr(pre_boxes, GT_attr, max_attr) + \
                       self.alpha * self.RepGT(pre_boxes, GT_rep, max_attr) + \
                       self.beta * self.RepBox(proposal_overlaps)
        else:
            rep_loss = self.Attr(GT_attr, pre_boxes, max_attr) + self.beta * self.RepBox(proposal_overlaps)
        return rep_loss

    def Attr(self, gt_boxes, pre_boxes, max_iou):
        Attr_loss = 0
        for index, (gt_box, pre_box) in enumerate(zip(gt_boxes, pre_boxes)):
            Attr_loss += self.SmoothL1(gt_box, pre_box)
        Attr_loss = Attr_loss.sum() / len(gt_boxes)
        return Attr_loss

    def RepGT(self, gt_boxes, pre_boxes, max_iou):
        RepGT_loss = 0
        count = 0
        for index, (gt_box, pre_box) in enumerate(zip(gt_boxes, pre_boxes)):

            count += 1
            IOG = self.RepGT_iog(gt_box, pre_box)
            if IOG > self.sigma:
                RepGT_loss += ((IOG - self.sigma) / ((1 - self.sigma) - math.log(1 - self.sigma))).sum()
            else:
                RepGT_loss += -(1 - IOG).clamp(min=self.eps).log().sum()
        RepGT_loss = RepGT_loss.sum() / count
        return RepGT_loss

    def RepBox(self, overlaps):
        RepBox_loss = 0
        overlap_loss = 0
        count = 0

        for i in range(0, overlaps.shape[0]):
            for j in range(1 + i, overlaps.shape[0]):
                count += 1
                if overlaps[i][j] > self.sigma:
                    RepBox_loss += ((overlaps[i][j] - self.sigma) / ((1 - self.sigma) - math.log(1 - self.sigma))).sum()
                else:
                    RepBox_loss += -(1 - overlaps[i][j]).clamp(min=self.eps).log().sum()
        RepBox_loss = RepBox_loss / count
        return RepBox_loss

    def SmoothL1(self, pred, target, beta=1.0):
        diff = torch.abs(pred - target)
        cond = torch.lt(diff, beta)
        loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss

    def RepGT_iog(self, box1, box2, List=True):
        if List:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        g_area = torch.abs(b2_x2 - b2_x1) * torch.abs(b2_y2 - b2_y1)

        iog = inter / g_area
        return iog

    def bbox_iou(self, bboxes1, bboxes2, xywh=True, eps=1e-7):
        if xywh:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = bboxes1.chunk(4, -1), bboxes2.chunk(4, -1)
            bboxes1[:, 0:1], bboxes1[:, 1:2], bboxes1[:, 2:3], bboxes1[:,
                                                               3:4] = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
            bboxes2[:, 0:1], bboxes2[:, 1:2], bboxes2[:, 2:3], bboxes2[:,
                                                               3:4] = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])

        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1[:, None] + area2 - overlap).clamp(min=eps)

        return ious.clamp(min=eps, max=1)


class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
