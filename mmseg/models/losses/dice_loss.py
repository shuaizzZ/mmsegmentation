
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(DiceLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.smooth = 1e-6

    def forward(self,
                predict,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight).type_as(predict)
        else:
            class_weight = None

        N, C, H, W = predict.size()
        # TODO ignore_index
        # if ignore_index != None:
        #     mask_ignore = target.eq(ignore_index)
        #     target[mask_ignore] = 0

        assert torch.max(target).item() <= C, 'max_id({}) > C({})'.format(torch.max(target).item(), C)
        pt = F.softmax(predict, dim=1) # N,C,H,W
        ## convert target(N,H,W) into onehot vector (N,C,H,W)
        target_onehot = torch.zeros(predict.size()).type_as(target)  # N,C,H,W
        target_onehot.scatter_(1, target.view(N, 1, H, W), 1)  # N,C,H,W

        inter = torch.sum(pt * target_onehot, dim=1)  # N,H,W
        union = torch.sum(pt, dim=1) + torch.sum(target_onehot, dim=1)  # N,H,W
        # TODO version of inter been change
        # if ignore_index != None:
        #     inter = torch.masked_select(inter, mask_ignore==False)
        #     union = torch.masked_select(union, mask_ignore==False)
        dice_coef = torch.mean((2 * torch.sum(inter) + self.smooth) /
                               (torch.sum(union) + self.smooth))

        loss = self.loss_weight * (1 - dice_coef)
        return loss


@LOSSES.register_module()
class CDiceLoss(nn.Module):
    """class-wise DiceLoss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CDiceLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.smooth = 1e-6

    def forward(self,
                predict,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight).type_as(predict)
        else:
            class_weight = None

        N, C, H, W = predict.size()
        pt = F.softmax(predict, dim=1) # N,C,H,W
        ## convert target(N,H,W) into onehot vector (N,C,H,W)
        target_onehot = torch.zeros(predict.size()).type_as(target)  # N,C,H,W
        target_onehot.scatter_(1, target.view(N, 1, H, W), 1)  # N,C,H,W

        intersection = torch.sum(pt * target_onehot, dim=(2, 3))  # N, C
        union = torch.sum(pt.pow(2), dim=(2, 3)) + torch.sum(target_onehot, dim=(2, 3))  # N, C
        ## a^2 + b^2 >= 2ab, target_onehot^2 == target_onehot
        class_wise_loss = (2 * intersection + self.smooth) / (union + self.smooth)  # N, C
        if class_weight is not None:
            class_wise_loss = class_wise_loss * class_weight

        ## do the reduction for the weighted loss
        loss = self.loss_weight * (1 - weight_reduce_loss(
            class_wise_loss, weight, reduction=reduction, avg_factor=avg_factor))
        return loss


@LOSSES.register_module()
class RecallLoss(nn.Module):
    """RecallLoss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(RecallLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.smooth = 1e-6

    def forward(self,
                predict,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight).type_as(predict)
        else:
            class_weight = None

        N, C, H, W = predict.size()
        pt = F.softmax(predict, dim=1)  # N,C,H,W
        ## convert target(N,H,W) into onehot vector (N,C,H,W)
        target_onehot = torch.zeros(predict.size()).type_as(target)  # N,C,H,W
        target_onehot.scatter_(1, target.view(N, 1, H, W), 1)  # N,C,H,W

        true_positive = torch.sum(pt * target_onehot, dim=(2, 3))  # N, C
        total_target = torch.sum(target_onehot, dim=(2, 3))
        ## a^2 + b^2 >= 2ab, target_onehot^2 == target_onehot
        class_wise_loss = (true_positive + self.smooth) / (total_target + self.smooth)  # N, C
        if not class_weight:
            class_wise_loss = class_wise_loss * class_weight

        ## do the reduction for the weighted loss
        loss = self.loss_weight * (1 - weight_reduce_loss(
            class_wise_loss, weight, reduction=reduction, avg_factor=avg_factor))
        return loss


@LOSSES.register_module()
class F1Loss(nn.Module):
    """F1Loss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 rp_weight=[1.0, 1.0],
                 class_weight=None,
                 loss_weight=1.0):
        super(F1Loss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.rp_weight = rp_weight
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.smooth = 1e-6

    def forward(self,
                predict,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = torch.tensor(self.class_weight).type_as(predict)
        else:
            class_weight = None

        N, C, H, W = predict.size()
        _, maxpred = torch.max(predict, 1)
        # convert predict,target (N,H,W) into one hot vector (N,C,H,W)
        predict_onehot = torch.zeros(predict.size()).type_as(maxpred)  # N,C,H,W
        predict_onehot.scatter_(1, maxpred.view(N, 1, H, W), 1)  # N,C,H,W
        target_onehot = torch.zeros(predict.size()).type_as(target)  # N,C,H,W
        target_onehot.scatter_(1, target.view(N, 1, H, W), 1)  # N,C,H,W

        true_positive = torch.sum(predict_onehot * target_onehot, dim=(2, 3))  # N, C
        total_target = torch.sum(target_onehot, dim=(2, 3))
        total_predict = torch.sum(predict_onehot, dim=(2, 3))
        recall = self.rp_weight[0] * (true_positive + self.smooth) / (total_target + self.smooth)
        precision = self.rp_weight[1] * (true_positive + self.smooth) / (total_predict + self.smooth)
        class_wise_loss = (2 * recall * precision) / (recall + precision)  # N, C
        if not class_weight:
            class_wise_loss = class_wise_loss * class_weight

        ## do the reduction for the weighted loss
        loss = self.loss_weight * (1 - weight_reduce_loss(
            class_wise_loss, weight, reduction=reduction, avg_factor=avg_factor))
        return loss