
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class MSELoss(nn.Module):
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
        super(MSELoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

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

        error = (predict - target) ** 2  # N,C,H,W
        class_wise_loss = torch.mean(error, dim=(2, 3))  # N, C
        if class_weight is not None:
            class_wise_loss = class_wise_loss * class_weight

        ## do the reduction for the weighted loss
        loss = self.loss_weight * weight_reduce_loss(
            class_wise_loss, weight, reduction=reduction, avg_factor=avg_factor)
        return loss