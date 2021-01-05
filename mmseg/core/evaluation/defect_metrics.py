
import time
import threading

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F


def points_2_box(points):
    ymin, ymax = points[0].min(), points[0].max()
    xmin, xmax = points[1].min(), points[1].max()
    return xmin, ymin, xmax, ymax

def points_2_area(points):
    area = len(points[0])
    return area

def points_2_minrect(points):
    rect = cv2.minAreaRect(np.array(points).transpose())
    return rect

## --------------- segmentation metric implemented by zhangshuai --------------- ##
class SegmentationMetric(object):
    """Computes pixAcc, mIoU, recall, precision, F1 metric scroes
    """
    def __init__(self, nclass, f1_cfg, ignore_index=[]):
        self.nclass = nclass
        self.com_f1 = f1_cfg.com_f1
        self.metric_type = f1_cfg.type
        assert self.metric_type in ['pix_iou', 'pix_iof', 'box_iou', 'box_iof']
        self.metric_threshold = f1_cfg.threshold
        assert len(self.metric_threshold) >= self.nclass
        self.filter_type = f1_cfg.defect_filter.type
        assert self.filter_type in ['none', 'box', 'area', 'minRect']
        filter_size = f1_cfg.defect_filter.size
        filter_size = filter_size if mmcv.is_list_of(filter_size, list) else [filter_size]
        self.filter_size = filter_size * self.nclass if len(filter_size) == 1 else filter_size
        assert len(self.filter_size) == self.nclass
        if ignore_index:
            self.ignore_index = ignore_index if isinstance(ignore_index, list) else [ignore_index]
        else:
            self.ignore_index = []

        self.smooth = np.spacing(1)
        self.smooth_n = np.array([self.smooth] * self.nclass)
        self.pixel_fields = ['label', 'inter', 'union']
        self.defect_fields = ['target', 'tp', 'predict']

    def batch_pixelAcc_iou(self, predicts, targets):
        """update_pixelAcc_iou.
        :param predicts: 3D nparray
        :param targets: 3D nparray
        :return: update batch pixelAcc and iou
        """
        for c in range(0, self.nclass):
            if c in self.ignore_index:
                continue
            class_predict_mask = predicts == c
            class_target_mask = targets == c

            self.class_pixel_info['label'][c] += np.count_nonzero(class_target_mask)
            self.class_pixel_info['inter'][c] += np.count_nonzero(class_target_mask & class_predict_mask)
            self.class_pixel_info['union'][c] += np.count_nonzero(class_target_mask | class_predict_mask)

    def f1_defect_filter(self, points_pre, pre_id):
        """
        Take predict as an example to filter defects by size.
        :param points_pre: 2D array
        :param pre_id: int
        :return: filter_flag: bool
        """
        filter_flag = False
        if self.filter_type == 'none':
            return filter_flag

        filter_size = self.filter_size[pre_id]
        if self.filter_type == 'box':
            pre_xmin, pre_ymin, pre_xmax, pre_ymax = points_2_box(points_pre)
            if pre_ymax - pre_ymin <= filter_size[0] and pre_xmax - pre_xmin <= filter_size[1]:
                filter_flag = True
        elif self.filter_type == 'area':
            defect_area = points_2_area(points_pre)
            if defect_area <= filter_size[0]:
                filter_flag = True
        elif self.filter_type == 'minRect':
            rect = points_2_minrect(points_pre)
            ((cx, cy), (w, h), theta) = rect
            long = max(h, w)
            short = min(h, w)
            if long <= filter_size[0] and short <= filter_size[1]:
                filter_flag = True
        return filter_flag

    def oneimg_tp_pre_tar(self, predict, target):
        """ Calculate image-wise defect nums (tp, predict, target).
        :param predict: 2D nparray, (h, w).
        :param target: 2D nparray, (h, w).
        :return:
        """
        ## 不计算predict和target都是无缺陷的情况
        # if np.count_nonzero(one_predict) + np.count_nonzero(one_target) == 0:
        if cv2.countNonZero(predict) + cv2.countNonZero(target) == 0:
            return
        ## 初始化
        tp_total_num = 0
        tp_class_num = np.zeros((self.nclass,))
        ## 计算连通域
        pre_total_num, pre_labels = cv2.connectedComponents(predict, connectivity=8)
        tar_total_num, tar_labels = cv2.connectedComponents(target, connectivity=8)
        # 先设置为100，保证缺陷少时的准确性
        max_num = 100
        max_num = max(max_num, tar_total_num * 2)
        pre_total_num = min(pre_total_num, max_num)
        # pre_total_num = min(pre_total_num, tar_total_num * 2)
        pre_class_num = np.zeros((self.nclass,))
        tar_class_num = np.zeros((self.nclass,))
        # tp_mask = target * ((predict == target) & (predict != 0))
        # tp_total_num, tp_labels = connected(tp_mask)
        ## 计算总数
        for id in range(1, self.nclass):
            # 这里可能会算重复，原因未明
            pre_class_num[id] = len(np.unique(pre_labels[predict == id]))
            tar_class_num[id] = len(np.unique(tar_labels[target == id]))
            # tp_class_num[id] = len(np.unique(tp_labels[tp_mask == id]))
        ## 过滤target缺陷
        if self.filter_type != 'none':
            for tar_index in range(1, tar_total_num):
                mask_tar = tar_labels == tar_index
                points_tar = np.where(mask_tar)
                tar_id = target[points_tar[0][0], points_tar[1][0]]
                ## ignore_index
                if tar_id in self.ignore_index:
                    continue
                tar_filter_flag = self.f1_defect_filter(points_tar, tar_id)
                if tar_filter_flag:
                    target[mask_tar] = 0
                    tar_class_num[tar_id] -= 1
                    tar_total_num -= 1
        ## 过滤predict缺陷, pre与tar连通域匹配, 计算tp和total
        for pre_index in range(1, pre_total_num):
            mask_pre = pre_labels == pre_index
            points_pre = np.where(mask_pre)
            pre_id = predict[points_pre[0][0], points_pre[1][0]]
            ## ignore_index
            if pre_id in self.ignore_index:
                continue
            ## 过滤predict缺陷
            pre_filter_flag = self.f1_defect_filter(points_pre, pre_id)
            if pre_filter_flag:
                predict[mask_pre] = 0
                pre_class_num[pre_id] -= 1
                pre_total_num -= 1
                continue
            mask_inter = mask_pre & (target == pre_id)  ## 交集的mask
            if np.count_nonzero(mask_inter) == 0:  ## 交集为空相当于gt里没有这个缺陷
                continue
            # points_inter = np.where(mask_inter)
            # tar_index = tar_labels[points_inter[0][0], points_inter[1][0]]
            tar_index = tar_labels[mask_inter][0]
            mask_tar = tar_labels == tar_index
            ## 计算度量指标
            if self.metric_type == 'pix_iof':
                metric_score = np.count_nonzero(mask_inter) / np.count_nonzero(mask_tar)
            elif self.metric_type == 'pix_iou':
                mask_union = mask_pre | mask_tar  ## 并集的mask
                metric_score = np.count_nonzero(mask_inter) / np.count_nonzero(mask_union)
            else:
                points_tar = np.where(mask_tar)
                tar_ymin, tar_ymax = points_tar[0].min(), points_tar[0].max()
                tar_xmin, tar_xmax = points_tar[1].min(), points_tar[1].max()
                pre_ymin, pre_ymax = points_pre[0].min(), points_pre[0].max()
                pre_xmin, pre_xmax = points_pre[1].min(), points_pre[1].max()
                inter_xmin = max(pre_xmin, tar_xmin)
                inter_ymin = max(pre_ymin, tar_ymin)
                inter_xmax = min(pre_xmax, tar_xmax)
                inter_ymax = min(pre_ymax, tar_ymax)
                inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                if self.metric_type == 'box_iof':
                    tar_area = (tar_xmax - tar_xmin) * (tar_ymax - tar_ymin)
                    metric_score = inter_area / tar_area
                elif self.metric_type == 'box_iou':
                    union_area = (pre_xmax - pre_xmin) * (pre_ymax - pre_ymin) + \
                                 (tar_xmax - tar_xmin) * (tar_ymax - tar_ymin) - inter_area
                    metric_score = inter_area / union_area

            if metric_score >= self.metric_threshold[pre_id]:
                tar_labels[mask_tar] = 0  ## 防止一个gt被对应到多个predict,从而导致recall虚高
                tp_total_num += 1
                tp_class_num[pre_id] += 1
        ## 累加defect
        self.class_defect_info['tp'] += tp_class_num
        self.class_defect_info['target'] += tar_class_num
        self.class_defect_info['predict'] += pre_class_num
        # label_num-1 去掉背景
        self.total_defect_info['tp'] += tp_total_num
        self.total_defect_info['target'] += tar_total_num - 1
        self.total_defect_info['predict'] += pre_total_num - 1

    def segmentation_defect_f1(self, predicts, targets):
        """Batch truePositive_and_total
        Args:
            predicts: 3D nparray, (n, h, w).
            targets: 3D nparray, (n, h, w).
        """
        batch = predicts.shape[0]
        for i in range(batch):
            one_predict = predicts[i, :, :]
            one_target = targets[i, :, :]
            self.oneimg_tp_pre_tar(one_predict, one_target)

    def update_batch_metrics(self, predicts, targets):
        """Calculate batch-wise segmentation metric.

        :param predicts: 3D nparray, (n, h, w), after processing with torch.max().
        :param targets: 3D nparray, (n, h, w).
        :return:
        """
        sta_data = time.time()
        predicts = np.uint8(predicts)
        targets = np.uint8(targets)
        assert np.max(predicts) <= self.nclass-1
        assert np.max(targets) <= self.nclass-1
        end_data = time.time()

        ## ------ 计算pixAcc, IoU, F1 ------ ##
        self.batch_pixelAcc_iou(predicts, targets)
        end_piexl = time.time()
        if self.com_f1:
            self.segmentation_defect_f1(predicts, targets)
        end_defect = time.time()

    def get_epoch_results(self):
        ## pixAcc
        self.class_pixel_info['label'] += self.smooth_n
        class_pixAcc = self.class_pixel_info['inter'] / self.class_pixel_info['label']
        mean_pixAcc = np.nanmean(class_pixAcc)
        sum_pixAcc = np.nansum(self.class_pixel_info['inter']) / np.nansum(self.class_pixel_info['label'])
        ## IoU
        self.class_pixel_info['union'] += self.smooth_n
        class_iou = self.class_pixel_info['inter'] / self.class_pixel_info['union']
        mean_iou = np.nanmean(class_iou)
        sum_iou = np.nansum(self.class_pixel_info['inter']) / np.nansum(self.class_pixel_info['union'])
        ## recall, precision, F1
        self.class_defect_info['target'] += self.smooth_n
        self.class_defect_info['predict'] += self.smooth_n
        class_recall = self.class_defect_info['tp'] / self.class_defect_info['target']
        class_precision = self.class_defect_info['tp'] / self.class_defect_info['predict']
        class_F1 = (2 * class_recall * class_precision) / (class_recall + class_precision + self.smooth_n)

        mean_recall = np.nanmean(class_recall)
        mean_precision = np.nanmean(class_precision)
        mean_F1 = np.nanmean(class_F1)

        sum_recall = self.total_defect_info['tp'] / (self.total_defect_info['target'] + self.smooth)
        sum_precision = self.total_defect_info['tp'] / (self.total_defect_info['predict'] + self.smooth)
        sum_F1 = (2 * sum_recall * sum_precision) / (sum_recall + sum_precision + self.smooth)
        ## eval_results
        eval_results = {}
        eval_results['IoU'] = {'class': class_iou, 'mean': mean_iou, 'sum': sum_iou}
        eval_results['Acc'] = {'class': class_pixAcc, 'mean': mean_pixAcc, 'sum': sum_pixAcc}
        eval_results['Recall'] = {'class': class_recall, 'mean': mean_recall, 'sum': sum_recall}
        eval_results['Precision'] = {'class': class_precision, 'mean': mean_precision, 'sum': sum_precision}
        eval_results['F1'] = {'class': class_F1, 'mean': mean_F1, 'sum': sum_F1}

        return eval_results

    def reset(self):
        self.class_pixel_info = {}
        self.class_defect_info = {}
        self.total_defect_info = {}
        # TODO how to define 0/0
        for pf, df in zip(self.pixel_fields, self.defect_fields):
            self.class_pixel_info[pf] = np.zeros((self.nclass, )) #self.smooth_n.copy()
            self.class_defect_info[df] = np.zeros((self.nclass, )) #self.smooth_n.copy()
            self.total_defect_info[df] = 0.0

        # fill np.nan for ignore_index in class_info
        for c in range(0, self.nclass):
            if c in self.ignore_index:
                for pf, df in zip(self.pixel_fields, self.defect_fields):
                    self.class_pixel_info[pf][c] = np.nan
                    self.class_defect_info[df][c] = np.nan
