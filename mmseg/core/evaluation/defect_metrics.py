
import cv2
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame


## --------------- segmentation metric implemented by zhangshuai --------------- ##
class SegmentationMetric(object):
    """Computes pixAcc, mIoU, recall, precision, F1 metric scroes
    """
    def __init__(self, nclass, defect_metric, defect_filter, ignore_index=[], com_f1=True):
        self.nclass = nclass
        self.defect_metric = defect_metric
        self.defect_filter = defect_filter
        if ignore_index:
            self.ignore_index = ignore_index if isinstance(ignore_index, list) else [ignore_index]
        else:
            self.ignore_index = []
        self.com_f1 = com_f1

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
        filter_type = self.defect_filter.TYPE
        filter_size = self.defect_filter.SIZE_ALL
        filter_flag = False
        if filter_type == '':
            pass
        elif filter_type == 'box':
            pre_xmin, pre_ymin, pre_xmax, pre_ymax = points_2_box(points_pre)
            if pre_ymax - pre_ymin <= filter_size[0] and pre_xmax - pre_xmin <= filter_size[1]:
                filter_flag = True
        elif filter_type == 'area':
            defect_area = points_2_area(points_pre)
            if defect_area <= filter_size[0]:
                filter_flag = True
        elif filter_type == 'minRect':
            rect = points_2_minrect(points_pre)
            ((cx, cy), (w, h), theta) = rect
            long = max(h, w)
            short = min(h, w)
            if long <= filter_size[0] and short <= filter_size[1]:
                filter_flag = True
        else:
            raise RuntimeError('unknown defect filter type!!!')

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
        ## 参数解析
        metric_type = self.defect_metric.TYPE
        metric_threshold = self.defect_metric.THRESHOLD
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
        # print('pred:{}, tar:{}'.format(pre_total_num, tar_total_num))
        # tp_mask = target * ((predict == target) & (predict != 0))
        # tp_total_num, tp_labels = connected(tp_mask)
        ## 计算总数
        for id in range(1, self.nclass):
            # 这里可能会算重复，原因未明
            pre_class_num[id] = len(np.unique(pre_labels[predict == id]))
            tar_class_num[id] = len(np.unique(tar_labels[target == id]))
            # tp_class_num[id] = len(np.unique(tp_labels[tp_mask == id]))
        ## 过滤target缺陷
        if self.defect_filter.TYPE != '':
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
            if self.defect_filter.TYPE != '':
                pre_filter_flag = self.f1_defect_filter(points_tar, tar_id)
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
            if metric_type == 'pix_iof':
                metric_score = np.count_nonzero(mask_inter) / np.count_nonzero(mask_tar)
            elif metric_type == 'pix_iou':
                mask_union = mask_pre | mask_tar  ## 并集的mask
                metric_score = np.count_nonzero(mask_inter) / np.count_nonzero(mask_union)
            else:
                points_tar = np.where(mask_tar)
                tar_ymin, tar_ymax = points_tar[0].min(), points_tar[0].max()
                tar_xmin, tar_xmax = points_tar[1].min(), points_tar[1].max()
                inter_xmin = max(pre_xmin, tar_xmin)
                inter_ymin = max(pre_ymin, tar_ymin)
                inter_xmax = min(pre_xmax, tar_xmax)
                inter_ymax = min(pre_ymax, tar_ymax)
                inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                if metric_type == 'box_iof':
                    tar_area = (tar_xmax - tar_xmin) * (tar_ymax - tar_ymin)
                    metric_score = inter_area / tar_area
                elif metric_type == 'box_iou':
                    union_area = (pre_xmax - pre_xmin) * (pre_ymax - pre_ymin) + \
                                 (tar_xmax - tar_xmin) * (tar_ymax - tar_ymin) - inter_area
                    metric_score = inter_area / union_area

            if metric_score >= metric_threshold[pre_id]:
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


def time_test_num(num = 100):
    start = time.time()
    for _ in range(num):
        point_list = np.where(mask_index != 0)
        id = target[point_list[0][0], point_list[1][0]]
    end1 = time.time()
    print((end1 - start)/num)


if __name__ == '__main__':
    smooth = np.spacing(1)
    tar = np.zeros((1024, 1280), dtype='uint8')
    tar[50:150, 50:150] = 1
    # mask_tar = tar == 1
    # ss = np.uint8(mask_tar)
    # start1 = time.time()
    # for i in range(1000):
    #     points_tar = np.where(mask_tar)
    # start2 = time.time()
    # for i in range(1000):
    #     points_tar1 = np.nonzero(ss)
    # start3 = time.time()
    # print((start2-start1), (start3-start2))
    num = 100
    time1, time2 = 0, 0
    for i in range(num):
        start1 = time.time()
        mask_tar = cv2.bitwise_and(tar, 1)
        contours, _ = cv2.findContours(mask_tar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = [np.zeros(shape=(len(points_tar[1]), 1, 2), dtype=np.int32)]
        # for l in range(len(points_tar[0])):
        #     contours[0][l] = [points_tar[1][l], points_tar[0][l]]
        # 根据contours得到最小外接矩形
        rect1 = cv2.minAreaRect(contours[0])

        # pre_id = tar[mask_tar][0]
        # area = np.count_nonzero(mask_tar)

        end1 = time.time()
        mask_tar = tar == 1
        points_tar = np.where(mask_tar)
        rect2 = cv2.minAreaRect(np.array(points_tar).transpose())

        # points_tar = np.where(mask_tar)
        # pre_id1 = tar[points_tar[0][0], points_tar[1][0]]
        # area1 = len(points_tar[0])
        # area1 = points_tar[0].shape[0]

        end2 = time.time()
        time1 += end1-start1
        time2 += end2-end1
    print(time1/num, time2/num)
    print('rect1', rect1)
    print('rect2', rect2)
    # 得到最小外接矩形的形状
    # ((cx, cy), (w, h), theta) = rect


    # t1, t2 = 0, 0
    # num = 1000
    # for i in range(num):
    #     mask_pre = tar[0] == 1
    #     st = time.time()
    #     pre_id1 = tar[0][mask_pre][0]
    #     end1 = time.time()
    #     points = np.where(mask_pre)
    #     pre_id = tar[0][points[0][0], points[1][0]]
    #     end2 = time.time()
    #     t1 += end1-st
    #     t2 += end2-st
    # print('t1:{}, t2:{}'.format(t1/num, t2/num))
    # tar[160:210, 160:210] = 2
