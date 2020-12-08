##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import cv2
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from pandas import DataFrame
from encoding.utils.files import *

import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt

class SegmentationMetric_zh(object):
    """Computes pixAcc and mIoU metric scroes
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def sk_prc(true_cls_list, pred_score_list, save_path, pos_label=1):
    '''

    :return:
    '''
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true=true_cls_list,
                                                                          probas_pred=pred_score_list,
                                                                          pos_label=pos_label,
                                                                          sample_weight=None)

    plt.figure(2)
    plt.plot(precision, recall)

    plt.xlabel('recall')
    plt.ylabel('precision')
    pr_title = osp.basename(save_path).split('.')[0]
    plt.title(pr_title)

    plt.savefig(save_path)
    # plt.show()

    return precision, recall, threshold

def sk_roc(true_cls_list, pred_score_list, save_path, pos_label=None, sample_weight=None, drop_intermediate=True):
    '''
    ROC曲线即receiver operating characteristic curve.
    ROC观察模型正确地识别正例的比例与模型错误地把负例数据识别成正例的比例之间的权衡。
    TPR为纵轴，FPR为横轴。TPR的增加以FPR的增加为代价。
    :return:
    '''
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=true_cls_list,
                                                     y_score=pred_score_list,
                                                     pos_label=pos_label,
                                                     sample_weight=sample_weight,
                                                     drop_intermediate=drop_intermediate)

    plt.figure(1)
    plt.plot(fpr, tpr, lw=1, label='ROC')
    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_title = osp.basename(save_path).split('.')[0]
    plt.title(roc_title)
    plt.savefig(save_path)
    # plt.show()

    return fpr, tpr, thresholds

def sk_auc(x, y):
        '''
        ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）。
        :return:
        '''
        auc = sklearn.metrics.auc(x=x, y=y, reorder='deprecated')

        return auc

def batch_pix_accuracy(output, target, nclass):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))

    class_pixel_labeled = []
    class_pixel_correct = []
    for one_class in range(0, nclass):
        one_pixel_labeled = np.sum(target == (one_class + 1))
        one_pixel_correct = np.sum((predict == target) * (target == (one_class + 1)))
        class_pixel_labeled.append(one_pixel_labeled)
        class_pixel_correct.append(one_pixel_correct)

    np_imgs = [predict - 1, target - 1]
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return np_imgs, pixel_correct, pixel_labeled, class_pixel_correct, class_pixel_labeled

def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    # predict[predict == 2] = 1
    # target[target == 2] = 1
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled

def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def batch_ok_score(output, target, nclass):
    """Batch Score and Ok
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    ok = []
    score = []
    for i in range(output.size(0)):
        output_score = 1 - output[i][0]
        max_score = output_score.max().item()
        score.append(max_score)
        if target[i].sum():
            ok.append(1)
        else:
            ok.append(0)

    return ok, score

def auc(dict_score):
    """aud
    Args:
        dict_score: ok score
    """
    pd_score = DataFrame.from_dict(dict_score)
    pd_score['score'] = pd_score['score'] * 100
    TP = []
    FP = []
    FN = []
    TN = []
    for i in range(100):
        p = pd_score[pd_score['score'] >= i]
        tp = p[p['ok'] == 1]
        fp = p[p['ok'] == 0]
        n = pd_score[pd_score['score'] < i]
        fn = n[n['ok'] == 1]
        tn = n[n['ok'] == 0]
        TP.append(len(tp))
        FP.append(len(fp))
        FN.append(len(fn))
        TN.append(len(tn))

    TP = np.asarray(TP).astype('float32')
    FP = np.asarray(FP).astype('float32')
    FN = np.asarray(FN).astype('float32')
    TN = np.asarray(TN).astype('float32')

    if 0 in (FP + TN):
        return 0
    if 0 in (TP + FN):
        return 1

    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)

    auc = 0
    for i in range(len(FPR)):
        auc = auc + TPR[i]
    auc = auc / len(FPR)

    return auc

## --------------- segmentation metric implemented by zhangshuai --------------- ##
def segmentation_pixelAcc_iou(predict, target, nclass, ignore_index):
    """segmentation_pixelAcc_iou
    Args:
        predict: input 3D nparray
        target: label 3D nparray
        nclass: number of categories (int)
    """

    batch_pixel_labeled = np.zeros(nclass, np.uint64) #['NaN' for _ in range(nclass)]
    batch_pixel_correct = np.zeros(nclass, np.uint64)
    batch_pixel_union = np.zeros(nclass, np.uint64)

    for c in range(0, nclass):
        # if c in ignore_index:
        #     continue
        class_predict_mask = predict == c
        class_target_mask = target == c

        class_pixel_labeled = np.count_nonzero(class_target_mask)
        # import time
        # times = 1000
        # start = time.time()
        # for _ in range(times):
        #     class_pixel_labeled = np.count_nonzero(class_target_mask)
        # end = time.time()
        #
        # for _ in range(times):
        #     class_pixel_labeled1 = cv2.countNonZero(class_target_mask1.astype(np.uint8))
        # end1 = time.time()
        # print((end-start) * 1000, (end1-end) * 1000, class_pixel_labeled, class_pixel_labeled1)
        class_pixel_correct = np.count_nonzero(class_target_mask & class_predict_mask)
        class_pixel_union = np.count_nonzero(class_target_mask | class_predict_mask)

        batch_pixel_labeled[c] = class_pixel_labeled
        batch_pixel_correct[c] = class_pixel_correct
        batch_pixel_union[c] = class_pixel_union

    return batch_pixel_labeled, batch_pixel_correct, batch_pixel_union

def segmentation_auc(dict_score):
    """aud
    Args:
        dict_score: ok score
    """
    if len(dict_score['score']) == 0:
        return 0

    pd_score = DataFrame.from_dict(dict_score)
    pd_score['score'] = pd_score['score'] * 100
    TP = []
    FP = []
    FN = []
    TN = []
    for i in range(100):
        p = pd_score[pd_score['score'] >= i]
        tp = p[p['ok'] == 1]
        fp = p[p['ok'] == 0]
        n = pd_score[pd_score['score'] < i]
        fn = n[n['ok'] == 1]
        tn = n[n['ok'] == 0]
        TP.append(len(tp))
        FP.append(len(fp))
        FN.append(len(fn))
        TN.append(len(tn))

    TP = np.asarray(TP).astype('float32')
    FP = np.asarray(FP).astype('float32')
    FN = np.asarray(FN).astype('float32')
    TN = np.asarray(TN).astype('float32')

    if 0 in (FP + TN):
        return 0
    if 0 in (TP + FN):
        return 1

    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)

    auc = 0
    for i in range(len(FPR)):
        auc = auc + TPR[i]
    auc = auc / len(FPR)

    return auc

def f1_defect_filter(points_pre, pre_id, defect_filter, station):
    filter_type = defect_filter.TYPE
    if defect_filter.STATION:
        filter_size_station = defect_filter.SIZE_STATION[0]
        filter_size = filter_size_station[station][pre_id-1] ## 减1跳过背景
    else:
        filter_size = defect_filter.SIZE_ALL
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

def oneimg_tp_pre_tar(predict, target, defect_metric, defect_filter, nclass, station, ignore_index):
    ## 参数解析
    metric_type = defect_metric.TYPE
    metric_threshold = defect_metric.THRESHOLD
    ## 初始化
    tp_total_num = 0
    tp_class_num = np.zeros((nclass,), dtype=np.uint64)
    ## 计算连通域
    pre_total_num, pre_labels = connected(predict)
    tar_total_num, tar_labels = connected(target)
    # 先设置为100，保证缺陷少时的准确性
    max_num = 100
    max_num = max(max_num, tar_total_num * 2)
    pre_total_num = min(pre_total_num, max_num)
    #pre_total_num = min(pre_total_num, tar_total_num * 2)
    pre_class_num = np.zeros((nclass,), dtype=np.uint64)
    tar_class_num = np.zeros((nclass,), dtype=np.uint64)
    # print('pred:{}, tar:{}'.format(pre_total_num, tar_total_num))
    # tp_mask = target * ((predict == target) & (predict != 0))
    # tp_total_num, tp_labels = connected(tp_mask)
    ## 计算总数
    for id in range(1, nclass):
        # 这里可能会算重复，原因未明
        pre_class_num[id] = len(np.unique(pre_labels[predict == id]))
        tar_class_num[id] = len(np.unique(tar_labels[target == id]))
        # tp_class_num[id] = len(np.unique(tp_labels[tp_mask == id]))
    ## 过滤target缺陷
    if defect_filter.TYPE != '':
        for tar_index in range(1, tar_total_num):
            mask_tar = tar_labels == tar_index
            points_tar = np.where(mask_tar)
            tar_id = target[points_tar[0][0], points_tar[1][0]]
            ## ignore_index
            if tar_id in ignore_index:
                continue
            tar_filter_flag = f1_defect_filter(points_tar, tar_id, defect_filter, station)
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
        if pre_id in ignore_index:
            continue
        if defect_filter.TYPE != '':
            pre_filter_flag = f1_defect_filter(points_pre, pre_id, defect_filter, station)
            if pre_filter_flag:
                predict[mask_pre] = 0
                pre_class_num[pre_id] -= 1
                pre_total_num -= 1
                continue
        mask_inter = mask_pre & (target == pre_id) ## 交集的mask
        if np.count_nonzero(mask_inter) == 0: ## 交集为空相当于gt里没有这个缺陷
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
            inter_area = (inter_xmax-inter_xmin) * (inter_ymax-inter_ymin)
            if metric_type == 'box_iof':
                tar_area = (tar_xmax-tar_xmin)*(tar_ymax-tar_ymin)
                metric_score = inter_area / tar_area
            elif metric_type == 'box_iou':
                union_area = (pre_xmax-pre_xmin)*(pre_ymax-pre_ymin) + \
                             (tar_xmax-tar_xmin)*(tar_ymax-tar_ymin) - inter_area
                metric_score = inter_area / union_area

        if metric_score >= metric_threshold[pre_id]:
            tar_labels[mask_tar] = 0 ## 防止一个gt被对应到多个predict,从而导致recall虚高
            tp_total_num += 1
            tp_class_num[pre_id] += 1
    ## label_num-1 去掉背景
    one_total_defect_f1 = np.array([tp_total_num, pre_total_num-1, tar_total_num-1], dtype=np.uint64)
    one_class_defect_f1 = np.array([tp_class_num, pre_class_num, tar_class_num], dtype=np.uint64)
    #print(one_total_defect_f1)
    #print(one_class_defect_f1)
    return one_total_defect_f1, one_class_defect_f1

def segmentation_defect_f1(predict, target, defect_metric, defect_filter, nclass, stations, ignore_index):
    """Batch truePositive_and_total
    Args:
        predict: input 3D nparray
        target: label 3D nparray
        defect_metric: defect_metric.TYPE, defect_metric.THRESHOLD
        defect_filter: defect_filter.TYPE, defect_filter.SIZE
        nclass: number of categories (int)
        stations: 工位 (list)
        ignore_index: Categories not calculated (list)
    """
    batch_total_defect_f1 = np.zeros((3,), dtype=np.uint64)
    batch_class_defect_f1 = np.zeros((3, nclass), dtype=np.uint64)

    b = predict.shape[0]
    for i in range(b):
        one_predict = predict[i, :, :]
        one_target = target[i, :, :]
        ## 不计算predict和target都是无缺陷的情况
        # TODO
        # if np.count_nonzero(one_predict) + np.count_nonzero(one_target) == 0:
        if cv2.countNonZero(one_predict) + cv2.countNonZero(one_target) == 0:
            continue
        if stations == []:
            station = ''
        else:
            station = stations[i]
        one_total_defect_f1, one_class_defect_f1 = oneimg_tp_pre_tar(one_predict, one_target,
                                         defect_metric, defect_filter, nclass, station, ignore_index)
        batch_total_defect_f1 += one_total_defect_f1
        batch_class_defect_f1 += one_class_defect_f1

    return batch_total_defect_f1, batch_class_defect_f1

def segmentation_batch_eval(model, image, target, defect_metric, defect_filter, station, ignore_index=[0], com_f1=True):
    start_val = time.time()
    with torch.no_grad():
        outputs = model(image)
    end_model = time.time()
    ## ------ 收集每个GPU上的output ------ ##
    if not isinstance(outputs, list):
        outputs = [outputs]
    predicts = []
    for output in outputs:
        predicts.append(output[0])  # 只取0, 丢弃掉aux的输出
    predict = torch.cat(predicts, 0)
    nclass = predict.size(1)
    end_cat = time.time()
    ## ------ 使用Tensor计算auc相关指标 ------ ##
    ok, score = batch_ok_score(predict, target, nclass)
    end_auc = time.time()
    ## ------ predict, target 转 numpy ------ ##
    _, predict = torch.max(predict, 1)
    predict = predict.cpu().numpy().astype(np.uint8)
    target = target.cpu().numpy().astype(np.uint8)
    end_npy = time.time()
    ## ------ 计算pixAcc, IoU, F1 ------ ##
    batch_pixel_labeled, batch_pixel_correct, batch_pixel_union = \
                                      segmentation_pixelAcc_iou(predict, target, nclass, ignore_index)
    end_iou = time.time()
    if com_f1:
        batch_total_defect_f1, batch_class_defect_f1 = segmentation_defect_f1(predict, target,
                                          defect_metric, defect_filter, nclass, station, ignore_index)
    else:
        batch_total_defect_f1 = np.zeros((3,), dtype=np.uint64)
        batch_class_defect_f1 = np.zeros((3, nclass), dtype=np.uint64)
    end_f1 = time.time()
    # print('model: {}, cat: {}, auc: {}, npy: {}, iou: {}, f1: {}'.format((end_model-start_val),
    #     (end_cat-end_model), (end_auc-end_cat), (end_npy-end_auc), (end_iou-end_npy), (end_f1-end_iou)))
    return ok, score, batch_pixel_labeled, batch_pixel_correct, batch_pixel_union, \
                batch_total_defect_f1, batch_class_defect_f1

class SegmentationDefectF1(object):
    def __init__(self, nclass, defect_metric, defect_filter, ignore_index=[0]):
        """ Computes recall, precision, F1 metric scroes
        Args:
            nclass: number of categories (int)
            defect_metric: defect_metric.TYPE, defect_metric.THRESHOLD
            defect_filter: defect_filter.TYPE, defect_filter.SIZE
            ignore_index: Categories not calculated (list)
        """
        self.nclass = nclass
        self.defect_filter = defect_filter
        self.defect_metric = defect_metric
        self.ignore_index = ignore_index

    def _init_batch_matrics(self):
        ## total_num
        self.pre_total_num, self.tar_total_num, self.tp_total_num = 0, 0, 0
        ## class_num
        self.pre_class_num = np.zeros((self.nclass,), dtype=np.uint64)
        self.tar_class_num = np.zeros((self.nclass,), dtype=np.uint64)
        self.tp_class_num = np.zeros((self.nclass,), dtype=np.uint64)

    def _init_oneimg_matrics(self):
        ## total_num
        self.pre_total_num, self.tar_total_num, self.tp_total_num = 0, 0, 0
        ## class_num
        self.pre_class_num = np.zeros((self.nclass,), dtype=np.uint64)
        self.tar_class_num = np.zeros((self.nclass,), dtype=np.uint64)
        self.tp_class_num = np.zeros((self.nclass,), dtype=np.uint64)
        ## label_map_dict : predict to target
        self.index_map = []
        self.index_pre2tar_dict = dict()
        self.index_tar2pre_dict = dict()

    def _update_index_dict(self, pre_cnt_indexes, tar_cnt_index):
        for pre_index in pre_cnt_indexes:
            ## update index_pre2tar_dict
            if pre_index not in self.index_pre2tar_dict:
                self.index_pre2tar_dict[pre_index] = [tar_cnt_index]
            else:
                if tar_cnt_index not in self.index_pre2tar_dict[pre_index]:
                    self.index_pre2tar_dict[pre_index].append(tar_cnt_index)
            ## update index_pre2tar_dict
            if tar_cnt_index not in self.index_tar2pre_dict:
                self.index_tar2pre_dict[tar_cnt_index] = pre_cnt_indexes
            else:
                for pre_index in pre_cnt_indexes:
                    if pre_index not in self.index_pre2tar_dict[tar_cnt_index]:
                        self.index_tar2pre_dict[tar_cnt_index].append(pre_index)

    def _f1_defect_filter(self, mask, pre_id, station):
        filter_flag = False
        filter_type = self.defect_filter.TYPE
        if filter_type == '':
            pass
        else:
            ## 参数
            if self.defect_filter.STATION:
                filter_size_station = self.defect_filter.SIZE_STATION[0]
                filter_size = filter_size_station[station][pre_id - 1]  ## 减1跳过背景
            else:
                filter_size = self.defect_filter.SIZE_ALL
            ## 计算
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if filter_type == 'box':
                x, y, w, h = cv2.boundingRect(contours[0])
                if h <= filter_size[0] and w <= filter_size[1]:
                    filter_flag = True
            elif filter_type == 'area':
                defect_area = cv2.contourArea(contours[0])
                if defect_area <= filter_size[0]:
                    filter_flag = True
            elif filter_type == 'minRect':
                ((cx, cy), (w, h), theta) = cv2.minAreaRect(contours[0])
                long = max(h, w)
                short = min(h, w)
                if long <= filter_size[0] and short <= filter_size[1]:
                    filter_flag = True
            else:
                raise RuntimeError('unknown defect filter type!!!')
        return filter_flag

    def _f1_defect_metric(self, mask_pre, mask_tar, label):
        ## 参数
        metric_type = self.defect_metric.TYPE
        metric_threshold = self.defect_metric.THRESHOLD
        ## 计算
        if 'pix' in metric_type:
            mask_inter = cv2.bitwise_and(mask_pre, mask_tar)
            if metric_type == 'pix_iof':
                metric_score = cv2.countNonZero(mask_inter) / cv2.countNonZero(mask_tar)
            elif metric_type == 'pix_iou':
                mask_union = cv2.bitwise_or(mask_pre, mask_tar)
                metric_score = cv2.countNonZero(mask_inter) / cv2.countNonZero(mask_union)
        elif 'box' in metric_type:
            contours_pre, _ = cv2.findContours(mask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_tar, _ = cv2.findContours(mask_tar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pre_xmin, pre_ymin, pre_w, pre_h = cv2.boundingRect(contours_pre[0])
            tar_xmin, tar_ymin, tar_w, tar_h = cv2.boundingRect(contours_tar[0])
            inter_xmin = max(pre_xmin, tar_xmin)
            inter_ymin = max(pre_ymin, tar_ymin)
            inter_xmax = min(pre_xmin + pre_w, tar_xmin + tar_w)
            inter_ymax = min(pre_ymin + pre_h, tar_ymin + tar_h)
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            if filter_type == 'box_iof':
                metric_score = inter_area / (tar_w * tar_h)
            elif metric_type == 'box_iou':
                union_area = (pre_w, pre_h) + (tar_w * tar_h) - inter_area
                metric_score = inter_area / union_area
        ## 输出
        if metric_score >= metric_threshold[label]:
            self.tp_total_num += 1
            self.tp_class_num[label] += 1
            return True
        else:
            return False

    def batch_eval(self, predict, target, stations):
        """
        :param predict: input 3D nparray
        :param target: label 3D nparray
        :param stations: 工位 (list)
        :return:
        """
        self._init_batch_matrics()
        b = predict.shape[0]
        for i in range(b):
            one_predict = predict[i, :, :]
            one_target = target[i, :, :]
            ## 不计算predict和target都是无缺陷的情况
            if cv2.countNonZero(one_predict) + cv2.countNonZero(one_target) == 0:
                continue
            if stations == []:
                station = ''
            else:
                station = stations[i]
            self.oneimg_pre_tar_m2o(one_predict, one_target, station)
        batch_total_defect_f1 = np.uint64([self.tp_total_num, self.pre_total_num-1, self.tar_total_num-1])
        batch_class_defect_f1 = np.uint64([self.tp_class_num, self.pre_class_num, self.tar_class_num])

        return batch_total_defect_f1, batch_class_defect_f1

    def oneimg_pre_tar_m2o(self, predict, target, station):
        """
        One target correspond to multi predict
        """
        ## 初始化
        # self._init_oneimg_matrics()
        for label in range(0, self.nclass):
            if label in self.ignore_index:
             continue
            predict_label = cv2.bitwise_and(predict, label)
            target_label = cv2.bitwise_and(target, label)
            pre_cnt_num, pre_cnt_masks = cv2.connectedComponents(predict_label, connectivity=8)
            tar_cnt_num, tar_cnt_masks = cv2.connectedComponents(target_label, connectivity=8)
            pre_cnt_num = min(pre_cnt_num, tar_cnt_num * 2)
            pre_iter_num = 0

            ## 过滤target缺陷
            for tar_cnt_index in range(1, tar_cnt_num):
                mask_tar = cv2.bitwise_and(tar_cnt_masks, tar_cnt_index)
                tar_filter_flag = self._f1_defect_filter(mask_tar, label, station)
                ## target缺陷被过滤, 则跳过
                if tar_filter_flag:
                    continue
                ## pre与tar连通域匹配, 计算tp和total
                # pre_cnt_indexes = []
                # for id in np.unique(pre_cnt_masks[mask_tar.astype(np.bool)]):
                #     pre_cnt_indexes.append(int(id))
                points_acc = pre_cnt_masks[mask_tar.astype(np.bool)]
                ## target与predict无交集, 则是误检
                if len(points_acc) == 0:
                    pre_iter_num += 1
                    self.pre_total_num += 1
                    self.pre_class_num[label] += 1
                    continue
                pre_cnt_indexes = np.unique(points_acc)
                while pre_iter_num <= pre_cnt_num:
                    for pre_cnt_index in pre_cnt_indexes:
                        ## pre_cnt_index==0, 属于背景, 不参与计算
                        if pre_cnt_index == 0:
                            continue
                        mask_pre = cv2.bitwise_and(pre_cnt_masks, int(pre_cnt_index))
                        pre_filter_flag = self._f1_defect_filter(mask_pre, label, station)
                        ## predict缺陷被过滤, 则跳过
                        if pre_filter_flag:
                            continue
                        pre_iter_num += 1
                        self.tar_total_num += 1
                        self.tar_class_num[label] += 1
                        self.pre_total_num += 1
                        self.pre_class_num[label] += 1
                        ac = self._f1_defect_metric(mask_pre, mask_tar, label)


    def oneimg_tp_pre_tar_m2m(self, one_predict, one_target, station):
        ## 初始化
        self._init_oneimg_matrics()

        for label in range(0, self.nclass):
            if label in ignore_index:
                continue
            predict_label = cv2.bitwise_and(predict, label)
            target_label = cv2.bitwise_and(target, label)
            pre_cnt_num, pre_cnt_masks = cv2.connectedComponents(predict_label, connectivity=8)
            tar_cnt_num, tar_cnt_masks = cv2.connectedComponents(target_label, connectivity=8)
            pre_cnt_num = min(pre_cnt_num, tar_cnt_num * 2)

            ## 过滤target缺陷
            for tar_cnt_index in range(1, tar_cnt_num):
                mask_tar = cv2.bitwise_and(tar_cnt_masks, tar_cnt_index)
                tar_filter_flag = self._f1_defect_filter(mask_tar, label, station)
                if tar_filter_flag:
                    continue
                ## pre与tar连通域匹配, 计算tp和total
                pre_cnt_indexes = list(np.unique(pre_cnt_masks[mask_tar.astype(np.bool)]))
                for pre_cnt_index in pre_cnt_indexes:
                    self.tar_total_num += 1
                    self.tar_class_num[label] += 1
                    mask_pre = cv2.bitwise_and(pre_cnt_masks, pre_cnt_index)
                    pre_filter_flag = self._f1_defect_filter(mask_pre, label, station)
                    if pre_filter_flag:
                        continue
                    self.pre_total_num += 1
                    self.pre_class_num[label] += 1
                    self._f1_defect_metric(mask_pre, mask_tar)

            #     self._update_index_dict(pre_cnt_indexes, tar_cnt_index)
            #
            # ## 过滤predict缺陷, pre与tar连通域匹配, 计算tp和total
            # for pre_cnt_index in range(1, pre_cnt_num):
            #     mask_pre = cv2.bitwise_and(pre_cnt_masks, pre_cnt_index)
            #     pre_filter_flag = self._f1_defect_filter(mask_pre, label, station)
            #     if not pre_filter_flag:
            #         self.pre_total_num += 1
            #         self.pre_class_num[label] += 1
            #     tar_cnt_indexes = self.index_dict.get(pre_cnt_index)
            #     if tar_cnt_indexes == None:
            #         continue
            #     for i, tar_index in enumerate(tar_cnt_indexes):
            #         if i == 0:
            #             mask_tar = cv2.bitwise_and(tar_cnt_masks, tar_index)
            #         else:
            #             mask_tar = cv2.bitwise_and(mask_tar, cv2.bitwise_and(tar_cnt_masks, tar_index))
            #     self._f1_defect_metric(mask_pre, mask_tar)

        return 1, 2

class SegmentationMetric0601(object):
    """Computes pixAcc, mIoU, recall, precision, F1 metric scroes
    """
    def __init__(self, nclass, defect_metric, defect_filter, ignore_index=[0], com_f1=True):
        self.nclass = nclass
        self.defect_metric = defect_metric
        self.defect_filter = defect_filter
        self.ignore_index = ignore_index
        self.com_f1 = com_f1

        self.C1 = np.spacing(1)
        self.Cn = np.array([self.C1 for i in range(self.nclass)])
        self.f1_metric = SegmentationDefectF1(nclass, defect_metric, defect_filter, ignore_index)

    def update_batch_metrics(self, predict, target, station):
        ## ------ 使用Tensor计算auc相关指标 ------ ##
        start = time.time()
        ok, score = batch_ok_score(predict, target, self.nclass)
        end_auc = time.time()
        ## ------ predict, target 转 numpy ------ ##
        _, predict = torch.max(predict, 1)
        predict = predict.cpu().numpy().astype(np.uint8)
        target = target.cpu().numpy().astype(np.uint8)
        end_npy = time.time()
        ## ------ 计算pixAcc, IoU, F1 ------ ##
        batch_pixel_labeled, batch_pixel_correct, batch_pixel_union = \
            segmentation_pixelAcc_iou(predict, target, self.nclass, self.ignore_index)
        end_iou = time.time()
        if self.com_f1:
            batch_total_defect_f1, batch_class_defect_f1 = self.f1_metric.batch_eval(predict, target, station)
        else:
            batch_total_defect_f1 = np.zeros((3,), dtype=np.uint64)
            batch_class_defect_f1 = np.zeros((3, self.nclass), dtype=np.uint64)
        end_f1 = time.time()
        print('auc: {}, npy: {}, iou: {}, f1: {}'.format((end_auc-start),
                                     (end_npy-end_auc), (end_iou-end_npy), (end_f1-end_iou)))
        ###------------------------- 度量指标累加 ------------------------###
        ## 累加pixAcc, IoU
        self.total_pixel_labeled += batch_pixel_labeled
        self.total_pixel_correct += batch_pixel_correct
        self.total_pixel_union += batch_pixel_union
        ## 累加auc
        self.total_scores['ok'] += ok
        self.total_scores['score'] += score
        ## 累加recall, precision, F1
        self.total_defect_f1 += batch_total_defect_f1
        self.class_defect_f1 += batch_class_defect_f1

    def get_epoch_results(self):
        ## pixAcc
        class_pixAcc = self.total_pixel_correct / (self.total_pixel_labeled + self.Cn)
        pixAcc = np.sum(self.total_pixel_correct) / (np.sum(self.total_pixel_labeled + self.Cn))
        ## mIoU
        class_iou = self.total_pixel_correct / (self.total_pixel_union + self.Cn)
        mIoU = class_iou.mean()
        ## auc
        auc = segmentation_auc(self.total_scores)
        ## recall, precision, F1
        total_precision, total_recall = self.total_defect_f1[0] / (self.total_defect_f1[1:] + self.C1)
        class_precision, class_recall = self.class_defect_f1[0] / (self.class_defect_f1[1:] + self.C1)
        total_F1 = (2 * total_recall * total_precision) / (total_recall + total_precision + self.C1)
        class_F1 = (2 * class_recall * class_precision) / (class_recall + class_precision + self.C1)

        return pixAcc, class_pixAcc, mIoU, class_iou, auc, total_recall, class_recall, \
               total_precision, class_precision, total_F1, class_F1

    def reset(self):
        self.total_pixel_labeled = np.zeros(self.nclass, np.uint64)
        self.total_pixel_correct = np.zeros(self.nclass, np.uint64)
        self.total_pixel_union = np.zeros(self.nclass, np.uint64)

        self.total_scores = {'score': [], 'ok': []}

        self.total_defect_f1 = np.zeros((3,), dtype=np.uint64)
        self.class_defect_f1 = np.zeros((3, self.nclass), dtype=np.uint64)
        return

class SegmentationMetric(object):
    """Computes pixAcc, mIoU, recall, precision, F1 metric scroes
    """
    def __init__(self, nclass, defect_metric, defect_filter, ignore_index=[0], com_f1=True):
        self.nclass = nclass
        self.defect_metric = defect_metric
        self.defect_filter = defect_filter
        self.ignore_index = ignore_index
        self.com_f1 = com_f1

        self.C1 = np.spacing(1)
        self.Cn = np.array([self.C1 for i in range(self.nclass)])

    def update_batch_metrics(self, predict, target, station):
        # ## ------ 使用Tensor计算auc相关指标 ------ ##
        # start = time.time()
        # predict = F.softmax(predict, dim=1)
        # # ok, score = batch_ok_score(predict, target, self.nclass)
        # end_auc = time.time()
        # ## ------ predict, target 转 numpy ------ ##
        # _, predict = torch.max(predict, 1)
        predict = torch.Tensor(predict)
        target = torch.Tensor(target)
        predict = predict.cpu().numpy().astype(np.uint8)
        target = target.cpu().numpy().astype(np.uint8)
        end_npy = time.time()

        ## ------ 计算pixAcc, IoU, F1 ------ ##
        batch_pixel_labeled, batch_pixel_correct, batch_pixel_union = \
            segmentation_pixelAcc_iou(predict, target, self.nclass, self.ignore_index)
        end_iou = time.time()
        if self.com_f1:
            batch_total_defect_f1, batch_class_defect_f1 = segmentation_defect_f1(predict, target,
                          self.defect_metric, self.defect_filter, self.nclass, station, self.ignore_index)
        else:
            batch_total_defect_f1 = np.zeros((3,), dtype=np.uint64)
            batch_class_defect_f1 = np.zeros((3, self.nclass), dtype=np.uint64)
        end_f1 = time.time()
        # print('auc: {}, npy: {}, iou: {}, f1: {}'.format((end_auc-start),
        #                              (end_npy-end_auc), (end_iou-end_npy), (end_f1-end_iou)))
        ###------------------------- 度量指标累加 ------------------------###
        ## 累加pixAcc, IoU
        self.total_pixel_labeled += batch_pixel_labeled
        self.total_pixel_correct += batch_pixel_correct
        self.total_pixel_union += batch_pixel_union
        ## 累加auc
        # self.total_scores['ok'] += ok
        # self.total_scores['score'] += score
        ## 累加recall, precision, F1
        self.total_defect_f1 += batch_total_defect_f1
        self.class_defect_f1 += batch_class_defect_f1

    def get_epoch_results(self):
        ## pixAcc
        class_pixAcc = self.total_pixel_correct / (self.total_pixel_labeled + self.Cn)
        pixAcc = np.sum(self.total_pixel_correct) / (np.sum(self.total_pixel_labeled + self.Cn))
        ## mIoU
        class_iou = self.total_pixel_correct / (self.total_pixel_union + self.Cn)
        mIoU = class_iou.mean()
        ## auc
        auc = segmentation_auc(self.total_scores)
        ## recall, precision, F1
        total_precision, total_recall = self.total_defect_f1[0] / (self.total_defect_f1[1:] + self.C1)
        class_precision, class_recall = self.class_defect_f1[0] / (self.class_defect_f1[1:] + self.C1)
        total_F1 = (2 * total_recall * total_precision) / (total_recall + total_precision + self.C1)
        class_F1 = (2 * class_recall * class_precision) / (class_recall + class_precision + self.C1)

        return pixAcc, class_pixAcc, mIoU, class_iou, auc, total_recall, class_recall, \
               total_precision, class_precision, total_F1, class_F1

    def reset(self):
        self.total_pixel_labeled = np.zeros(self.nclass, np.uint64)
        self.total_pixel_correct = np.zeros(self.nclass, np.uint64)
        self.total_pixel_union = np.zeros(self.nclass, np.uint64)

        self.total_scores = {'score': [], 'ok': []}

        self.total_defect_f1 = np.zeros((3,), dtype=np.uint64)
        self.class_defect_f1 = np.zeros((3, self.nclass), dtype=np.uint64)
        return

## --------------- segmentation metric implemented by zhangshuai old --------------- ##
def connected(mask, connectivity=8):
    label_num, label_map = cv2.connectedComponents(mask, connectivity=connectivity)

    return label_num, label_map

def separation(mask, connectivity=8):
    label_num, label_map = connected(mask, connectivity)
    masks = []
    for i in range(1, label_num + 1):
        mask = np.uint8(label_map == i)
        masks.append(mask)
    return masks

def one_class_iou():
    nbins = 1
    mini, maxi = 1, 1
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

def compute_recall(predict, target, threshold, nclass):
    total_label_recall = 0
    total_predict_recall = 0

    class_label_recall = np.zeros((nclass,), dtype=np.int)
    class_predict_recall = np.zeros((nclass,), dtype=np.int)

    # target_mask = np.uint8(target>0) * 255
    # masklist = separation(target_mask)

    ### 修改简化!!!!
    masklist = separation(target) # .astype('uint8')

    for i in range(len(masklist)):
        point_list = np.where(masklist[i] != 0)
        index = target[point_list[0][0], point_list[1][0]]

        mask = masklist[i]
        mask_index = mask * index
        # total_label = np.sum(temp == index)
        total_label = np.sum(mask)
        total_predict = np.sum((predict == mask_index) * mask)
        total_label_recall = total_label_recall + 1
        class_label_recall[index] = class_label_recall[index] + 1
        if total_predict / total_label > threshold:
            total_predict_recall = total_predict_recall + 1
            class_predict_recall[index] = class_predict_recall[index] + 1

    ### 这里添加精度计算
    return total_label_recall, total_predict_recall, class_label_recall, class_predict_recall

def batch_recall(output, target, threshold, nclass):
    """Batch recall
    Args:
        output: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    total_label_recall = 0
    total_predict_recall = 0

    class_label_recall = np.zeros((nclass,), dtype=np.int)
    class_predict_recall = np.zeros((nclass,), dtype=np.int)

    # _, predict = torch.max(output, 1)
    predict = output
    b, h, w = predict.size()

    if b == 1:
        predict = np.squeeze(predict.cpu().numpy())
        target = np.squeeze(target.cpu().numpy())
        a, b, c, d = compute_recall(predict, target, threshold, nclass)
        total_label_recall = total_label_recall + a
        total_predict_recall = total_predict_recall + b
        class_label_recall = class_label_recall + c
        class_predict_recall = class_predict_recall + d
    else:
        for i in range(b):
            temp_predict = np.uint8(predict[i, :, :].cpu())
            temp_target = np.uint8(target[i, :, :].cpu())
            a, b, c, d = compute_recall(temp_predict, temp_target, threshold, nclass)
            total_label_recall = total_label_recall + a
            total_predict_recall = total_predict_recall + b
            class_label_recall = class_label_recall + c
            class_predict_recall = class_predict_recall + d

    return total_label_recall, total_predict_recall, class_label_recall, class_predict_recall

def np_oneimg_tp_pre_tar(predict, target, defect_metric, defect_filter, nclass, station, ignore_index):
    ## 参数解析
    metric_type = defect_metric.TYPE
    metric_threshold = defect_metric.THRESHOLD
    ## 初始化
    tp_total_num = 0
    tp_class_num = np.zeros((nclass,), dtype=np.uint64)
    ## 计算连通域
    pre_total_num, pre_labels = connected(predict)
    tar_total_num, tar_labels = connected(target)
    pre_total_num = min(pre_total_num, tar_total_num * 2)
    pre_class_num = np.zeros((nclass,), dtype=np.uint64)
    tar_class_num = np.zeros((nclass,), dtype=np.uint64)
    # print('pred:{}, tar:{}'.format(pre_total_num, tar_total_num))
    # tp_mask = target * ((predict == target) & (predict != 0))
    # tp_total_num, tp_labels = connected(tp_mask)
    ## 计算总数
    for id in range(1, nclass):
        pre_class_num[id] = len(np.unique(pre_labels[predict == id]))
        tar_class_num[id] = len(np.unique(tar_labels[target == id]))
        # tp_class_num[id] = len(np.unique(tp_labels[tp_mask == id]))
    ## 过滤target缺陷
    if defect_filter.TYPE != '':
        iter_tar = range(0, tar_total_num)
        for tar_index in iter_tar:
            mask_tar = tar_labels == tar_index
            points_tar = np.where(mask_tar)
            # print('{} index pixels {}'.format(tar_index, len(points_tar[0])))
            tar_id = target[points_tar[0][0], points_tar[1][0]]
            ## ignore_index
            if tar_id in ignore_index:
                continue
            tar_filter_flag = f1_defect_filter(mask_tar, points_tar, tar_id, defect_filter, station)
            if tar_filter_flag:
                target[mask_tar] = 0
                tar_class_num[tar_id] -= 1
                tar_total_num -= 1
    ## 过滤predict缺陷, pre与tar连通域匹配, 计算tp和total
    iter_pre = range(0, pre_total_num)
    for pre_index in iter_pre:
        mask_pre = pre_labels == pre_index
        points_pre = np.where(mask_pre)
        pre_id = predict[points_pre[0][0], points_pre[1][0]]
        ## ignore_index
        if pre_id in ignore_index:
            continue
        if defect_filter.TYPE != '':
            pre_filter_flag = f1_defect_filter(mask_pre, points_pre, pre_id, defect_filter, station)
            if pre_filter_flag:
                predict[mask_pre] = 0
                pre_class_num[pre_id] -= 1
                pre_total_num -= 1
                continue
        mask_inter = mask_pre & (target == pre_id) ## 交集的mask
        np_mask_inter = np.uint8(mask_inter)
        if np.count_nonzero(mask_inter) == 0: ## 交集为空相当于gt里没有这个缺陷
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
            inter_area = (inter_xmax-inter_xmin) * (inter_ymax-inter_ymin)
            if metric_type == 'box_iof':
                tar_area = (tar_xmax-tar_xmin)*(tar_ymax-tar_ymin)
                metric_score = inter_area / tar_area
            elif metric_type == 'box_iou':
                union_area = (pre_xmax-pre_xmin)*(pre_ymax-pre_ymin) + \
                             (tar_xmax-tar_xmin)*(tar_ymax-tar_ymin) - inter_area
                metric_score = inter_area / union_area

        if metric_score >= metric_threshold[pre_id]:
            tar_labels[mask_tar] = 0 ## 防止一个gt被对应到多个predict,从而导致recall虚高
            tp_total_num += 1
            tp_class_num[pre_id] += 1
    ## label_num-1 去掉背景
    one_total_defect_f1 = np.array([tp_total_num, pre_total_num-1, tar_total_num-1], dtype=np.uint64)
    one_class_defect_f1 = np.array([tp_class_num, pre_class_num, tar_class_num], dtype=np.uint64)
    return one_total_defect_f1, one_class_defect_f1

def batch_tp_and_total(output, target, metric, defect_filter, nclass, stations):
    """Batch truePositive_and_total
    Args:
        output: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    total_num = np.zeros((3,), dtype=np.uint8)
    class_num = np.zeros((3,nclass), dtype=np.uint8)

    _, predict = torch.max(output, 1)
    # predict[predict == 2] = 1
    # target[target == 2] = 1
    # predict = output  ## 测试用
    b = predict.size()[0]

    for i in range(b):
        temp_predict = np.uint8(predict[i, :, :].cpu())
        temp_target = np.uint8(target[i, :, :].cpu())
        if np.count_nonzero(temp_predict) + np.count_nonzero(temp_target) == 0:
            continue ## 不计算predict和target都是无缺陷的情况
        if stations == []:
            station = 'no'
        else:
            station = stations[i]
        b_total_num, b_class_num = np_oneimg_tp_pre_tar(temp_predict, temp_target,
                                              metric, defect_filter, nclass, station, ignore_index=[0])
        total_num += b_total_num
        class_num += b_class_num

    return total_num, class_num


def zfb_seg2cls(output, target):
    N, C, H, W = output.size()
    fore_nclass = C-1
    _, predict = torch.max(output, 1)
    label, _ = torch.max(target.view(N, -1), 1)
    # predict = predict.cpu()
    tar_label = np.uint8(label.cpu()-1)
    pred_label = np.zeros(tar_label.shape, dtype=np.uint8)
    for n in range(N):
        n_pred = predict[n]
        label_num = np.zeros(fore_nclass, dtype=np.uint64)
        for i in range(fore_nclass):
            fore_label = i+1
            label_num[i] = torch.nonzero(n_pred == fore_label).size(0)
        pred_label[n] = np.argmax(label_num)
    return pred_label, tar_label

def zfb_label2confusionMatric(pred_label, tar_label):
    cls_acc = np.count_nonzero(tar_label == pred_label)/len(pred_label)
    my_cfm = np.zeros((3, 3))
    cfm = metrics.confusion_matrix(tar_label, pred_label, labels=[0, 1])
    my_cfm[:2, :2] = cfm
    my_cfm[0, 2] = my_cfm[0, 0] / sum(my_cfm[0, :])
    my_cfm[1, 2] = my_cfm[1, 1] / sum(my_cfm[1, :])
    my_cfm[2, 0] = my_cfm[0, 0] / sum(my_cfm[:, 0])
    my_cfm[2, 1] = my_cfm[1, 1] / sum(my_cfm[:, 1])
    my_cfm[2, 2] = cls_acc

    return my_cfm


def time_test_num(num = 100):
    start = time.time()
    for _ in range(num):
        point_list = np.where(mask_index != 0)
        id = target[point_list[0][0], point_list[1][0]]
    end1 = time.time()
    print((end1 - start)/num)

def cls_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
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




# # Metrics for one image
# def per_img_metrics(defect_mask, img_gt):
#     one = np.ones(defect_mask.shape, np.uint8)
#     Inter = one[(defect_mask>0)*(img_gt>0)]
#     Union = one[(defect_mask>0)+(img_gt>0)]
#     print(np.sum(Inter), np.sum(Union))
#     IoU = np.sum(Inter)/np.sum(Union)
#     if np.sum(Inter) == 0 and np.sum(Union) == 0 :
#         IoU = 1
    
#     TP = np.sum(one[(defect_mask>0)*(img_gt>0)])
#     FP = np.sum(one[(defect_mask>0)*(img_gt==0)])
#     FN = np.sum(one[(defect_mask==0)*(img_gt>0)])

#     precise = TP/(TP+FP)
#     recall = TP/(TP+FN)
#     if (TP+FP)==0:
#         precise = 0
#     if (TP+FN)==0:
#         recall = 0
#     if recall==0 and precise==0:
#         F1 = 0
#     else:
#         F1 = 2*(precise*recall)/(precise + recall)
#     return IoU, precise, recall, F1

# def defect_metrics(results, gt_seg_maps, ignore_index, nan_to_num=None):
#     """Calculate defect metrics (IoU, precision, recall, F1)

#     Args:
#         results (list[ndarray]): List of prediction segmentation maps
#         gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
#         ignore_index (int): Index that will be ignored in evaluation.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.

#      Returns:
#          float: Overall accuracy on all images.
#          ndarray: Per category accuracy, shape (num_classes, )
#          ndarray: Per category IoU, shape (num_classes, )
#     """

#     num_imgs = len(results)
#     assert len(gt_seg_maps) == num_imgs
#     mIoU, mPrecision, mRecall, mF1 = [], [], [], []
#     for i in range(num_imgs):
#         IoU, precision, recall, F1 = per_img_metrics(results[i], gt_seg_maps[i])
#         mIoU.append(IoU)
#         mPrecision.append(precision)
#         mRecall.append(recall)
#         mF1.append(F1)

#     mIoU, mPrecision, mRecall, mF1 = np.nanmean(mIoU), np.nanmean(mPrecision), np.nanmean(mRecall), np.nanmean(mF1)
#     # if nan_to_num is not None:
#     #     return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
#     #         np.nan_to_num(iou, nan=nan_to_num)
#     return mIoU, mPrecision, mRecall, mF1

# def per_img_metrics(defect_mask, img_gt, ignore_index):
#     mask = (img_gt != ignore_index)
#     defect_mask = defect_mask[mask]
#     img_gt = img_gt[mask]

#     one = np.ones(defect_mask.shape, np.uint8)
#     Inter = np.sum(one[(defect_mask>0)*(img_gt>0)])
#     Union = np.sum(one[(defect_mask>0)+(img_gt>0)])
#     # print(Inter, Union)
#     TP = np.sum(one[(defect_mask>0)*(img_gt>0)])
#     FP = np.sum(one[(defect_mask>0)*(img_gt==0)])
#     FN = np.sum(one[(defect_mask==0)*(img_gt>0)])

#     return np.array([Inter, Union, TP, FP, FN])  

# def defect_metrics(results, gt_seg_maps, ignore_index, nan_to_num=None):
#     """Calculate defect metrics (IoU, precision, recall, F1)

#     Args:
#         results (list[ndarray]): List of prediction segmentation maps
#         gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
#         ignore_index (int): Index that will be ignored in evaluation.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.

#      Returns:
#          float: Overall accuracy on all images.
#          ndarray: Per category accuracy, shape (num_classes, )
#          ndarray: Per category IoU, shape (num_classes, )
#     """

#     num_imgs = len(results)
#     assert len(gt_seg_maps) == num_imgs
#     allimgs_metrics = np.zeros((5, ), dtype=np.float) #[Inter, Union, TP, FP, FN]
#     for i in range(num_imgs):
#         per_results = per_img_metrics(results[i], gt_seg_maps[i], ignore_index)
#         allimgs_metrics += per_results

#     Inter, Union, TP, FP, FN = list(allimgs_metrics)

#     IoU = Inter/Union
#     if Inter == 0 and Union == 0 :
#         IoU = 1
#     precise = TP/(TP+FP)
#     recall = TP/(TP+FN)
#     if (TP+FP)==0:
#         precise = 0
#     if (TP+FN)==0:
#         recall = 0
#     if recall==0 and precise==0:
#         F1 = 0
#     else:
#         F1 = 2*(precise*recall)/(precise + recall)

#     # if nan_to_num is not None:
#     #     return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
#     #         np.nan_to_num(iou, nan=nan_to_num)
#     # print(Inter, Union)
#     return IoU, precise, recall, F1