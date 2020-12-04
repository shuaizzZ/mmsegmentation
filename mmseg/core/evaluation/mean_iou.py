import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect
    # print(area_intersect, area_union)

    return area_intersect, area_union, area_pred_label, area_label

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

def per_img_metrics(defect_mask, img_gt, ignore_index):
    mask = (img_gt != ignore_index)
    defect_mask = defect_mask[mask]
    img_gt = img_gt[mask]

    one = np.ones(defect_mask.shape, np.uint8)
    Inter = np.sum(one[(defect_mask>0)*(img_gt>0)])
    Union = np.sum(one[(defect_mask>0)+(img_gt>0)])
    # print(Inter, Union)
    TP = np.sum(one[(defect_mask>0)*(img_gt>0)])
    FP = np.sum(one[(defect_mask>0)*(img_gt==0)])
    FN = np.sum(one[(defect_mask==0)*(img_gt>0)])

    return np.array([Inter, Union, TP, FP, FN])  

def defect_metrics(results, gt_seg_maps, ignore_index, nan_to_num=None):
    """Calculate defect metrics (IoU, precision, recall, F1)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    allimgs_metrics = np.zeros((5, ), dtype=np.float) #[Inter, Union, TP, FP, FN]
    for i in range(num_imgs):
        per_results = per_img_metrics(results[i], gt_seg_maps[i], ignore_index)
        allimgs_metrics += per_results

    Inter, Union, TP, FP, FN = list(allimgs_metrics)

    IoU = Inter/Union
    if Inter == 0 and Union == 0 :
        IoU = 1
    precise = TP/(TP+FP)
    recall = TP/(TP+FN)
    if (TP+FP)==0:
        precise = 0
    if (TP+FN)==0:
        recall = 0
    if recall==0 and precise==0:
        F1 = 0
    else:
        F1 = 2*(precise*recall)/(precise + recall)

    # if nan_to_num is not None:
    #     return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
    #         np.nan_to_num(iou, nan=nan_to_num)
    # print(Inter, Union)
    return IoU, precise, recall, F1

def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
    """Calculate Intersection and Union (IoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    print(total_area_intersect, total_area_union)
    if nan_to_num is not None:
        return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
            np.nan_to_num(iou, nan=nan_to_num)
    return all_acc, acc, iou
