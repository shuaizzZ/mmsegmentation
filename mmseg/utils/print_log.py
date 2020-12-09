import numpy as np
from mmcv.utils import print_log

def print_metrics(logger, all_acc, acc, iou, CLASSES=None, num_classes=2):
    summary_str = ''
    summary_str += 'per class results:\n'

    line_format = '{:<15} {:>10} {:>10}\n'
    summary_str += line_format.format('Class', 'IoU', 'Acc')
    if CLASSES is None:
        class_names = tuple(range(num_classes))
    else:
        class_names = CLASSES
    for i in range(num_classes):
        iou_str = '{:.2f}'.format(iou[i] * 100)
        acc_str = '{:.2f}'.format(acc[i] * 100)
        summary_str += line_format.format(class_names[i], iou_str, acc_str)
    summary_str += 'Summary:\n'
    line_format = '{:<15} {:>10} {:>10} {:>10}\n'
    summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

    iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
    acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
    all_acc_str = '{:.2f}'.format(all_acc * 100)
    summary_str += line_format.format('global', iou_str, acc_str,
                                        all_acc_str)
    print_log(summary_str, logger)

def print_defect_metrics(logger, all_acc, acc, iou, recall, precision, f1, CLASSES=None, num_classes=2):
    summary_str = ''
    summary_str += 'per class results:\n'

    line_format = '{:<15} {:^10} {:^10} {:^10} {:^10} {:^10}\n'
    summary_str += line_format.format('Class', 'IoU', 'Acc', 'Recall', 'Precision', 'F1')
    if CLASSES is None:
        class_names = tuple(range(num_classes))
    else:
        class_names = CLASSES
    for i in range(num_classes):
        iou_str = '{:.2f}'.format(iou[i] * 100)
        acc_str = '{:.2f}'.format(acc[i] * 100)
        recall_str = '{:.2f}'.format(recall[i] * 100)
        precision_str = '{:.2f}'.format(precision[i] * 100)
        f1_str = '{:.2f}'.format(f1[i] * 100)      
        summary_str += line_format.format(class_names[i], iou_str, acc_str, recall_str, precision_str, f1_str)
    summary_str += 'Summary:\n'
    line_format = '{:<15} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n'
    summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'mRecall', 'mPrecision', 'mF1', 'aAcc')

    iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
    acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
    recall_str = '{:.2f}'.format(np.nanmean(recall) * 100)
    precision_str = '{:.2f}'.format(np.nanmean(precision) * 100)
    f1_str = '{:.2f}'.format(np.nanmean(f1) * 100)     
    all_acc_str = '{:.2f}'.format(all_acc * 100)
    summary_str += line_format.format('global', iou_str, acc_str, recall_str, precision_str, f1_str,
                                        all_acc_str)
    print_log(summary_str, logger)