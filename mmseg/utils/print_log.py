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
    
def print_defect_metrics(log_dict, CLASSES=None, num_classes=2):
    all_acc, acc, iou, recall, precision, f1 = log_dict['aAcc'], log_dict['Acc'], log_dict['IoU'], log_dict['Recall'], log_dict['Precision'], log_dict['F1']
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
        
    best_iou_str = '{:.2f}'.format(log_dict['best_pred_mIoU'][0] * 100) + '/' + str(log_dict['best_pred_mIoU'][1])
    best_acc_str = '{:.2f}'.format(log_dict['best_pred_aAcc'][0] * 100) + '/' + str(log_dict['best_pred_aAcc'][1])
    best_recall_str = '{:.2f}'.format(log_dict['best_pred_total_recall'][0] * 100) + '/' + str(log_dict['best_pred_total_recall'][1])
    best_precision_str = '{:.2f}'.format(log_dict['best_pred_total_precision'][0] * 100) + '/' + str(log_dict['best_pred_total_precision'][1])
    best_f1_str = '{:.2f}'.format(log_dict['best_pred_total_F1'][0] * 100) + '/' + str(log_dict['best_pred_total_F1'][1])       
    
    summary_str += 'Summary:\n'
    line_format = '{:<15} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n'
    summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'tRecall', 'tPrecision', 'total_F1', 'pixAcc')

    iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
    acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
    recall_str = '{:.2f}'.format(log_dict['total_recall'] * 100)
    precision_str = '{:.2f}'.format(log_dict['total_precision'] * 100)
    f1_str = '{:.2f}'.format(log_dict['total_F1'] * 100)     
    all_acc_str = '{:.2f}'.format(all_acc * 100)
    summary_str += line_format.format('global', iou_str, acc_str, recall_str, precision_str, f1_str,
                                        all_acc_str)
    summary_str += line_format.format('best/epoch', best_iou_str, ' ', best_recall_str, best_precision_str, best_f1_str, best_acc_str)                                        
                                          
    return summary_str
    
def print_defect_loss(log_dict):
    summary_str = ''

    decode, aux = {}, {}
    for name, val in log_dict.items():
        if 'decode' in name:
            decode[name.split('.')[1]] = '{:.6f}'.format(val)
        if 'aux' in name:
            aux[name.split('.')[1]] = '{:.6f}'.format(val)

    line_format = '{:<15} '
    title, decode_val, aux_val = [], [], [] 
    for name, val in decode.items():
        line_format += '{:^'+str(len(name))+'} '
        title.append(name)
        decode_val.append(val)
        aux_val.append(aux[name])
    line_format += '\n'    
    summary_str += line_format.format('Type', *title)
    summary_str += line_format.format('decode', *decode_val)
    summary_str += line_format.format('aux', *aux_val) 
    summary_str += 'total_loss \t' + '{:.6f}'.format(log_dict['loss'])
                                          
    return summary_str    