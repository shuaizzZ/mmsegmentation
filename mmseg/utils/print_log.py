

def print_defect_metrics(log_dict, class_names=None):
    metrics = ['IoU', 'Acc', 'Recall', 'Precision', 'F1']
    acc, iou, recall, precision, f1 = \
        log_dict['Acc'], log_dict['IoU'], log_dict['Recall'], log_dict['Precision'], log_dict['F1']
    summary_str = ''
    summary_str += 'Validation results:\n'

    line_format = '{:<15} {:^10} {:^10} {:^10} {:^10} {:^10}\n'
    head_line = line_format.format('ClassName', 'IoU', 'Acc', 'Recall', 'Precision', 'F1')
    summary_str += head_line

    for i, class_name in enumerate(class_names):
        iou_str = '{:.2f}'.format(iou['class'][i] * 100)
        acc_str = '{:.2f}'.format(acc['class'][i] * 100)
        recall_str = '{:.2f}'.format(recall['class'][i] * 100)
        precision_str = '{:.2f}'.format(precision['class'][i] * 100)
        f1_str = '{:.2f}'.format(f1['class'][i] * 100)
        summary_str += line_format.format(class_name, iou_str, acc_str, recall_str, precision_str, f1_str)

    dash_line = '-' * (len(head_line) // 2 - len(' Summary ') // 2 - 1)
    summary_str += dash_line + ' Summary ' + dash_line + '\n'
    for name in ['mean', 'sum']:
        iou_str = '{:.2f}'.format(iou[name] * 100)
        acc_str = '{:.2f}'.format(acc[name] * 100)
        recall_str = '{:.2f}'.format(recall[name] * 100)
        precision_str = '{:.2f}'.format(precision[name] * 100)
        f1_str = '{:.2f}'.format(f1[name] * 100)
        summary_str += line_format.format(name, iou_str, acc_str, recall_str, precision_str, f1_str)

    best_iou_str = '{:.2f}/{}'.format(log_dict['best_pred_IoU'][0] * 100, log_dict['best_pred_IoU'][1])
    best_acc_str = '{:.2f}/{}'.format(log_dict['best_pred_Acc'][0] * 100, log_dict['best_pred_Acc'][1])
    best_recall_str = '{:.2f}/{}'.format(log_dict['best_pred_Recall'][0] * 100, log_dict['best_pred_Recall'][1])
    best_precision_str = '{:.2f}/{}'.format(log_dict['best_pred_Precision'][0] * 100, log_dict['best_pred_Precision'][1])
    best_f1_str = '{:.2f}/{}'.format(log_dict['best_pred_F1'][0] * 100, log_dict['best_pred_F1'][1])
    summary_str += line_format.format('best/epoch', best_iou_str, best_acc_str, best_recall_str, best_precision_str, best_f1_str)                                        
                                          
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