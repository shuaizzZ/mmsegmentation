import os
import time
import random
import os.path as osp
from tqdm import tqdm
from functools import reduce

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmseg.core import mean_iou, SegmentationMetric
from mmseg.utils import get_root_logger, print_metrics

from torch.utils.data import Dataset
from .pipelines import Compose
from .builder import DATASETS

CLASSES = ['background', '1diaojiao', '2liewen', '3kongdong', '4jiaza',
           '5tongyin', '6naobu', '7xiliewen', '8shengxiu', '9baicha']

@DATASETS.register_module()
class AinnoDataset(Dataset):
    def __init__(self,
                 pipeline,
                 classes=None,
                 split='train',
                 test_mode=None,
                 dataset='',
                 data_root='',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 reduce_zero_label=False,):

        if classes is not None:
            self.CLASSES = classes
        else:
            self.CLASSES = CLASSES
        self.PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                   [128, 0, 128], ]
        self.pipeline = Compose(pipeline)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.test_mode = test_mode
        self.img_infos = []

        if not self.test_mode:
            _split_file = osp.join(self.data_root, '{}.csv'.format(self.split))
            if not osp.isfile(_split_file):
                raise ValueError('Unknown dataset _split_file: {}'.format(_split_file))
            with open(_split_file, "r") as lines:
                lines = list(lines)
            for line in tqdm(lines, desc=self.split):
                ##-- 读取_image, _mask --##
                _image, _mask = line.rstrip('\n').split(',')[:2]
                _image = osp.join(self.data_root, _image)
                _mask = osp.join(self.data_root, _mask)
                if not osp.isfile(_image):
                    print('image error: {}'.format(_image))
                    continue
                if not osp.isfile(_mask):
                    print('mask error: {}'.format(_mask))
                    continue
                sample = dict(filename=_image,
                              ann=dict(seg_map=_mask))
                self.img_infos.append(sample)
        else:
            test_dir = osp.join(self.data_root, self.split)
            assert osp.isdir(test_dir), test_dir
            for img_name in os.listdir(test_dir):
                sample = dict(filename=osp.join(test_dir, img_name),
                              ann=dict(seg_map=None))
                self.img_infos.append(sample)

        self.set_len = len(self.img_infos)
        time.sleep(1)

    def load_annotations(self, ):
        pass

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        # results['img_prefix'] = self.img_dir
        # results['seg_prefix'] = self.ann_dir
        # if self.custom_classes:
        #     results['label_map'] = self.label_map

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def __len__(self):
        return self.set_len


    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            # TODO
            # seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            seg_map = img_info['ann']['seg_map']
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        #all_acc, acc, iou = mean_iou(
        #    results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        #print_metrics(logger, all_acc, acc, iou, self.CLASSES, num_classes)
        
        ###----------------------- 参数初始化 -----------------------##
        self.metrics = SegmentationMetric(num_classes, kwargs['defect_metric'], kwargs['defect_filter'],
                                          ignore_index=[0], com_f1=kwargs['com_f1'])

        ###------------------------- segmentation_batch_eval ------------------------###
        self.metrics.reset()
        self.metrics.update_batch_metrics(results, gt_seg_maps, station=[])
        pixAcc, class_pixAcc, mIoU, class_iou, auc, total_recall, class_recall, total_precision, \
                    class_precision, total_F1, class_F1 = self.metrics.get_epoch_results()

        eval_results['ClassName'] = self.CLASSES
        eval_results['IoU'] = class_iou
        eval_results['Acc'] = class_pixAcc
        eval_results['pixAcc'] = pixAcc

        eval_results['Precision'] = class_precision
        eval_results['Recall'] = class_recall
        eval_results['F1'] = class_F1

        eval_results['mIoU'] = mIoU
        eval_results['total_precision'] = total_precision
        eval_results['total_recall'] = total_recall
        eval_results['total_F1'] = total_F1    

        return  eval_results


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def __len__(self):
        return self.set_len