import os
import time
import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmcv.runner.dist_utils import master_only
from mmseg.core import mean_iou, SegmentationMetric
from mmseg.utils import get_root_logger, print_metrics

from .pipelines import Compose
from .builder import DATASETS

CLASSES = ['background', 'abnormal']
PALETTE = [[0, 0, 0], [0, 0, 255]]

@DATASETS.register_module()
class AinnoDataset(Dataset):
    def __init__(self,
                 pipeline,
                 data_root='',
                 dataset='',
                 classes=None,
                 labels=None,
                 palette=None,
                 split='train',
                 test_mode=None,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 reduce_zero_label=False,):

        self.data_root = data_root
        self.dataset = dataset
        self.CLASSES = classes if classes is not None else CLASSES
        self.PALETTE = palette if palette is not None else PALETTE
        self.labels=labels
        self.split = split
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

        self.img_infos = []
        self.invalid_img_infos = []
        self.dataset_infos = dict()

        self.load_annotations()
        self.print_dataset_info()


    def info2sample(self, line):
        _image, _mask = line.rstrip('\n').split(',')[:2]
        image_path = osp.join(self.data_root, _image)
        mask_path = osp.join(self.data_root, _mask)
        image_exist = self.check_image_exist(image_path)
        mask_exist = self.check_image_exist(mask_path)
        if image_exist and mask_exist:
            sample = dict(filename=image_path,
                          ann=dict(seg_map=mask_path))
            return sample
        return None

    def check_image_exist(self, image_path):
        if not osp.isfile(image_path):
            self.update_invalid_imgs(image_path)
            return False
        return True

    @master_only
    def update_invalid_imgs(self, image_path):
        self.invalid_img_infos.append(image_path)

    @master_only
    def print_dataset_info(self):
        if self.invalid_img_infos:
            invalid_str = 'The following picture does not exist:\n'
            for inv_path in self.invalid_img_infos:
                invalid_str += inv_path + '\n'
            print(invalid_str)
        str_info = '[{} / {}] - dataset_infos :\n'.format(self.__class__.__name__, self.split)
        prefix_len = len(str_info)
        for key, val in self.dataset_infos.items():
            str_info += '{}{}: {} \n'.format(' '*prefix_len, key, val)
        print(str_info)

    def load_annotations(self, ):
        # test_mode 改为分csv或者dir
        if not self.test_mode:
            _split_file = osp.join(self.data_root, '{}.csv'.format(self.split))
            if not osp.isfile(_split_file):
                raise ValueError('Unexist dataset _split_file: {}'.format(_split_file))
            with open(_split_file, "r") as lines:
                lines = list(lines)
            for line in lines:
                sample = self.info2sample(line)
                if not sample:
                    continue
                self.img_infos.append(sample)
        else:
            test_dir = osp.join(self.data_root, self.split)
            assert osp.isdir(test_dir), test_dir
            for img_name in os.listdir(test_dir):
                sample = dict(filename=osp.join(test_dir, img_name),
                              ann=dict(seg_map=None))
                self.img_infos.append(sample)

        self.set_len = len(self.img_infos)
        self.dataset_infos['set_len'] = self.set_len
        time.sleep(1)

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
            seg_map = img_info['ann']['seg_map']
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            for i, label in enumerate(self.labels):
                if label != i:
                    gt_seg_map[gt_seg_map==i] = label
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

        ###----------------------- SegmentationMetric -----------------------##
        self.metrics = SegmentationMetric(num_classes, kwargs['defect_metric'], kwargs['defect_filter'],
                                          ignore_index=[0], com_f1=kwargs['com_f1'])
        self.metrics.reset()
        for predicts, targets in zip(results, gt_seg_maps):
            predicts = np.expand_dims(predicts, 0)
            targets = np.expand_dims(targets, 0)
            self.metrics.update_batch_metrics(predicts, targets)
        eval_results['Acc'], eval_results['IoU'], eval_results['Recall'], eval_results['Precision'], eval_results['F1'] = self.metrics.get_epoch_results()

        eval_results['ClassName'] = self.CLASSES

        return  eval_results