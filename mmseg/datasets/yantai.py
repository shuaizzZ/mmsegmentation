
import time
import random
import os.path as osp
from tqdm import tqdm
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import mean_iou, SegmentationMetric
from mmseg.utils import get_root_logger

from torch.utils.data import Dataset
# from encoding.utils.files import *
# from encoding.transforms.augmentation import augment
from .pipelines import Compose
from .builder import DATASETS

CLASSES = ['background', '1diaojiao', '2liewen', '3kongdong', '4jiaza',
           '5tongyin', '6naobu', '7xiliewen', '8shengxiu', '9baicha']

@DATASETS.register_module()
class YantaiDataset(Dataset):
    def __init__(self,
                 pipeline,
                 classes=None,
                 labels=None,
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
        self.labels = labels
        self.PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                   [128, 0, 128], ]
        self.pipeline = Compose(pipeline)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.test_mode = test_mode

        self.ok_ori_samples = []
        self.ng_ori_samples = []
        self.class_samples = {}
        self.total_index = 0

        dataset_root = osp.join(self.data_root, 'yantai_datasets', dataset)
        _split_file = osp.join(dataset_root, '{}.csv'.format(self.split))
        if not osp.isfile(_split_file):
            raise ValueError('Unknown dataset _split_file: {}'.format(_split_file))

        with open(_split_file, "r") as lines:
            lines = list(lines)
        print(_split_file)
        for line in tqdm(lines, desc=self.split):
            ##-- 读取_image, _mask --##
            _image, _mask, label = line.rstrip('\n').split(',')
            st, date = osp.basename(_image).split('_')[:2]
            _image = osp.join(self.data_root, _image)
            _mask = osp.join(self.data_root, _mask)
            if not osp.isfile(_image):
                print('image error: {}'.format(_image))
                continue
            if not osp.isfile(_mask):
                print('mask error: {}'.format(_mask))
                continue

            sample = dict(filename=_image,
                          label=label,
                          st=st,
                          date=date,
                          ann=dict(seg_map=_mask))
            if self.split == 'train':
                if label == 'OK':
                    self.ok_ori_samples.append(sample)
                elif label not in self.class_samples.keys():
                    label_init = {label: [sample]}
                    self.class_samples.update(label_init)
                elif label in self.class_samples.keys():
                    self.class_samples[label].append(sample)
                else:
                    raise ValueError('Unkonwn label type !!!')
            elif self.split == 'val':
                if label == 'OK':
                    self.ok_ori_samples.append(sample)
                else:
                    self.ng_ori_samples.append(sample)
            else:
                raise ValueError('Unkonwn dataset split : {}'.format(self.split))
        ##---------------- ok取部分 ----------------##
        self.ok_ori_len = len(self.ok_ori_samples)
        if '06' in dataset:
            self.ok_len = int(self.ok_ori_len * 0.15)
            self.shift = [7, 4]
        elif '12' in dataset:
            self.ok_len = int(self.ok_ori_len * 0.2)
            # self.shift = [7, 4]
            self.shift = [4, 2]
        elif '57' in dataset:
            self.ok_len = int(self.ok_ori_len * 0.2)
            # self.shift = [7, 4]
            self.shift = [6, 3]

        ##---------------- 训练集类别平衡 ----------------##
        if self.split == 'train':
            self.pre_class_balance(max_times=10)
            self.epoch_balance()
            self.ng_len = len(self.ng_samples)
            if '06' in dataset:
                self.ok_len = int(self.ok_ori_len * 0.15)
                self.shift = [7, 4]
            elif '12' in dataset:
                self.ok_len = int(self.ok_ori_len * 0.2)
                # self.shift = [7, 4]
                self.shift = [4, 2]
            elif '57' in dataset:
                self.ok_len = int(self.ok_ori_len * 0.2)
                # self.shift = [7, 4]
                self.shift = [6, 3]
        elif self.split == 'val':
            self.img_infos = self.ok_ori_samples + self.ng_ori_samples
            self.ng_len = len(self.ng_ori_samples)
            self.ok_len = self.ok_ori_len
        self.set_len = len(self.img_infos)
        assert self.set_len == self.ok_len + self.ng_len
        print('ok: {}, ng: {}, total: {}'.format(self.ok_len, self.ng_len, self.set_len))
        time.sleep(2)

    def pre_class_balance(self, max_times=10):
        out_str = ''
        sort_samples = []
        ng_num = 0
        for k, v in self.class_samples.items():
            out_str += '{}: {}, '.format(k, len(v))
            sort_samples.append(v)
            ng_num += len(v)
        self.class_avg_num = int(ng_num / len(self.class_samples.keys()))
        print(out_str)
        # 按长度升序
        sort_samples.sort(key=lambda x: len(x))
        class_times = [self.class_avg_num / len(sort_samples[i]) for i in range(len(sort_samples))]
        print('class_times : ', class_times)
        class_times = [min(round(t), max_times) for t in class_times] # round 4舍5入
        self.major_smaples = sort_samples[-1]
        self.few_samples = []
        for i, t in enumerate(class_times[:-1]):
            t = max(t+1, 1)
            smaples = sort_samples[i] * t
            self.few_samples += smaples[:]

    def epoch_balance(self, ):
        self.ok_samples = random.sample(self.ok_ori_samples, self.ok_len)
        random.shuffle(self.major_smaples)
        self.ng_samples = self.few_samples + self.major_smaples[:self.class_avg_num]
        self.img_infos = self.ok_samples + self.ng_samples
        random.shuffle(self.img_infos)

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
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
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

        # all_acc, acc, iou = mean_iou(
        #    results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        # print_metrics(logger, all_acc, acc, iou, self.CLASSES, num_classes)

        ###----------------------- 参数初始化 -----------------------##
        self.metrics = SegmentationMetric(num_classes, kwargs['defect_metric'], kwargs['defect_filter'],
                                          ignore_index=[0], com_f1=kwargs['com_f1'])

        ###------------------------- segmentation_batch_eval ------------------------###
        self.metrics.reset()
        for predicts, targets in zip(results, gt_seg_maps):
            predicts = np.expand_dims(predicts, 0)
            targets = np.expand_dims(targets, 0)
            self.metrics.update_batch_metrics(predicts, targets)
        eval_results['Acc'], eval_results['IoU'], eval_results['Recall'], eval_results['Precision'], eval_results['F1'] = self.metrics.get_epoch_results()

        eval_results['ClassName'] = self.CLASSES

        return eval_results

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            if self.split == 'train':
                self.total_index += 1
                if self.total_index % self.set_len == 0:
                    self.epoch_balance()
            return self.prepare_train_img(idx)


    def __len__(self):
        return self.set_len