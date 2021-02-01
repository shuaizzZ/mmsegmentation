
import time
import random
import os
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.ainno import AinnoDataset

CLASSES = ['background', 'huahen', 'zangwu', 'laji']
LABELS = [0, 1, 2, 3]
PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0]]

@DATASETS.register_module()
class AnjieDataset(AinnoDataset):
    def __init__(self,
                 classes=CLASSES,
                 palette=PALETTE,
                 labels=LABELS,
                 **kwargs):
        super(AnjieDataset, self).__init__(
            classes=classes,
            palette=palette,
            labels=labels,
            **kwargs)


    def _get_split_file(self):
        self._split_file = osp.join(self.data_root, '{}/{}.csv'.format(self.dataset, self.split))
        if not osp.isfile(self._split_file):
            raise ValueError('Unexist dataset _split_file: {}'.format(self._split_file))

    def _pre_class_balance(self, max_times=10):
        class_nums_str = ''
        class_times_str = ''
        sort_samples = []
        ng_num = 0
        for k, v in self.class_samples.items():
            class_nums_str += '{}: {}, '.format(k, len(v))
            sort_samples.append(v)
            ng_num += len(v)
        self.class_avg_num = int(ng_num / len(self.class_samples.keys()))
        self.dataset_infos['class_nums'] = class_nums_str
        # 按长度升序
        sort_samples.sort(key=lambda x: len(x))
        class_times = [self.class_avg_num / len(sort_samples[i]) for i in range(len(sort_samples))]
        for t in class_times:
            class_times_str += '{:.2f}, '.format(t)
        self.dataset_infos['class_times'] = class_times_str
        class_times = [min(round(t), max_times) for t in class_times] # round 4舍5入
        self.major_smaples = sort_samples[-1]
        self.few_samples = []
        for i, t in enumerate(class_times[:-1]):
            t = max(t+1, 1)
            smaples = sort_samples[i] * t
            self.few_samples += smaples[:]

    def _epoch_balance(self, ):
        self.ok_samples = random.sample(self.ok_ori_samples, self.ok_len)
        random.shuffle(self.major_smaples)
        self.ng_samples = self.few_samples + self.major_smaples[:self.class_avg_num]
        self.img_infos = self.ok_samples + self.ng_samples
        random.shuffle(self.img_infos)

    def info2sample(self, line):
        _image, _mask, label = line.rstrip('\n').split(',')[:3]
        image_path = osp.join(self.data_root, _image)
        mask_path = osp.join(self.data_root, _mask)
        image_exist = self.check_image_exist(image_path)
        mask_exist = self.check_image_exist(mask_path)
        sample = None
        if image_exist and mask_exist:
            sample = dict(filename=image_path,
                          label=label,
                          ann=dict(seg_map=mask_path))
        return sample

    def update_samples(self, sample):
        if not sample:
            return
        label = sample['label']
        if self.split == 'train':
            if label == 'ok' or label == 'OK':
                self.ok_ori_samples.append(sample)
            elif label not in self.class_samples.keys():
                label_init = {label: [sample]}
                self.class_samples.update(label_init)
            elif label in self.class_samples.keys():
                self.class_samples[label].append(sample)
            else:
                raise ValueError('Unkonwn label type !!!')
        elif self.split == 'val':
            if label == 'ok' or label == 'OK':
                self.ok_ori_samples.append(sample)
            else:
                self.ng_ori_samples.append(sample)
        else:
            raise ValueError('Unkonwn dataset split : {}'.format(self.split))

    def load_annotations(self, ):
        self.ok_ori_samples = []
        self.ng_ori_samples = []
        self.class_samples = {}
        self.total_index = 0
        if not self.test_mode:
            self._get_split_file()
            if not osp.isfile(self._split_file):
                raise ValueError('Unexist dataset _split_file: {}'.format(self._split_file))
            with open(self._split_file, "r") as lines:
                lines = list(lines)
            for line in lines:
                sample = self.info2sample(line)
                self.update_samples(sample)
        else:
            # test_dir = osp.join(self.data_root, self.split)
            test_dir = '/root/public02/manuag/zhangshuai/data/anjie/real_data/train_split/image'
            assert osp.isdir(test_dir), test_dir
            for img_name in os.listdir(test_dir):
                sample = dict(filename=osp.join(test_dir, img_name),
                              ann=dict(seg_map=None))
                self.img_infos.append(sample)
                # self.update_samples(sample)
        ##---------------- 训练集类别平衡 ----------------##
        self.ok_ori_len = len(self.ok_ori_samples)
        if self.split == 'train':
            self.ok_len = int(self.ok_ori_len * 0.1)
            self._pre_class_balance(max_times=10)
            self._epoch_balance()
            self.ng_len = len(self.ng_samples)
        elif self.split == 'val':
            self.ok_len = self.ok_ori_len
            self.img_infos = self.ok_ori_samples + self.ng_ori_samples
            self.ng_len = len(self.ng_ori_samples)

        self.set_len = len(self.img_infos)
        assert self.set_len == self.ok_len + self.ng_len
        self.dataset_infos['sample_nums'] = '(OK: {}, NG: {} ,Total: {})'.format(
            self.ok_len, self.ng_len, self.set_len)
        time.sleep(1)

    def epoch_ops(self):
        """Some operations that need to be performed every n epochs. """
        print('_epoch_balance')
        self._epoch_balance()

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def __len__(self):
        return self.set_len