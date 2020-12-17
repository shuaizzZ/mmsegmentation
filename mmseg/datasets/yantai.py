
import time
import random
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.ainno import AinnoDataset

CLASSES = ['background', '1diaojiao', '2liewen', '3kongdong', '4jiaza',
           '5tongyin', '6naobu', '7xiliewen', '8shengxiu', '9baicha']
PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], ]

@DATASETS.register_module()
class YantaiDataset(AinnoDataset):
    def __init__(self,
                 classes=CLASSES,
                 palette=PALETTE,
                 **kwargs):
        super(YantaiDataset, self).__init__(
            classes=classes,
            palette=palette,
            **kwargs)


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
        # a = sorted(self.class_samples.items(), key=lambda x: x[1])
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
        st, date = osp.basename(_image).split('_')[:2]
        image_path = osp.join(self.data_root, _image)
        mask_path = osp.join(self.data_root, _mask)
        image_exist = self.check_image_exist(image_path)
        mask_exist = self.check_image_exist(mask_path)
        if image_exist and mask_exist:
            sample = dict(filename=image_path,
                          label=label,
                          st=st,
                          date=date,
                          ann=dict(seg_map=mask_path))
            return sample
        return None

    def update_samples(self, sample):
        if not sample:
            return
        label = sample['label']
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

    def load_annotations(self, ):
        self.ok_ori_samples = []
        self.ng_ori_samples = []
        self.class_samples = {}
        self.total_index = 0

        dataset_root = osp.join(self.data_root, 'yantai_datasets', self.dataset)
        _split_file = osp.join(dataset_root, '{}.csv'.format(self.split))
        if not osp.isfile(_split_file):
            raise ValueError('Unexist dataset _split_file: {}'.format(_split_file))
        with open(_split_file, "r") as lines:
            lines = list(lines)
        for line in lines:
            sample = self.info2sample(line)
            self.update_samples(sample)

        ##---------------- 训练集类别平衡 ----------------##
        self.ok_ori_len = len(self.ok_ori_samples)
        if self.split == 'train':
            if '06' in self.dataset:
                self.ok_len = int(self.ok_ori_len * 0.15)
            elif '12' in self.dataset:
                self.ok_len = int(self.ok_ori_len * 0.2)
            elif '57' in self.dataset:
                self.ok_len = int(self.ok_ori_len * 0.2)
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
        time.sleep(2)


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