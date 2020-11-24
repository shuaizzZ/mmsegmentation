# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "psp"
## 'psp', 'fcn', 'deeplab', 'encnet', 'icnet', 'bisenet'
_C.MODEL.BACKBONE = "resnet18"
## 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
## 'densenet121', 'densenet161', 'densenet196', 'resnet201'
_C.MODEL.PRETRAIN_BACKBONE = True
_C.MODEL.PRETRAINED_DIR = ""
_C.MODEL.DU_SCALE = 8
_C.MODEL.SPP_SIZE = [1, 2, 3, 6]
_C.MODEL.DROPOUT2d = 0.1
_C.MODEL.CHANNEL_PRUNE_RATIO = 1.0
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT = "~/.encode/data/clothes/"
_C.DATASETS.TYPE = 'clothes'
_C.DATASETS.CLASSES = []
_C.DATASETS.LABELS = []
_C.DATASETS.STATIONS = []
_C.DATASETS.WORKS = 16
_C.DATASETS.BASE_SIZE = 0
_C.DATASETS.INCHANNEL = 3
_C.DATASETS.MEAN = [0.485,0.456,0.406]
_C.DATASETS.STD = [0.229,0.224,0.225]
_C.DATASETS.PLUS_TRADITIONAL_CHANNEL = False

_C.DATASETS.DEFECTSIMULATE = CN()
_C.DATASETS.DEFECTSIMULATE.ENABLE = False
_C.DATASETS.DEFECTSIMULATE.MODE = []
_C.DATASETS.DEFECTSIMULATE.NUM = []
_C.DATASETS.DEFECTSIMULATE.DENSITY = [0.0025, 0.001, 0.002, 0.005]
_C.DATASETS.DEFECTSIMULATE.SIZE = []

_C.DATASETS.AUGMENT = CN()
_C.DATASETS.AUGMENT.TRANSPOSE = [0,1]
_C.DATASETS.AUGMENT.CROP_SIZE = [480, 480] # h, w
_C.DATASETS.AUGMENT.RESIZE_MODE = 1
_C.DATASETS.AUGMENT.RESIZE_RANGE_H = [0.8, 1.2]
_C.DATASETS.AUGMENT.RESIZE_RANGE_W = [0.8, 1.2]
_C.DATASETS.AUGMENT.ROTATE_RANGE = [90, 90]
_C.DATASETS.AUGMENT.BRIGHT_RANGE = [0.9, 1.1]
_C.DATASETS.AUGMENT.CONTRAST_RANGE = [0.9, 1.1]
_C.DATASETS.AUGMENT.COLOR_RANGE = [0.9, 1.1]
_C.DATASETS.AUGMENT.SHARP_RANGE = [0.9, 1.1]

_C.DATASETS.MORPH = CN()
_C.DATASETS.MORPH.MORPH_MODE = 1
_C.DATASETS.MORPH.KERNEL_TYPE = 0
_C.DATASETS.MORPH.KERNEL_SIZE = [5, 5]
_C.DATASETS.MORPH.MORPH_ITER = 1

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.FUN_LIST = ['SegmentationFocalLosses']
_C.LOSS.WEIGHT_LIST = []
## 'CrossEntropyLoss', 'SegmentationFocalLosses', 'ICNetLoss'
_C.LOSS.AUTO_WEIGHT = False

_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.GAMA = [2.0]
_C.LOSS.FOCAL.ALPHA = [0.5]
_C.LOSS.FOCAL.SIZE_AVERAGE = False

_C.LOSS.LovaszLoss = CN()
_C.LOSS.LovaszLoss.PER_IMAGE = False
_C.LOSS.LovaszLoss.ALPHA = [0.5]
_C.LOSS.LovaszLoss.FOCAL_GAMA = 2.0

_C.LOSS.CrossEntropyLoss = CN()
_C.LOSS.CrossEntropyLoss.WEIGHT = [1.0, 1.0, 1.0]

_C.LOSS.ICNETLOSS = CN()
_C.LOSS.ICNETLOSS.WEIGHT = [0.16, 0.4, 1.0]

_C.LOSS.AUX = True
_C.LOSS.AUX_WEIGHT = [1.0, 1.0, 1.0]
_C.LOSS.SE = False
_C.LOSS.SE_WEIGHT = 0.2

_C.LOSS.OHEM = CN()
_C.LOSS.OHEM.MODE = 0  ## 0代表不使用, 1代表pt阈值模式, 2代表百分比阈值模式
_C.LOSS.OHEM.THRESHOLD = 0.60
_C.LOSS.OHEM.KEEP = 2e5

# TODO
_C.LOSS.LABEL_SMOOTHING = 0.00
# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.LR = CN()
_C.SOLVER.LR.BASE_LR = 0.01
_C.SOLVER.LR.ADJUST_LR = 1.0
_C.SOLVER.LR.UPDATE_POLICY = 'LR_Scheduler_Batch'
## 'LR_Scheduler_Batch', 'LR_Scheduler_Epoch'
_C.SOLVER.LR.LR_SCHEDULER = "poly"
## "poly", "step", "cos", "auto"

_C.SOLVER.LR.CYCLE_LR = True
_C.SOLVER.LR.CYCLE_LR_STEP = 50

_C.SOLVER.LR.POLY = CN()
_C.SOLVER.LR.POLY.POWER = 0.9

_C.SOLVER.LR.STEP = CN()
_C.SOLVER.LR.STEP.LR_STEP = 1
_C.SOLVER.LR.STEP.LR_DECAY = 0.95

_C.SOLVER.OPT = CN()
_C.SOLVER.OPT.OPTIMIZER = "sgd"

_C.SOLVER.OPT.MOMENTUM = 0.9
_C.SOLVER.OPT.WEIGHT_DECAY = 5e-4

_C.SOLVER.WARMUP = CN()
_C.SOLVER.WARMUP.WARMUP = False
_C.SOLVER.WARMUP.POWER = 0.01
_C.SOLVER.WARMUP.WARMUP_EPOCH = 10

_C.SOLVER.DU_WARMUP = CN()
_C.SOLVER.DU_WARMUP.WARMUP = False
_C.SOLVER.DU_WARMUP.WARMUP_LR = 0.5
_C.SOLVER.DU_WARMUP.WARMUP_EPOCH = 10

# ---------------------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.FOCAL_INIT = CN()
_C.TRAIN.FOCAL_INIT.WITH_FOCAL_INIT = False
_C.TRAIN.FOCAL_INIT.PI = 0.01
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.BATCH_SIZE = 32

_C.TRAIN.AMP = False
_C.TRAIN.ACCUMULATION_STEPS = 1

_C.TRAIN.ADAP_LR = CN()
_C.TRAIN.ADAP_LR.LOSS_ERROR_THR = 1.4
_C.TRAIN.ADAP_LR.LR_DECAY = 0.8
_C.TRAIN.DATA_SAMPLER = CN()
## 是否自定义数据加载比例, SAMPLE_NUMS的长度要和CLASSES长度一致, 且采样个数与CLASSES一一对应
_C.TRAIN.DATA_SAMPLER.CUSTOM = False
_C.TRAIN.DATA_SAMPLER.LOOP_LOADER = False
_C.TRAIN.DATA_SAMPLER.SAMPLE_NUMS = []
_C.TRAIN.DATA_SAMPLER.ITRE_PER_EPOCH = 100
_C.TRAIN.FT = CN()
_C.TRAIN.FT.RESUME = False
_C.TRAIN.FT.CHECKPATH = ''
_C.TRAIN.FT.CONTINUE_TRAIN = False
## 如果继续训练, CHECKNAME要与之前相同,不同的话就会新建一个model目录
_C.TRAIN.CHECKNAME = ""

# ---------------------------------------------------------------------------- #
# VAL
# ---------------------------------------------------------------------------- #
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 1
_C.VAL.VAL_START_EPOCH = 100
_C.VAL.VAL_FREQUENCY = 1

_C.VAL.IN_SIZE = CN()
_C.VAL.IN_SIZE.MODE = 'crop'
_C.VAL.IN_SIZE.SIZE = [1024, 1024]

_C.VAL.COM_F1 = True
_C.VAL.BEST_TYPE = 'F1'

_C.VAL.METRIC = CN()
_C.VAL.METRIC.TYPE = 'pix_iou'
_C.VAL.METRIC.THRESHOLD = []

_C.VAL.DEFECT_FILTER = CN()
_C.VAL.DEFECT_FILTER.STATION = True
_C.VAL.DEFECT_FILTER.TYPE = 'box'
_C.VAL.DEFECT_FILTER.SIZE_ALL = [16, 16]
_C.VAL.DEFECT_FILTER.SIZE_STATION = []

# ---------------------------------------------------------------------------- #
# OTHERS
# ---------------------------------------------------------------------------- #
_C.CUDA = True
_C.GPU = [0,1]
_C.SEED = 1
_C.MODEL_ZOO = ""
_C.EVAL = False
_C.NO_VAL = False
_C.TEST_FOLDER = ""