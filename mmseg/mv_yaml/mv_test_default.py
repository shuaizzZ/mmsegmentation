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

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.ROOT = ""

_C.TEST.BEST_TYPE = 'F1'

_C.TEST.IN_SIZE = CN()
_C.TEST.IN_SIZE.MODE = 'crop'
_C.TEST.IN_SIZE.SIZE = [1024, 1024]
_C.TEST.IN_SIZE.SCALES = [1.0]
_C.TEST.IN_SIZE.BASESIZE = 0
_C.TEST.IN_SIZE.FLIP = False

_C.TEST.DEFECT_FILTER = CN()
_C.TEST.DEFECT_FILTER.TYPE = 'box'
_C.TEST.DEFECT_FILTER.SIZE = [16, 16]

_C.TEST.BATCH_SIZE = 1

# 默认是none,也可以使用有特色的自定的处理
_C.TEST.CHANNEL_PREPROCESS = 'none'
_C.TEST.CLASS_NUM = 0

# ---------------------------------------------------------------------------- #
# OTHERS
# ---------------------------------------------------------------------------- #
_C.GPU = [0,1]
