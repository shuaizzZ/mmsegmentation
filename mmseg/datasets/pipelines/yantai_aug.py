import mmcv
import numpy as np
from numpy import random

from ..builder import PIPELINES

@PIPELINES.register_module()
class Relabel(object):
    def __init__(self,
                 labels=None,):

        self.label_modify = []
        for i, label in enumerate(labels):
            if label != i:
                self.label_modify.append([i, label])

    def modify_labels(self, mask):
        if len(self.label_modify) > 0:
            for modify in self.label_modify:
                mask[mask == modify[0]] = modify[1]
        return mask

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            results[key] = self.modify_labels(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(label_modify={self.label_modify})')
        return repr_str