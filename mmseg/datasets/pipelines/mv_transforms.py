
import cv2
import mmcv
import numpy as np
from numpy import random

from ..builder import PIPELINES

@PIPELINES.register_module()
class Relabel(object):
    def __init__(self,
                 labels=None,):
        self.labels = labels

    def modify_labels(self, mask):
        for idx, label in enumerate(self.labels):
            if idx != label:
                mask[mask == idx] = label
        assert np.max(mask) <= max(self.labels), '{} > {}'.format(np.max(mask), max(self.labels))
        return mask

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            results[key] = self.modify_labels(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(label_modify={self.label_modify})')
        return repr_str


@PIPELINES.register_module()
class MVRotate(object):
    """Rotate images & seg with random angle.

    This transform rotates the input image.

    Args:
        angle_range (tuple[int]): range of random angle.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
    """

    def __init__(self,
                 angle_range=(-30, 30),
                 center=None,):

        assert angle_range[0] <= angle_range[1]
        self.angle_range = angle_range
        self.center = center

    def _get_random_angle(self, results):
        angle = random.randint(self.angle_range[0], self.angle_range[1])
        results['rotate_angle'] = angle
        results['rotate_center'] = self.center

    def _rotate_img(self, results):
        """Rotate images with ``results['rotate_angle'] and results['rotate_center']``."""
        img = mmcv.imrotate(
            results['img'], results['rotate_angle'], self.center, interpolation='bilinear')
        results['img'] = img

    def _rotate_seg(self, results):
        """Rotate semantic segmentation map with ``results['rotate_angle']
        and results['rotate_center']``.
        """
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(
                results[key], results['rotate_angle'], interpolation='nearest')

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'rotate_angle', 'rotate_center', keys
            are added into result dict.
        """
        self._get_random_angle(results)
        self._rotate_img(results)
        self._rotate_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(angle_range={self.angle_range}, '
                     f'center={self.center})')
        return repr_str


@PIPELINES.register_module()
class MVResize(object):
    """Resize images & seg according to its' own size.

    This transform resizes the input image to some scale.

    Args:
        h_range (tuple[float]): (min_ratio, max_ratio) of height
        w_range (tuple[float]): (min_ratio, max_ratio) of width
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 h_range=(0.8, 1.2),
                 w_range=(0.8, 1.2),
                 keep_ratio=True):

        assert h_range[0] <= h_range[1]
        assert w_range[0] <= w_range[1]
        self.h_range = h_range
        self.w_range = w_range
        self.keep_ratio = keep_ratio

    def _get_random_size(self, results):
        h, w = results['img'].shape[:2]
        rh = random.randint(int(h * self.h_range[0]), int(h * self.h_range[1]))
        if self.keep_ratio:
            rw = int(rh / h * w)
        else:
            rw = random.randint(int(w * self.w_range[0]), int(w * self.w_range[1]))
        scale = rw, rh
        results['scale'] = scale
        results['scale_idx'] = None

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            gt_seg = mmcv.imresize(
                results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results:
            self._get_random_size(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(h_range={self.h_range}, '
                     f'w_range={self.w_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class MVCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        mode (string): random or center.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, crop_mode='random', cat_max_ratio=1., ignore_index=255,
                 pad_mode=['constant', 'constant'], pad_fill=[0, 0], pad_expand=1.0):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        assert crop_mode in ['random', 'center']
        self.crop_mode = crop_mode
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        # TODO range youwenti
        assert isinstance(pad_mode, list)
        assert all(pm in ['constant', 'range', 'select', ] for pm in pad_mode)
        self.pad_mode = pad_mode
        assert isinstance(pad_fill, list) and len(pad_fill) == 2
        self.pad_fill = pad_fill
        assert pad_expand >= 1.0
        self.pad_expand = pad_expand if crop_mode != 'center' else 1.0

    def _get_pad_border(self, results):
        h, w = results['img'].shape[:2]
        ch, cw = self.crop_size

        top = (ch - h) // 2 if ch - h > 0 else 0
        bottom = ch - h - top if ch - h - top > 0 else 0
        left = (cw - w) // 2 if cw - w > 0 else 0
        right = cw - w - left if cw - w - left > 0 else 0
        if self.pad_expand > 1:
            expand_h, expand_w = int(h * self.pad_expand - h), int(w * self.pad_expand - w)
            top, bottom = top + expand_h, bottom + expand_h
            left, right = left + expand_w, right + expand_w

        pad_border = (left, top, right, bottom)
        results['pad_border'] = pad_border

    @staticmethod
    def _get_pad_fill(mode, fill):
        if mode == 'constant':
            assert isinstance(fill, int)
            value = fill
        elif mode == 'range':
            assert mmcv.is_list_of(fill, int) and fill[0] < fill[1]
            value = random.randint(fill[0], fill[1])
        elif mode == 'select':
            assert mmcv.is_list_of(fill, int)
            value = random.choice(fill)

        return value

    def _pad_if_need(self, results):
        pad_border = results['pad_border']
        if sum(pad_border) == 0:
            return
        # pad the img
        img_fill = self._get_pad_fill(self.pad_mode[0], self.pad_fill[0])
        img = mmcv.impad(results['img'], padding=pad_border, pad_val=img_fill)
        results['img'] = img
        results['img_shape'] = img.shape
        # pad semantic seg
        seg_fill = self._get_pad_fill(self.pad_mode[1], self.pad_fill[1])
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], padding=pad_border, pad_val=seg_fill)

    def _get_random_bbox(self, img_size):
        """Randomly get a crop bounding box."""
        h, w = img_size
        ch, cw = self.crop_size

        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        x2, y2 = x1 + cw, y1 + ch

        return (y1, y2, x1, x2)

    def _get_center_bbox(self, img_size):
        """Get a center crop bounding box."""
        h, w = img_size
        ch, cw = self.crop_size
        x1 = (w - cw) // 2
        y1 = (h - ch) // 2
        x2, y2 = x1 + cw, y1 + ch

        return (y1, y2, x1, x2)

    def _get_crop_bbox(self, results):
        """Get a crop bounding box."""
        self._get_pad_border(results)
        self._pad_if_need(results)
        img_size = results['img'].shape[:2]
        crop_bbox = self.__getattribute__('_get_{}_bbox'.format(self.crop_mode))(img_size)
        # Repeat 10 times for cat_max_ratio
        if self.crop_mode == 'random' and self.cat_max_ratio < 1.:
            for _ in range(10):
                seg_temp = self._crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self._get_random_bbox(img)

        return crop_bbox

    def _crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        crop_bbox = self._get_crop_bbox(results)

        # crop the image
        img = results['img']
        img = self._crop(img, crop_bbox)
        results['img'] = img
        results['img_shape'] = img.shape
        results['crop_bbox'] = crop_bbox

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self._crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(crop_size={self.crop_size}, '
                     f'crop_mode={self.crop_mode}, '
                     f'cat_max_ratio={self.cat_max_ratio}, '
                     f'ignore_index={self.ignore_index}, '
                     f'pad_mode={self.pad_mode}, '
                     f'pad_fill={self.pad_fill}, '
                     f'pad_expand={self.pad_expand})')
        return repr_str


@PIPELINES.register_module()
class XYShift(object):
    def __init__(self,
                 shift=None,):
        self.shift = shift if isinstance(shift, tuple) else tuple(shift)

    def _get_random_shift(self, results):
        if not self.shift or self.shift==(0, 0):
            return
        sx = random.randint(-self.shift[0], self.shift[0])
        sy = random.randint(-self.shift[1], self.shift[1])
        results['shift_xy'] = (sx, sy)

    def _shift_img(self, results):
        sx, sy = results['shift_xy']
        img = results['img']
        h, w = img.shape[:2]
        ming, an, fusion = cv2.split(img) # opencv 读图按BGR

        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shift_an = cv2.warpAffine(an, M, (w, h))
        new_fusion = cv2.addWeighted(ming, 0.5, shift_an, 0.5, 0)
        img = cv2.merge((ming, shift_an, new_fusion))
        results['img'] = img
        # abs_shift = abs(sx) + abs(sy)

    def __call__(self, results):
        self._get_random_shift(results)
        if results.get('shift_xy'):
            self._shift_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(shift={self.shift})')
        return repr_str