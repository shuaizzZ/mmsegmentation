
import sys
import time
import shutil
import os
import os.path as osp

from yutils.csv.csv import CSV
import torch
from torch.optim import Optimizer

import mmcv
from mmcv.parallel import is_module_wrapper
from mmcv.utils import mkdir_or_exist
from mmcv.runner import HOOKS, Hook
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu #, save_checkpoint
from mmcv.runner.dist_utils import allreduce_params, master_only

class CheckRunstateHook(Hook):
    def __init__(self, runstate):
        self.runstate = runstate

    def before_iter(self, runner):
        if self.runstate[0] == 0:
            sys.exit(0)


class TrainerLogHook(Hook):
    def __init__(self, trainer_log_path):
        if os.path.isfile(trainer_log_path):
            os.remove(trainer_log_path)
        self.log_csv = CSV(trainer_log_path)
        log_head = ['epoch', 'iou']
        self.log_csv.append(log_head)

    def after_train_epoch(self, runner):
        log_info = [runner.epoch, runner.log_buffer.output['mIoU']]
        self.log_csv.append(log_info)


@HOOKS.register_module()
class TrainerCheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 sync_buffer=False,
                 config=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.config = config

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        trainer_model_path = osp.join(self.out_dir, 'F1_best_model.pth')
        save_checkpoint(runner.model, trainer_model_path,
                        optimizer=runner.optimizer, meta=runner.meta, config=self.config)
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = os.path.join(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(_step))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        runner.logger.info(
            f'Saving checkpoint at {runner.iter + 1} iterations')
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner)

def save_checkpoint(model, filename, optimizer=None, meta=None, config=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    mmcv.mkdir_or_exist(osp.dirname(filename))
    if is_module_wrapper(model):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()
    # save config in the checkpoint
    checkpoint['config'] = config
    # immediately flush buffer
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()