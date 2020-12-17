
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
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from mmcv.runner.dist_utils import allreduce_params, master_only
import numpy as np

class CheckRunstateHook(Hook):
    def __init__(self, runstate):
        self.runstate = runstate

    def before_iter(self, runner):
        if self.runstate[0] == 0:
            sys.exit(0)


class TrainerLogHook(Hook):
    def __init__(self, trainer_csv_path, ndigits=3):
        self.trainer_csv_path = trainer_csv_path
        self.ndigits = ndigits

    @master_only
    def before_run(self, runner):
        if os.path.isfile(self.trainer_csv_path):
            os.remove(self.trainer_csv_path)
        self.log_csv = CSV(self.trainer_csv_path)
        log_head = ['epoch'] + runner.best_metrics
        self.log_csv.append(log_head)

    @master_only
    def after_train_epoch(self, runner):
        log_info = [runner.epoch]
        runner.cur_eval_res['IoU']
        for name in runner.best_metrics:
            log_info.append(round(runner.cur_eval_res[name] * 100, self.ndigits))
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
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer

    def before_run(self, runner):
        runner.best_metrics = ['IoU', 'Acc', 'Recall', 'Precision', 'F1']
        runner.best_eval_res = {}
        runner.cur_eval_res = {}
        for name in runner.best_metrics:
            runner.best_eval_res[name] = [0, 0]
            runner.cur_eval_res[name] = 0

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        # runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
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

        for name, val in runner.best_eval_res.items():
            if name not in runner.best_metrics:
                continue
            cur_val = runner.log_buffer.output[name]['mean']
            runner.cur_eval_res[name] = cur_val
            if val[0] <= cur_val:
                runner.best_eval_res[name] = [cur_val, runner.epoch+ 1]
                runner.save_checkpoint(
                    self.out_dir, save_optimizer=self.save_optimizer,
                    filename_tmpl=f'{name}_best_model.pth.tar', **self.args)

                runner.logger.info(f'Saving {name}_best checkpoint at {runner.epoch + 1} epochs')
            runner.log_buffer.output['best_pred_'+name] = runner.best_eval_res[name]

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