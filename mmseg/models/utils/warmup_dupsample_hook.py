
import sys
import numpy as np
from apex import amp
from tqdm import tqdm
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import Hook, build_optimizer, get_dist_info
from mmcv.runner.dist_utils import master_only
from mmcv.runner.iter_based_runner import IterLoader
from mmseg.datasets.builder import build_dataloader, build_dataset
from mmseg.models.utils.dupsample_block import DUpsamplingBlock, MirrorDUpsamplingBlock
from mmseg.models.utils.custom_blocks import parallel_model

import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad

from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model


# 以后放工具类中
def reduce_loss_for_dist(loss_value):
    loss_value = loss_value.data.clone()
    dist.all_reduce(loss_value.div_(dist.get_world_size()))
    return loss_value


class AmpWarmUpDUpsampleHook(Hook):
    """WarmUpDUpsampleHook hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, model, cfg, distributed=False, runstate=np.array([1])):
        self._max_runs = cfg.warmup_du_cfg.total_runs
        self.dupsampleblock_list = self._find_dupsampleblocks(model)
        if len(self.dupsampleblock_list) * self._max_runs == 0:
            return

        self.runstate = runstate
        self.fp16_train = cfg.optimizer_config.get('type') == 'Fp16OptimizerHook'
        # self.model = model
        self.warmup_du_infos = []
        self.distributed = distributed

        self.interval = cfg.warmup_du_cfg.interval
        self.by_epoch = cfg.warmup_du_cfg.by_epoch
        self.log_head = 'Epoch' if self.by_epoch else 'Iter'
        self._iter = 1
        self._epoch = 1

        # build dataset
        dataset = build_dataset(cfg.data.train)
        # prepare data loaders
        self.data_loader = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=False)
        self.task_length = len(self.data_loader) if self.by_epoch else self._max_runs

        for dupsampleblock in self.dupsampleblock_list:
            warmup_du_model = MirrorDUpsamplingBlock(dupsampleblock).cuda(cfg.gpu_ids[0])
            optimizer = build_optimizer(warmup_du_model, cfg.warmup_du_cfg.optimizer)
            # put model on gpus
            if self.fp16_train:
                warmup_du_model, optimizer = amp.initialize(warmup_du_model, optimizer, opt_level="O1")
            warmup_du_model = parallel_model(warmup_du_model, cfg.gpu_ids, distributed)
            warmup_du_info = {'model': warmup_du_model, 'optimizer': optimizer, 'du_loss': []}
            self.warmup_du_infos.append(warmup_du_info)

    def _find_dupsampleblocks(self, model):
        dupsampleblock_list = []
        for m in model.modules():
            if isinstance(m, DUpsamplingBlock):
                dupsampleblock_list.append(m)

        return dupsampleblock_list

    def _run_one_batch(self, data_batch):
        # check runstate
        if self.runstate[0] == 0:
            sys.exit(0)

        for warmup_du_info in self.warmup_du_infos:
            warmup_du_info['optimizer'].zero_grad()
            seggt = data_batch['gt_semantic_seg'].data[0]
            seggt_onehot = warmup_du_info['model'].module.mirror_process(seggt)
            rec_loss = warmup_du_info['model'](seggt_onehot)
            if self.fp16_train:
                with amp.scale_loss(rec_loss, warmup_du_info['optimizer']) as scaled_rec_loss:
                    scaled_rec_loss.backward()
            else:
                rec_loss.backward()
            warmup_du_info['optimizer'].step()
            if self.distributed:
                warmup_du_info['du_loss'].append(reduce_loss_for_dist(rec_loss).item())
            else:
                warmup_du_info['du_loss'].append(rec_loss.item())
        self._print_train_progressbar()

    @master_only
    def _print_train_progressbar(self, ):
        rank, world_size = get_dist_info()
        if self._iter == 1:
            self.prog_bar = mmcv.ProgressBar(self.task_length * world_size)
            [self.prog_bar.update() for i in range(world_size)]
        elif self._iter == self.task_length:
            [self.prog_bar.update() for i in range(world_size)]
            self.prog_bar.file.write('\n')
        else:
            [self.prog_bar.update() for i in range(world_size)]

    def _print_warmup_infos(self, runner, cur_iter):
        log_str = f'{self.log_head} [{cur_iter}/{self._max_runs}]\t'
        for warmup_du_info in self.warmup_du_infos:
            log_str += 'Du_loss: {} '.format(np.mean(warmup_du_info['du_loss']))
            warmup_du_info['du_loss'] = []
        runner.logger.info(log_str)

    def _run_epochs(self, runner):
        data_loader = self.data_loader

        while self._epoch <= self._max_runs:
            for i, data_batch in enumerate(data_loader):
                self._iter = i+1
                self._run_one_batch(data_batch)

            self._print_warmup_infos(runner, self._epoch)
            self._epoch = self._epoch + 1

    def _run_iters(self, runner):
        iter_loader = IterLoader(self.data_loader)

        while self._iter <= self._max_runs:
            data_batch = next(iter_loader)
            # w_before1 = self.dupsampleblock_list[0].conv_w.weight.clone()
            # w_before2 = self.model.module.decode_head.dupsample.conv_w.weight.clone()
            self._run_one_batch(data_batch)
            # w_after1 = self.dupsampleblock_list[0].conv_w.weight.clone()
            # w_after2 = self.model.module.decode_head.dupsample.conv_w.weight.clone()
            # diff1 = torch.sum(w_after1 - w_before1) # verfy conv_w has been optimized
            # diff2 = torch.sum(w_after2 - w_before2)  # verfy conv_w has been optimized

            self._print_warmup_infos(runner, self._iter)
            self._iter = self._iter + 1

    def _destroy_resources(self):
        del self.warmup_du_infos
        del self.data_loader
        torch.cuda.empty_cache()

    def before_run(self, runner):
        if len(self.dupsampleblock_list) * self._max_runs == 0:
            return

        for warmup_du_info in self.warmup_du_infos:
            warmup_du_info['model'].train()
        if self.by_epoch:
            self._run_epochs(runner)
        else:
            self._run_iters(runner)

        self._destroy_resources()
        for i, hook in enumerate(runner._hooks):
            if isinstance(hook, WarmUpDUpsampleHook):
                runner._hooks.remove(hook)
                # del runner._hooks[i]
        runner.logger.info('Ending Warmup Dupsample Block !!!')

class WarmUpDUpsampleHook(Hook):
    """WarmUpDUpsampleHook hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, model, cfg, distributed=False, runstate=np.array([1])):
        self._max_runs = cfg.warmup_du_cfg.total_runs
        self.dupsampleblock_list = self._find_dupsampleblocks(model)
        if len(self.dupsampleblock_list) * self._max_runs == 0:
            return
        self.fp16_train = cfg.optimizer_config.get('type') == 'Fp16OptimizerHook'
        self.grad_clip = cfg.optimizer_config.get('grad_clip', None)

        # self.model = model
        self.runstate = runstate
        self.warmup_du_infos = []
        self.distributed = distributed

        self.interval = cfg.warmup_du_cfg.interval
        self.by_epoch = cfg.warmup_du_cfg.by_epoch
        self.log_head = 'Epoch' if self.by_epoch else 'Iter'
        self._iter = 1
        self._epoch = 1

        # build dataset and prepare data loaders
        dataset = build_dataset(cfg.data.train)
        self.data_loader = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True,
            pin_memory=False)
        self.task_length = len(self.data_loader) if self.by_epoch else self._max_runs

        for dupsampleblock in self.dupsampleblock_list:
            warmup_du_model = MirrorDUpsamplingBlock(dupsampleblock).cuda(cfg.gpu_ids[0])
            optimizer = build_optimizer(warmup_du_model, cfg.warmup_du_cfg.optimizer)
            warmup_du_model = parallel_model(warmup_du_model, cfg.gpu_ids, distributed)
            warmup_du_info = {'model': warmup_du_model,
                              'optimizer': optimizer,
                              'du_loss': [],
                              'loss': 0}
            self.warmup_du_infos.append(warmup_du_info)
        if self.fp16_train:
            self._init_fp16(cfg.optimizer_config)


    def _find_dupsampleblocks(self, model):
        dupsampleblock_list = []
        for m in model.modules():
            if isinstance(m, DUpsamplingBlock):
                dupsampleblock_list.append(m)

        return dupsampleblock_list

    def _run_one_batch(self, data_batch):
        # check runstate
        if self.runstate[0] == 0:
            sys.exit(0)
        # forward for loss
        for i, warmup_du_info in enumerate(self.warmup_du_infos):
            warmup_du_info['optimizer'].zero_grad()
            seggt = data_batch['gt_semantic_seg'].data[0]
            seggt_onehot = warmup_du_info['model'].module.mirror_process(seggt)
            warmup_du_info['loss'] = warmup_du_info['model'](seggt_onehot)
            # loss collect
            if self.distributed:
                warmup_du_info['du_loss'].append(reduce_loss_for_dist(warmup_du_info['loss']).item())
            else:
                warmup_du_info['du_loss'].append(warmup_du_info['loss'].item())
        # loss backward
        if self.fp16_train:
            self.fp16_optimize_one_step()
        else:
            self.optimize_one_step()
        # warmup bar
        self._print_train_progressbar()

    @master_only
    def _print_train_progressbar(self, ):
        rank, world_size = get_dist_info()
        if self._iter == 1:
            self.prog_bar = mmcv.ProgressBar(self.task_length * world_size)
            [self.prog_bar.update() for i in range(world_size)]
        elif self._iter == self.task_length:
            [self.prog_bar.update() for i in range(world_size)]
            self.prog_bar.file.write('\n')
        else:
            [self.prog_bar.update() for i in range(world_size)]

    def _print_warmup_infos(self, runner, cur_iter):
        log_str = f'{self.log_head} [{cur_iter}/{self._max_runs}]\t'
        for warmup_du_info in self.warmup_du_infos:
            log_str += 'Du_loss: {} '.format(np.mean(warmup_du_info['du_loss']))
            warmup_du_info['du_loss'] = []
            if warmup_du_info.get('overflow'):
                runner.logger.warning(
                    'Check overflow, downscale loss scale to {}'.format(warmup_du_info['overflow']))
        runner.logger.info(log_str + '\n')
        if cur_iter == self._max_runs:
            runner.logger.info('Ending Warmup Dupsample Block !!!\n')


    def _run_epochs(self, runner):
        data_loader = self.data_loader

        while self._epoch <= self._max_runs:
            for i, data_batch in enumerate(data_loader):
                self._iter = i+1
                self._run_one_batch(data_batch)

            self._print_warmup_infos(runner, self._epoch)
            self._epoch = self._epoch + 1

    def _run_iters(self, runner):
        iter_loader = IterLoader(self.data_loader)

        while self._iter <= self._max_runs:
            data_batch = next(iter_loader)
            # w_before1 = self.dupsampleblock_list[0].conv_w.weight.clone()
            # w_before2 = self.model.module.decode_head.dupsample.conv_w.weight.clone()
            self._run_one_batch(data_batch)
            # w_after1 = self.dupsampleblock_list[0].conv_w.weight.clone()
            # w_after2 = self.model.module.decode_head.dupsample.conv_w.weight.clone()
            # diff1 = torch.sum(w_after1 - w_before1) # verfy conv_w has been optimized
            # diff2 = torch.sum(w_after2 - w_before2)  # verfy conv_w has been optimized

            self._print_warmup_infos(runner, self._iter)
            self._iter = self._iter + 1

    def _destroy_resources(self):
        del self.warmup_du_infos
        del self.data_loader
        torch.cuda.empty_cache()

    def before_run(self, runner):
        if len(self.dupsampleblock_list) * self._max_runs == 0:
            return

        for warmup_du_info in self.warmup_du_infos:
            warmup_du_info['model'].train()
        if self.by_epoch:
            self._run_epochs(runner)
        else:
            self._run_iters(runner)

        self._destroy_resources()
        for i, hook in enumerate(runner._hooks):
            if isinstance(hook, WarmUpDUpsampleHook):
                runner._hooks.remove(hook)
                # del runner._hooks[i]


    def optimize_one_step(self):
        for warmup_du_info in self.warmup_du_infos:
            warmup_du_info['optimizer'].zero_grad()
            warmup_du_info['loss'].backward()
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(warmup_du_info['model'].parameters())
            warmup_du_info['optimizer'].step()

    ## fp16 train
    def _init_fp16(self, optimizer_config):
        """Preparing steps before Mixed Precision Training.

        1. Make a master copy of fp32 weights for optimization.
        2. Convert the main model from fp32 to fp16.
        """
        # init cfg for fp16_optimizer
        self.coalesce = optimizer_config.get('coalesce', True)
        self.bucket_size_mb = optimizer_config.get('bucket_size_mb', -1)
        loss_scale = optimizer_config.get('loss_scale', 512.)
        if loss_scale == 'dynamic':
            self.loss_scaler = LossScaler(mode='dynamic')
        elif isinstance(loss_scale, float):
            self.loss_scaler = LossScaler(init_scale=loss_scale, mode='static')
        else:
            raise ValueError('loss_scale must be of type float or str')
        # keep a copy of fp32 weights
        for warmup_du_info in self.warmup_du_infos:
            old_groups = warmup_du_info['optimizer'].param_groups
            warmup_du_info['optimizer'].param_groups = copy.deepcopy(
                warmup_du_info['optimizer'].param_groups)
            state = defaultdict(dict)
            p_map = {
                old_p: p
                for old_p, p in zip(
                chain(*(g['params'] for g in old_groups)),
                chain(*(g['params'] for g in warmup_du_info['optimizer'].param_groups)))
            }
            for k, v in warmup_du_info['optimizer'].state.items():
                state[p_map[k]] = v
            warmup_du_info['optimizer'].state = state
            # convert model to fp16
            wrap_fp16_model(warmup_du_info['model'])

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def fp16_optimize_one_step(self):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer `loss_scalar.py`

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        """
        # clear grads of last iteration
        for warmup_du_info in self.warmup_du_infos:
            warmup_du_info['model'].zero_grad()
            warmup_du_info['optimizer'].zero_grad()
            # scale the loss value
            scaled_loss = warmup_du_info['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()
            # copy fp16 grads in the model to fp32 params in the optimizer
            fp32_weights = []
            for param_group in warmup_du_info['optimizer'].param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(warmup_du_info['model'], fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow:
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    self.clip_grads(fp32_weights)
                # update fp32 params
                warmup_du_info['optimizer'].step()
                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(warmup_du_info['model'], fp32_weights)
            self.loss_scaler.update_scale(has_overflow)
            if has_overflow:
                warmup_du_info['overflow'] = self.loss_scaler.cur_scale