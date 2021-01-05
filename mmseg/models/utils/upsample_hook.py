
import sys
import numpy as np
from apex import amp
from tqdm import tqdm
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from mmcv.runner import Hook, build_optimizer
from mmcv.runner.iter_based_runner import IterLoader
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmseg.models.utils import DUpsamplingBlock


# 以后放工具类中
def reduce_loss_for_dist(loss_value):
    loss_value = loss_value.data.clone()
    dist.all_reduce(loss_value.div_(dist.get_world_size()))
    return loss_value


class UpsampleHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, model, cfg, distributed=False, runstate=np.array([1])):
        from mmseg.datasets import build_dataloader, build_dataset
        self.upsampleblock_list = self._find_upsampleblocks(model)
        self.upsampleblock_infos = []
        self.distributed = distributed
        # TODO only support one gpu run
        self.device_ids = cfg.gpu_ids[0]
        self.runstate = runstate

        if 0 == len(self.upsampleblock_list):
            return

        self.interval = cfg.du_config.interval
        self.by_epoch = cfg.du_config.by_epoch
        self._max_runs = cfg.du_config.total_runs
        self._iter = 0
        self._epoch = 0

        # build dataset
        datasets = [build_dataset(cfg.data.train)]

        # prepare data loaders
        datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        num_gpus = len(self.device_ids) if isinstance(self.device_ids, list) else 1
        self.data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                num_gpus,
                dist=distributed,
                seed=cfg.seed,
                drop_last=True) for ds in datasets
        ]
        # just support only one
        self.data_loader = self.data_loaders[0]

        for upsampleblock in self.upsampleblock_list:
            conv_w = upsampleblock.conv_w
            conv_p = upsampleblock.get_conv_p()
            mirror_process = upsampleblock.mirror_process
            mirror_du_module = nn.Sequential(conv_p, conv_w)
            criterion = nn.MSELoss()
            optimizer = build_optimizer(mirror_du_module, cfg.du_config.optimizer)
            model = mirror_du_module
            self.mix_prec = False

            # put model on gpus
            if distributed:
                find_unused_parameters = cfg.get('find_unused_parameters', False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                model = MMDataParallel(
                    model.cuda(self.device_ids), device_ids=cfg.gpu_ids)

            upsampleblock_info = {'model': model, 'preprocess': mirror_process,
                                  'criterion': criterion, 'optimizer': optimizer, 'du_loss': 0.0,
                                  'upsampleblock': upsampleblock}
            self.upsampleblock_infos.append(upsampleblock_info)

    def _find_upsampleblocks(self, model):
        upsampleblock_list = []
        for m in model.modules():
            if isinstance(m, DUpsamplingBlock):
                upsampleblock_list.append(m)

        return upsampleblock_list

    def _destroy_resources(self):
        del self.data_loader
        for upsampleblock_info in self.upsampleblock_infos:
            del upsampleblock_info['model']
            del upsampleblock_info['criterion']
            del upsampleblock_info['optimizer']
        torch.cuda.empty_cache()

    def _run_one_batch(self, data_batch):
        # check runstate
        if self.runstate[0] == 0:
            sys.exit(0)

        for upsampleblock_info in self.upsampleblock_infos:
            upsampleblock_info['optimizer'].zero_grad()
            target = data_batch['gt_semantic_seg'].data[0]
            # target = target.cuda(self.device_ids)
            seggt_onehot = upsampleblock_info['preprocess'](target)
            seggt_onehot_reconstructed = upsampleblock_info['model'](seggt_onehot)
            seggt_onehot = seggt_onehot.cuda(seggt_onehot_reconstructed.device.index)
            rec_loss = upsampleblock_info['criterion'](seggt_onehot, seggt_onehot_reconstructed)

            if self.mix_prec:
                with amp.scale_loss(rec_loss, upsampleblock_info['optimizer']) as scaled_rec_loss:
                    scaled_rec_loss.backward()
            else:
                rec_loss.backward()
            upsampleblock_info['optimizer'].step()
            if self.distributed:
                upsampleblock_info['du_loss'] += reduce_loss_for_dist(rec_loss).item()
            else:
                upsampleblock_info['du_loss'] += rec_loss.item()

    def _run_epochs(self, runner):
        data_loader = self.data_loader

        while self._epoch < self._max_runs:
            tbar = tqdm(data_loader, unit='batch', desc='==>DU_WarmUp', ncols=80)
            for i, data_batch in enumerate(tbar):
                self._run_one_batch(data_batch)

            log_str = f'Epoch [{self._epoch + 1}/{self._max_runs}]\t'
            for upsampleblock_info in self.upsampleblock_infos:
                log_str += 'Du_loss: {} '.format(upsampleblock_info['du_loss'] / (i + 1))
                upsampleblock_info['du_loss'] = 0
            runner.logger.info(log_str)

            self._epoch = self._epoch + 1

    def _run_iters(self, runner):
        iter_loader = IterLoader(self.data_loader)

        while self._iter < self._max_runs:
            data_batch = next(iter_loader)
            self._run_one_batch(data_batch)

            if (self._iter + 1) % self.interval == 0:
                log_str = f'Iter [{self._iter + 1}/{self._max_runs}]\t'
                for upsampleblock_info in self.upsampleblock_infos:
                    log_str += 'Du_loss: {} '.format(upsampleblock_info['du_loss']/self.interval)
                    upsampleblock_info['du_loss'] = 0
                runner.logger.info(log_str)

            self._iter = self._iter + 1

    def before_run(self, runner):
        if not len(self.upsampleblock_list):
            return

        for upsampleblock_info in self.upsampleblock_infos:
            upsampleblock_info['model'].train()

        if self.by_epoch:
            self._run_epochs(runner)
        else:
            self._run_iters(runner)

        self._destroy_resources()