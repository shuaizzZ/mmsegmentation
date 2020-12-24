
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


# 以后放工具类中
def reduce_loss_for_dist(loss_value):
    loss_value = loss_value.data.clone()
    dist.all_reduce(loss_value.div_(dist.get_world_size()))
    return loss_value


class WarmUpDUpsampleHook(Hook):
    """WarmUpDUpsampleHook hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, model, cfg, distributed=False, runstate=np.array([1])):
        self.mix_prec = 1
        self.model = model
        self.dupsampleblock_list = self._find_dupsampleblocks(model)
        self.warmup_du_infos = []
        self.distributed = distributed
        self.runstate = runstate
        if 0 == len(self.dupsampleblock_list):
            return

        self.interval = cfg.warmup_du_cfg.interval
        self.by_epoch = cfg.warmup_du_cfg.by_epoch
        self._max_runs = cfg.warmup_du_cfg.total_runs
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
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True)
        self.task_length = len(self.data_loader) if self.by_epoch else self._max_runs

        for dupsampleblock in self.dupsampleblock_list:
            warmup_du_model = MirrorDUpsamplingBlock(dupsampleblock).cuda(cfg.gpu_ids[0])
            optimizer = build_optimizer(warmup_du_model, cfg.warmup_du_cfg.optimizer)
            # put model on gpus
            if self.mix_prec:
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
            target = data_batch['gt_semantic_seg'].data[0]
            rec_loss = warmup_du_info['model'](target)
            if self.mix_prec:
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
        torch.cuda.empty_cache()

    def before_run(self, runner):
        if not len(self.dupsampleblock_list):
            return

        for warmup_du_info in self.warmup_du_infos:
            warmup_du_info['model'].train()

        if self.by_epoch:
            self._run_epochs(runner)
        else:
            self._run_iters(runner)
        runner.logger.info('Ending Warmup Dupsample Block !!!')
        self._destroy_resources()