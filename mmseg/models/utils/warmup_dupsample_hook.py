
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
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
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
        self.mix_prec = False
        self.model = model
        self.dupsampleblock_list = self._find_dupsampleblocks(model)
        self.warmup_du_infos = []
        self.distributed = distributed
        self.runstate = runstate

        if 0 == len(self.dupsampleblock_list):
            return

        self.interval = cfg.du_config.interval
        self.by_epoch = cfg.du_config.by_epoch
        self._max_runs = cfg.du_config.total_runs
        self._iter = 0
        self._epoch = 0

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

        for dupsampleblock in self.dupsampleblock_list:
            warmup_du_model = MirrorDUpsamplingBlock(dupsampleblock)
            optimizer = build_optimizer(warmup_du_model, cfg.du_config.optimizer)

            # put model on gpus
            warmup_du_model = parallel_model(warmup_du_model, cfg.gpu_ids, distributed)
            # if distributed:
            #     find_unused_parameters = cfg.get('find_unused_parameters', False)
            #     warmup_du_model = MMDistributedDataParallel(
            #         warmup_du_model.cuda(),
            #         device_ids=[torch.cuda.current_device()],
            #         broadcast_buffers=False,
            #         find_unused_parameters=find_unused_parameters)
            # else:
            #     warmup_du_model = MMDataParallel(
            #         warmup_du_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

            warmup_du_info = {'model': warmup_du_model, 'optimizer': optimizer, 'du_loss': 0.0}
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
                warmup_du_info['du_loss'] += reduce_loss_for_dist(rec_loss).item()
            else:
                warmup_du_info['du_loss'] += rec_loss.item()

    def _run_epochs(self, runner):
        data_loader = self.data_loader

        while self._epoch < self._max_runs:
            tbar = tqdm(data_loader, unit='batch', desc='==>DU_WarmUp', ncols=80)
            for i, data_batch in enumerate(tbar):
                self._run_one_batch(data_batch)

            log_str = f'Epoch [{self._epoch + 1}/{self._max_runs}]\t'
            for warmup_du_info in self.warmup_du_infos:
                log_str += 'Du_loss: {} '.format(warmup_du_info['du_loss'] / (i + 1))
                warmup_du_info['du_loss'] = 0
            runner.logger.info(log_str)

            self._epoch = self._epoch + 1

    def _run_iters(self, runner):
        iter_loader = IterLoader(self.data_loader)

        while self._iter < self._max_runs:
            data_batch = next(iter_loader)
            # w_before1 = self.dupsampleblock_list[0].conv_w.weight.clone()
            # w_before2 = self.model.module.decode_head.dupsample.conv_w.weight.clone()
            self._run_one_batch(data_batch)
            # w_after1 = self.dupsampleblock_list[0].conv_w.weight.clone()
            # w_after2 = self.model.module.decode_head.dupsample.conv_w.weight.clone()
            # diff1 = torch.sum(w_after1 - w_before1) # verfy conv_w has been optimized
            # diff2 = torch.sum(w_after2 - w_before2)  # verfy conv_w has been optimized

            if (self._iter + 1) % self.interval == 0:
                log_str = f'Iter [{self._iter + 1}/{self._max_runs}]\t'
                for warmup_du_info in self.warmup_du_infos:
                    log_str += 'Du_loss: {} '.format(warmup_du_info['du_loss']/self.interval)
                    warmup_du_info['du_loss'] = 0
                runner.logger.info(log_str)

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

        self._destroy_resources()