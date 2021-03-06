
import random
import warnings
import numpy as np
import os.path as osp

import torch
import mmcv
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.models.utils import parallel_model, WarmUpDUpsampleHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    runstate=np.array([1])):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    model = parallel_model(model, cfg.gpu_ids, distributed, find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    ## register WarmUpDUpsampleHook
    if not (cfg.resume_from and cfg.load_from) and cfg.get('dupsample'):
        runner.register_hook(WarmUpDUpsampleHook(model, cfg, distributed, runstate))

    runner.run(data_loaders, cfg.workflow)


def trainer_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    runstate=np.array([1])):
    """Launch segmentor training."""
    # from mmcv.runner import HOOKS
    from mmseg.utils import (CheckRunstateHook, TrainerLogHook,
                             TrainerCheckpointHook, StatisticTextLoggerHook)

    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    model = parallel_model(model, cfg.gpu_ids, distributed, find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    ## register hooks
    ## the priority of log_hook is VERY_LOW, and others is NORMAL
    checkpoint_config = cfg.checkpoint_config
    checkpoint_config.setdefault('type', 'TrainerCheckpointHook')
    checkpoint_config.setdefault('priority', 'LOW')
    cfg.optimizer_config = cfg.get('optimizer_config', dict())
    trainer_checkpoint_hook = runner.register_hook_from_cfg(checkpoint_config)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   trainer_checkpoint_hook, cfg.log_config,
                                   cfg.get('momentum_config', None))

    ## an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    ## register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)         
        eval_cfg = cfg.get('evaluation', {})
        ## ????????????      
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    ## register WarmUpDUpsampleHook
    if not (cfg.resume_from and cfg.load_from) and cfg.dupsample:
        runner.register_hook(WarmUpDUpsampleHook(model, cfg, distributed, runstate), priority='HIGH')
    ## register CheckRunstateHook and TrainerLogHook
    runner.register_hook(CheckRunstateHook(runstate), priority='HIGH')
    runner.register_hook(TrainerLogHook(cfg.trainer_csv_path, cfg.num_classes), priority='LOW')
    runner.logger.info('Start Training, Good Luck !!!')
    runner.run(data_loaders, cfg.workflow)