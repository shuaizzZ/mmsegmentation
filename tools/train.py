
import os
import argparse
import copy
import time
import os.path as osp

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor, trainer_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--config',
        # default='../configs/pspnet/pspnet_r50-d8_512x512_80k_ade20k.py',
        # default='../configs/pspnet/dupsp_r18_yantai_st12.py',
        default='../configs/pspnet/anjie.py',
        # default='../configs/pspnet/dupsp_r18_ainno.py',
        help='train config file path')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--load-from',
        help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from',
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        default=False,
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        default=1,
        type=int,
        help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        default=[0],
        type=int,
        nargs='+',
        help='ids of gpus to use (only applicable to non-distributed training)')

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu_ids)

    return args


def replace_BN_cfg(cfg):
    # replace the type of norm_cfg with BN
    cfg.norm_cfg.type = 'BN'
    for key, val in cfg.model.items():
        if 'norm_cfg' in cfg.model[key]:
            cfg.model[key]['norm_cfg'] = cfg.norm_cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    else:
        cfg.work_dir = osp.join(cfg.work_dir, cfg.get('task_name', 'unnamed_task'))

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids if isinstance(args.gpu_ids, list) else [args.gpu_ids]
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    # if not distributed, replace SyncBN with BN
    if len(cfg.gpu_ids) > 1:
        assert args.launcher in ['pytorch', 'slurm', 'mpi']
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    else:
        distributed = False
        replace_BN_cfg(cfg)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.trainer_csv_path = osp.join(cfg.work_dir, 'train_log.csv')
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    task_name = cfg.task_name if cfg.get('task_name') else timestamp
    log_file = osp.join(cfg.work_dir, f'{task_name}.log')

    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    # set random seeds
    cfg.seed = args.seed if args.seed is not None else cfg.get('seed')
    cfg.deterministic = args.deterministic if args.deterministic is not None else cfg.get('deterministic')
    if cfg.seed is not None:
        logger.info(f'Set random seed to {cfg.seed}, deterministic: '
                    f'{cfg.deterministic}')
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    meta['seed'] = cfg.seed
    meta['exp_name'] = osp.basename(args.config) + ' | ' + task_name
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    model = build_segmentor(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.val.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg, # cfg.pretty_text (ORG)
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # normal or manuvision train_segmentor
    if cfg.get('segmentor_type') == 'manuvision':
        segmentor = trainer_segmentor
    else:
        segmentor = train_segmentor
    segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()