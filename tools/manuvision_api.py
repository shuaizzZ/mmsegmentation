import traceback

try:
    import os
    import sys
    import time
    import copy
    import shutil
    import os.path as osp

    import torch
    # from mmseg.mv_yaml.mv_train_default import _C as mvcfg
    # from mmseg.mv_yaml.mv_test_default import _C as mvcfg_test

    import mmcv
    from mmcv.runner import init_dist
    from mmcv.utils import Config, DictAction, get_git_hash

    from mmseg import __version__
    from mmseg.apis import set_random_seed, train_segmentor, train
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmseg.utils import collect_env, get_root_logger
except Exception as ex:
    ex_type, ex_val, ex_stack = sys.exc_info()
    print('ex_type:',ex_type)
    print('ex_val:',ex_val)
    for stack in traceback.extract_tb(ex_stack):
        print(stack)


def merge_to_mmcfg_from_mvcfg(mmcfg, mvcfg):
    def modify_if_exist(mmpara, mmfields, mvpara, mvfields):
        for i in range(len(mvfields)):
            mmfield = mmfields[i]
            mvfield = mvfields[i]
            if mvpara.get(mvfield, None):
                mmpara[mmfield] = mvpara.get(mvfield)
    ## model

    ## dataset
    mmcfg.dataset_type = mvcfg.DATASETS.TYPE
    mmcfg.data_root = mvcfg.DATASETS.ROOT
    mmcfg.dataset = mvcfg.DATASETS.DATASET
    mmcfg.classes = mvcfg.DATASETS.CLASSES
    for mode in ['train', 'val', 'test']:
        modify_if_exist(mmcfg._cfg_dict['data'][mode], ['type'],
                        mmcfg._cfg_dict, ['dataset_type'])
        for para in ['data_root', 'dataset', 'classes']:
            modify_if_exist(mmcfg._cfg_dict['data'][mode], [para],
                            mmcfg._cfg_dict, [para])

    # pipeline train
    mmcfg.labels = mvcfg.DATASETS.LABELS
    mmcfg.crop_size = mvcfg.DATASETS.AUGMENT.CROP_SIZE
    option_para = {'Relabel': ['labels'],
                   'RandomCrop': ['crop_size'],}
    for i, trans_dict in enumerate(mmcfg.train_pipeline):
        trans_type = trans_dict.type
        if trans_type in option_para.keys():
            modify_if_exist(mmcfg._cfg_dict['train_pipeline'][i],
                            option_para[trans_type],
                            mmcfg._cfg_dict,
                            option_para[trans_type])
    mmcfg.data.train.pipeline = mmcfg.train_pipeline
    # pipeline val
    option_para = {'Relabel': ['labels'],
                   'Pad': ['size']}
    for i, trans_dict in enumerate(mmcfg.val_pipeline):
        trans_type = trans_dict.type
        if trans_type in option_para.keys():
            modify_if_exist(mmcfg._cfg_dict['val_pipeline'][i],
                            option_para[trans_type],
                            mmcfg._cfg_dict,
                            option_para[trans_type])
    mmcfg.data.val.pipeline = mmcfg.val_pipeline

    mmcfg.data.samples_per_gpu = mvcfg.TRAIN.BATCH_SIZE
    mmcfg.data.workers_per_gpu = 0

    ## schedule
    # mmcfg.optimizer.type = mvcfg.SOLVER.OPT.OPTIMIZER
    # mmcfg.optimizer.momentum = mvcfg.SOLVER.OPT.MOMENTUM
    # mmcfg.optimizer.weight_decay = mvcfg.SOLVER.OPT.WEIGHT_DECAY
    ## runtime
    if mvcfg.TRAIN.FT.RESUME:
        mmcfg.load_from = mvcfg.TRAIN.FT.CHECKPATH
    mmcfg.work_dir = mvcfg.TRAIN.CHECKNAME

    return mmcfg


class manuvision():
    def init(self):
        pass

    def train(self, runstate):
        try:
            self.train_py(runstate)
        except Exception as ex:
            ex_type, ex_val, ex_stack = sys.exc_info()
            print('ex_type:',ex_type)
            print('ex_val:',ex_val)
            for stack in traceback.extract_tb(ex_stack):
                print(stack)

    def train_py(self, runstate):
        # manuvision config
        mv_config_file = "../../configs/manuvision_train.yaml"
        mv_config_path = os.path.join(os.path.split(__file__)[0], mv_config_file)
        if not os.path.exists(mv_config_path):
            mv_config_file = "manuvision_train.yaml"
            mv_config_path = os.path.join(os.path.split(__file__)[0], mv_config_file)
        mvcfg = Config.fromfile(mv_config_path)
        # mmseg config
        # mm_config_path = '../work_dir/2020_1029/pspnet_r50-d8_512x512_80k_ade20k.py'
        mm_config_path = '../configs/pspnet/pspnet_r50-d8_yantai_st12.py'
        mmcfg = Config.fromfile(mm_config_path)
        cfg = merge_to_mmcfg_from_mvcfg(mmcfg, mvcfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(mm_config_path))[0])

        # init distributed env first, since logger depends on the dist info.
        if cfg.get('launcher', 'none') == 'none' or len(cfg.gpu_ids) == 1:
            distributed = False
        else:
            distributed = True
            init_dist(cfg.launcher, **cfg.dist_params)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(mm_config_path)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        cfg.seed = cfg.get('seed', None)
        if cfg.seed is not None:
            logger.info(f'Set random seed to {cfg.seed}, deterministic: '
                        f'{cfg.deterministic}')
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)

        meta['seed'] = cfg.seed
        meta['exp_name'] = osp.basename(mm_config_path)

        # validate
        cfg.validate = cfg.get('validate', True)

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
                config=cfg.pretty_text,
                CLASSES=datasets[0].CLASSES,
                PALETTE=datasets[0].PALETTE)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=cfg.validate,
            timestamp=timestamp,
            meta=meta,
            runstate=runstate)


    def inference(self, run_state):
        args = parse_args()

        assert args.out or args.eval or args.format_only or args.show \
               or args.show_dir, \
            ('Please specify at least one operation (save/eval/format/show the '
             'results / save the results) with the argument "--out", "--eval"'
             ', "--format-only", "--show" or "--show-dir"')

        if args.eval and args.format_only:
            raise ValueError('--eval and --format_only cannot be both specified')

        if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')

        cfg = mmcv.Config.fromfile(args.config)
        if args.options is not None:
            cfg.merge_from_dict(args.options)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if args.aug_test:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                dataset.evaluate(outputs, args.eval, **kwargs)

    def convert(self, run_state, mode=0):
        pass

    def uninit(self):
        print("python ainnovision uninit")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    runstate = np.array([1])

    mv = manuvision()
    mv.init()
    mv.train_py(runstate)
    # mv.inference(run_state)
    # mv.convert(run_state)