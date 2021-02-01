
import os
import sys
import pickle
import shutil
import tempfile
import numpy as np
import os.path as osp
from yutils.csv.csv import CSV

import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs, imwrite
from mmcv.runner import get_dist_info


def single_gpu_test(model, data_loader, rescale=True, show=False, out_dir=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    seg_targets = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        if 'gt_semantic_seg' in data:
            target = data.pop('gt_semantic_seg')
            for gt in target:
                gt = gt.cpu().numpy()[0] # 1*h*w ==> h*w
                seg_targets.append(gt)
        with torch.no_grad():
            result = model(return_loss=False, rescale=rescale, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    if seg_targets:
        return [results, seg_targets]
    return results


def mv_single_gpu_test(model, data_loader, runstate, draw_contours=False, draw_target=True, out_dir=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.

    Returns:
        list: The prediction results.
    """

    log_path = osp.join(out_dir, 'test_log.csv')
    if osp.isfile(log_path):
        os.remove(log_path)
    test_log = CSV(log_path)
    log_head = ['Image_ID']
    test_log.append(log_head)

    out_pt = osp.join(out_dir, 'test_predict')
    mmcv.mkdir_or_exist(out_pt)
    if draw_contours:
        out_cnt = osp.join(out_dir, 'test_drawContours')
        mmcv.mkdir_or_exist(out_cnt)
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    draw_target_flag = False
    for img_id, data in enumerate(data_loader):
        if runstate[0] == 0:
            sys.exit(0)
        if 'gt_semantic_seg' in data and draw_target:
            draw_target_flag = True
            target = data.pop('gt_semantic_seg')[0]
            target = target.cpu().numpy()[0]  # 1*h*w ==> h*w

        with torch.no_grad():
            result = model(return_loss=False, return_logit=True, **data)
        img_metas = data['img_metas'][0].data[0]
        img_path = img_metas[0]['filename']
        img_name = osp.basename(img_path)

        ## output pt map
        base_name = img_name.split('.')[0]
        for chn in range(1, result.size(1)):
            probability = np.uint8(result[0, chn, :, :].cpu() * 255)
            out_path = osp.join(out_pt, '{}_{}.png'.format(base_name, chn))
            # imwrite(probability, out_path)
            cv2.imwrite(out_path, probability)
        ## output image with draw_contours
        if draw_contours:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            line = max(int(np.sqrt(h*w) // 512), 1)
            predict = torch.max(result, 1)[1].cpu().numpy()
            predict = np.uint8(np.squeeze(predict))
            contours, _ = cv2.findContours(predict, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 0, 255), line)
            if draw_target_flag:
                contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), line)
            cv2.imwrite(osp.join(out_cnt, img_name), image)

        test_log.append(img_id)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()


def multi_gpu_test(model, data_loader, rescale=True, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    seg_targets = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if 'gt_semantic_seg' in data:
            target = data.pop('gt_semantic_seg')
            for gt in target:
                gt = gt.cpu().numpy()[0] # 1*h*w ==> h*w
                seg_targets.append(gt)
        with torch.no_grad():
            result = model(return_loss=False, rescale=rescale, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
        seg_targets = collect_results_cpu(seg_targets, len(dataset), tmpdir+'_target')
    if seg_targets:
        return [results, seg_targets]
    return results


def collect_results_cpu(result_part, size, tmpdir=None, filename_tmpl='part_{}.pkl'):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, filename_tmpl.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
