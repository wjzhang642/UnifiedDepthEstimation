import os
import sys
import glob
import tqdm
import argparse
import os.path as osp
import torch
import numpy as np
import matplotlib
from PIL import Image
from loguru import logger


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_metrics(gt, pred, mask):
    mask = np.logical_and(mask, gt > 0)
    mask = np.logical_and(mask, gt < np.inf)
    mask = np.logical_and(mask, pred > 0)
    mask = np.logical_and(mask, pred < np.inf)
    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) * np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    # avoid: np.sqrt encounter negative value
    diff = np.mean(err ** 2) - np.mean(err) ** 2
    diff = diff if diff >= 0 else 0
    silog = np.sqrt(diff) * 100
    # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', '--work_dir', type=str, default='.data/public_dataset/eval_fisheye_dataset')
    parser.add_argument('-d', '--dataset_name', type=str, required=True)
    parser.add_argument('-n', '--sample_num', type=int, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_dir = os.path.join(args.work_dir, '_'.join(['test_dataset', dataset_name, str(args.sample_num)]))

    compare_list = [
        'depth_anything_v1',
        'depth_anything_v2',
        '0.1unlabeled_0.5p',
    ]

    metrics_list = [RunningAverageDict() for _ in range(len(compare_list))]
    metrics_dict = {}

    if len(compare_list) == 1:
        save_dir = osp.join(dataset_dir, 'results', compare_list[0])
    else:
        save_dir = osp.join(dataset_dir, 'results', 'COMPARE_' + '@'.join(compare_list))
    os.makedirs(save_dir, exist_ok=True)
    log_path = osp.join(save_dir, 'result.txt')
    logger.add(log_path)
    logger.add(sys.stdout, format="{time}{level}{message}", level="INFO")

    c_map = matplotlib.colormaps.get_cmap('Spectral_r')

    rgb_path = osp.join(dataset_dir, 'rgb')
    depth_path = osp.join(dataset_dir, 'depth')
    image_list = glob.glob(osp.join(rgb_path, '*.png'))
    for image_path in tqdm.tqdm(image_list):
        rgb_filename = osp.basename(image_path)
        depth_filename = rgb_filename.replace('.png', '.npy')
        rgb = Image.open(image_path)
        rgb_npy = np.array(rgb)
        vis_combine = np.copy(rgb_npy)
        block_region = np.all(rgb_npy == [0, 0, 0], axis=-1)
        split_space = np.zeros(shape=(rgb_npy.shape[0], 6, 3), dtype=np.uint8)

        # gt
        gt_depth_path = osp.join(depth_path, 'gt', depth_filename)
        metric_gt_npy = np.load(gt_depth_path)

        valid_mask = metric_gt_npy > 0
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to('cuda:0')

        rel_gt_npy = np.zeros_like(metric_gt_npy)
        rel_gt_npy[valid_mask] = 1 / metric_gt_npy[valid_mask]
        max_valid_gt = rel_gt_npy[valid_mask].max()
        min_valid_gt = rel_gt_npy[valid_mask].min()
        rel_gt_tensor = torch.tensor(rel_gt_npy).unsqueeze(0).to('cuda:0')

        norm_rel_gt_npy = np.zeros_like(rel_gt_npy)
        norm_rel_gt_npy[valid_mask] = (rel_gt_npy[valid_mask] - min_valid_gt) / (max_valid_gt - min_valid_gt)

        gt_color_npy = np.zeros_like(rgb_npy)
        gt_color_npy[valid_mask] = (c_map(norm_rel_gt_npy[valid_mask])[:, :3] * 255).astype(np.uint8)
        vis_valid_mask = np.zeros((*valid_mask.shape, 3), dtype=np.uint8)  # 有效区域掩码的可视化
        vis_valid_mask[valid_mask] = [255, 255, 255]
        vis_combine = np.hstack([vis_combine, split_space, gt_color_npy, split_space, vis_valid_mask])

        for i, result in enumerate(compare_list):
            pred_depth_path = osp.join(depth_path, result, depth_filename)
            rel_pred_npy = np.load(pred_depth_path)
            max_valid_pred = np.max(rel_pred_npy)
            min_valid_pred = np.min(rel_pred_npy)

            norm_rel_pred_npy = (rel_pred_npy - min_valid_pred) / (max_valid_pred - min_valid_pred)

            pred_color_npy = np.zeros_like(rgb_npy)
            pred_color_npy[valid_mask] = (c_map(norm_rel_pred_npy[valid_mask])[:, :3] * 255).astype(np.uint8)
            vis_combine = np.hstack([vis_combine, split_space, pred_color_npy])

            norm_rel_pred_tensor = torch.tensor(norm_rel_pred_npy).unsqueeze(0).to('cuda:0')

            if not torch.any(norm_rel_pred_tensor[valid_mask_tensor]):
                continue

            scale, shift = compute_scale_and_shift(norm_rel_pred_tensor, rel_gt_tensor)
            if scale < 0:
                print('Warning! Found negative value of scale: {}, {}'.format(result, scale.item()))
            scaled_and_shifted_pred_tensor = scale.view(-1, 1, 1) * norm_rel_pred_tensor + shift.view(-1, 1, 1)
            scaled_and_shifted_metric_pred_npy = 1 / scaled_and_shifted_pred_tensor.squeeze().cpu().numpy()

            metrics = compute_metrics(metric_gt_npy, scaled_and_shifted_metric_pred_npy, mask=valid_mask)
            metrics_list[i].update(metrics)

            if any([pm in rgb_filename for pm in ['equidistant', 'equisolid', 'orthogonal', 'stereographic']]):
                splits = rgb_filename.split('_')
                pm_name = splits[1]
                focal_length = splits[2]
                metric_key = (compare_list[i], pm_name, focal_length)
                pick_metric = metrics_dict.get(metric_key, None)
                if pick_metric is None:
                    pick_metric = RunningAverageDict()
                    pick_metric.update(metrics)
                    metrics_dict[metric_key] = pick_metric
                else:
                    pick_metric.update(metrics)

            if metrics['abs_rel'] > 10000:
                logger.warning('Warning! abs_rel: {}'.format(metrics['abs_rel']))
                logger.warning(image_path)
                compute_metrics(metric_gt_npy, scaled_and_shifted_metric_pred_npy, mask=valid_mask)

            # this will slow the progress
            # Image.fromarray(vis_combine).save(osp.join(save_dir, rgb_filename))

    logger.info('====================Metric Summary====================')
    for i, metrics in metrics_list:
        logger.info(compare_list[i])
        logger.info({k: round(v, 4) for k, v in metrics.get_value().items() if k in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'sq_rel']})

    logger.info('====================Metric Detail====================')
    for metric_key, metrics in metrics_dict:
        logger.info(metric_key)
        logger.info({k: round(v, 4) for k, v in metrics.get_value().items() if k in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'sq_rel']})
