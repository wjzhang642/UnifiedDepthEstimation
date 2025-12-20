import os
import sys
import glob
import json
import os.path as osp
import numpy as np
import tqdm
import argparse
from PIL import Image
from scipy import sparse

sys.path.insert(0, osp.abspath(osp.join(os.getcwd(), '../..')))
os.chdir(osp.abspath(osp.join(os.getcwd(), '../..')))
from dataset.fisheye_aug import pinhole2fisheye

datasets_list = {
    'KITTI': {
        'data_root': './data',
        'txt_path': 'KITTI/train_public_KITTI.txt',
        'is_fisheye': False,
        'is_metric_gt': True,
    },
    'NYUv2': {
        'data_root': './data',
        'txt_path': 'NYUv2/train_public_NYUv2_val.txt',
        'is_fisheye': False,
        'is_metric_gt': True,
    },
    'Sintel': {
        'data_root': './data',
        'txt_path': 'Sintel/train_public_Sintel.txt',
        'is_fisheye': False,
        'is_metric_gt': True,
    },
    'DDAD': {
        'data_root': './data',
        'txt_path': 'DDAD/train_public_DDAD_val.txt',
        'is_fisheye': False,
        'is_metric_gt': True,
    },
    'ETH3D': {
        'data_root': './data',
        'txt_path': 'ETH3D/train_public_ETH3D.txt',
        'is_fisheye': False,
        'is_metric_gt': True,
    },
    'DIODE': {
        'data_root': './data',
        'txt_path': 'DIODE/train_public_DIODE_val.txt',
        'is_fisheye': False,
        'is_metric_gt': True,
    },
    'kitti360': {
        'data_root': './data',
        'txt_path': 'kitti360/data_2d_raw/train_public_kitti360_fisheye_0009.txt',
        'is_fisheye': True,
        'is_metric_gt': True,
    },
    'SynWoodScape': {
        'data_root': './data',
        'txt_path': 'SynWoodScape/train_public_SynWoodScape_val.txt',
        'is_fisheye': True,
        'is_metric_gt': True,
    },
    'MultiFOV': {
        'data_root': './data',
        'txt_path': 'MultiFOV/train_public_MultiFOV.txt',
        'is_fisheye': True,
        'is_metric_gt': True,
    },
    'Matterport360': {
        'data_root': './data',
        'txt_path': 'Matterport360/train_public_Matterport360.txt',
        'is_fisheye': True,
        'is_metric_gt': True,
    },
}
SUPPORT_EXT = ['.jpg', '.png', '.JPG', '.JPEG']
USE_V2_PSEUDO = False
FOCAL_LENGTH_LIST = [200, 250, 300, 350]
FISHEYE_PROJECTION_MODELS = ['equidistant', 'equisolid', 'orthogonal', 'stereographic']


def read_image(image_path):
    return Image.open(image_path).convert('RGB')


def read_npz(npz_path):
    return sparse.load_npz(npz_path).toarray()


def metric_depth_to_disp(depth):
    depth = np.copy(depth)
    mask = depth > 0
    inverse_depth = 1 / depth[mask]
    min_value = np.min(inverse_depth)
    max_value = np.max(inverse_depth)
    depth[mask] = (inverse_depth - min_value) / (max_value - min_value)
    return depth


def relative_depth_to_disp(relative_depth):
    depth = np.copy(relative_depth)
    mask = depth > 0
    min_value = np.min(depth[mask])
    max_value = np.max(depth[mask])
    depth[mask] = (depth[mask] - min_value) / (max_value - min_value)
    return depth


def read_KITTI_depth(depth_path):
    with open(depth_path, 'rb') as f:
        with Image.open(f) as img:
            metric_gt = np.array(img, dtype=np.float32)
    return metric_gt / 1000.0


def read_Sintel_depth(depth_path):
    with open(depth_path, 'rb') as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == 202021.25, ' depth_read:: Wrong tag in flow file (shoule be: {0}, is: {1}). Big-endian machine?'.format(202021.25, check)
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and 1 < size < 100000000
        metric_gt = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))

    metric_gt[metric_gt > 9999] = 0
    return metric_gt


def read_MultiFOV_depth(depth_path):
    with open(depth_path, 'r ') as f:
        depth_values = f.read().split()
        depth_array = np.array(depth_values, dtype=np.float32)
        metric_gt = depth_array.reshape(480, 640)

    metric_gt[metric_gt > 200] = 0
    return metric_gt


def read_DIODE_depth(depth_path):
    mask_path = depth_path.replace('_depth.npy', '_depth_mask.npy')
    metric_gt = np.load(depth_path).squeeze()
    mask = np.load(mask_path).astype(bool)
    metric_gt[~mask] = 0
    return metric_gt


def read_ETH3D_depth(depth_path):
    with open(depth_path, 'rb') as f:
        metric_gt = np.fromfile(f, dtype=np.float32).reshape(4032, 6048)
    metric_gt[metric_gt == np.inf] = 0
    return metric_gt


def read_depth(dataset_name, depth_path):
    if dataset_name == 'kitti360':
        metric_gt = sparse.load_npz(depth_path).toarray()
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'SynWoodScape':
        metric_gt = np.load(depth_path)
        metric_gt[(1000.0 - metric_gt) < 1e-6] = 0
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'MultiFOV':
        metric_gt = read_MultiFOV_depth(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'Matterport360':
        metric_gt = np.load(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'KITTI':
        metric_gt = read_KITTI_depth(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'NYUv2':
        metric_gt = np.load(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'Sintel':
        metric_gt = read_Sintel_depth(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'DIODE':
        metric_gt = read_DIODE_depth(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'DDAD':
        metric_gt = np.load(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    elif dataset_name == 'ETH3D':
        metric_gt = read_ETH3D_depth(depth_path)
        relative_gt = metric_depth_to_disp(metric_gt)
    else:
        raise NotImplementedError

    return metric_gt, relative_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--save_dir', type=str, default='./data/public_dataset/eval_fisheye_dataset')
    parser.add_argument('-d', '--dataset_name', type=str, required=True)
    parser.add_argument('-n', '--sample_num', type=int, default=-1)
    parser.add_argument('-r', '--sample_rate', type=float, default=1.0)
    parser.add_argument('--data_root', type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    sample_num = args.sample_num
    sample_rate = args.sample_rate

    if dataset_name in datasets_list:
        dataset = datasets_list[dataset_name]
    else:
        assert os.path.exists(args.data_root)
        dataset = {'data_root': args.data_root, 'is_fisheye': False}
    print('====================== Processing {} ======================'.format(dataset_name))
    data_root = dataset['data_root']

    image_list = []
    depth_list = []

    if 'txt_path' in dataset:
        txt_path = osp.join(data_root, dataset['txt_path'])
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if sample_num > 0:
            step = len(lines) // sample_num
            lines = lines[::step][:sample_num]
        else:
            step = len(lines) // int(len(lines) * sample_rate)
            lines = lines[::step]

        output_path = osp.join(args.save_dir, '_'.join(['test_dataset', dataset_name, str(len(lines))]))

        for line in lines:
            img_info = json.loads(line)
            img_path = img_info['img_path'].replace('public_dataset', dataset['data_root'])
            gt_path = osp.join(dataset['data_root'].replace('public_dataset', ''), img_info['depth_gt_path'])

            image_list.append(img_path)
            depth_list.append(gt_path)
    else:
        for ext in SUPPORT_EXT:
            pattern = osp.join(data_root, '**', '*' + ext)
            image_list = glob.glob(pattern, recursive=True)
            if len(image_list) > 0:
                break
        assert len(image_list) > 0

        if sample_num > 0:
            step = len(image_list) // sample_num
            if step == 0:
                step = 1
            image_list = image_list[::step][:sample_num]
        else:
            step = len(image_list) // int(len(image_list) * sample_rate)
            image_list = image_list[::step]
        output_path = osp.join(args.save_dir, '_'.join(['test_dataset', dataset_name, str(len(image_list))]))

    output_ori_vis_path = osp.join(output_path, 'ori_vis')
    output_rgb_path = osp.join(output_path, 'rgb')
    output_vis_mask_path = osp.join(output_path, 'vis_mask')
    output_depth_gt_path = osp.join(output_path, 'depth', 'gt')
    os.makedirs(output_ori_vis_path, exist_ok=True)
    os.makedirs(output_rgb_path, exist_ok=True)
    os.makedirs(output_vis_mask_path, exist_ok=True)
    os.makedirs(output_depth_gt_path, exist_ok=True)

    sample_list = []
    for i in tqdm.tqdm(range(len(image_list))):
        filename = osp.splitext(osp.basename(image_list[i]))[0] + '.png'
        ori_vis_path = osp.join(output_ori_vis_path, dataset_name + '_' + filename)

        if 'txt_path' in dataset:
            metric_depth, relative_depth = read_depth(dataset_name, depth_list[i])
            sample = {'image': read_image(image_list[i]), 'depth': metric_depth if metric_depth is not None else relative_depth}
            h, w = sample['image'].height, sample['image'].width
            combine_image = Image.new('RGB', (2 * w, h))
            combine_image.paste(sample['image'], (0, 0))
            n_depth_gt = (relative_depth * 255.0).astype(np.uint8)
            combine_image.paste(Image.fromarray(n_depth_gt), (w, 0))
            combine_image.save(ori_vis_path)
            sample['vis_mask'] = np.ones((h, w)).astype(bool)
        else:
            sample = {'image': read_image(image_list[i])}
            sample['image'].save(ori_vis_path)

        if dataset['is_fisheye']:
            image_filename = '_'.join([str(i), dataset_name, filename])
            sample['image'].save(osp.join(output_rgb_path, image_filename))

            if 'depth' in sample:
                depth_filename = image_filename.replace('.png', '.npy')
                np.save(osp.join(output_depth_gt_path, depth_filename), sample['depth'])
        else:
            sample['image'] = np.array(sample['image'])
            for pm_id, pm_name in enumerate(FISHEYE_PROJECTION_MODELS):
                for f in FOCAL_LENGTH_LIST:
                    distorted_sample = pinhole2fisheye(sample, f, projection_model=pm_id, crop_valid=True)

                    image_filename = '_'.join([dataset_name, pm_name, str(f), str(i), filename])
                    Image.fromarray(distorted_sample['image']).save(osp.join(output_rgb_path, image_filename))

                    if 'depth' in distorted_sample:
                        valid_min_depth = sample['depth'][sample['depth'] > 0].min()
                        invalid_depth = (distorted_sample['depth'] > 0) & (distorted_sample['depth'] < valid_min_depth)
                        if np.any(invalid_depth):
                            distorted_sample['depth'][invalid_depth] = 0
                            print('Minimum depth becomes smaller after pinhole2fisheye, so we set them to 0')

                        depth_filename = image_filename.replace('.png', '.npy')
                        np.save(osp.join(output_depth_gt_path, depth_filename), distorted_sample['depth'])

                    if 'vis_mask' in distorted_sample:
                        Image.fromarray(distorted_sample['vis_mask']).save(osp.join(output_vis_mask_path, image_filename))

    print('=================== {} samples done ==================='.format(len(image_list)))
    print('=================== save folder: {} ==================='.format(output_path))
