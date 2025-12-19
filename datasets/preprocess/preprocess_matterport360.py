import os
import cv2
import os.path as osp
import json
import glob
import struct
import numpy as np
from omnicv import fisheyeImgConv
from tqdm import tqdm


def read_matterport360_depth(depth_path):
    INT_BYTES = 4
    with open(depth_path, 'rb') as f:
        # Skip the first 4 bytes
        _ = f.read(INT_BYTES)

        # Read height and width
        w = struct.unpack('<i', f.read(INT_BYTES))[0]
        h = struct.unpack('<i', f.read(INT_BYTES))[0]

        # Read the pixel data (4 bytes each => 32-bit float)
        num_pixels = h * w
        pixel_bytes = f.read(num_pixels * 4)
        depth_vals = struct.unpack(f'<{num_pixels}f', pixel_bytes)

    # Reshape
    metric_gt = np.array(depth_vals, dtype=np.float32).reshape((h, w))
    return metric_gt


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/Matterport360'
    prefix = 'public_dataset/Matterport360'

    transform_dir = os.path.join(dataset_dir, 'transform_data')
    mapper = fisheyeImgConv()

    image_paths = glob.glob(os.path.join(dataset_dir, 'data', '**', '*_rgb.png'), recursive=True)
    txt_content = []
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        fisheye_bgr = mapper.equirect2Fisheye_EUCM(img, outShape=[1024, 1024], f=300, a_=0.6, b_=2, angle=[0, 0, 0])

        gt_path = img_path.replace('_rgb.png', '_depth.dpt')
        gt = read_matterport360_depth(gt_path)

        gt_min, gt_max = gt.min(), gt.max()
        gt = ((gt - gt_min) / (gt_max - gt_min) * 255).astype(np.uint8)
        gt_bgr = mapper.equirect2Fisheye_EUCM(np.stack([gt, gt, gt], axis=-1), outShape=[1024, 1024], f=300, a_=0.6, b_=2, angle=[0, 0, 0])
        gt_fisheye = gt_bgr[:, :, 0] == 255

        vis_mask = np.ones_like(gt, dtype=np.uint8) * 255
        vis_mask_bgr = mapper.equirect2Fisheye_EUCM(np.stack([vis_mask_bgr, vis_mask_bgr, vis_mask_bgr], axis=-1), outShape=[1024, 1024], f=300, a_=0.6, b_=2, angle=[0, 0, 0])
        vis_mask = vis_mask_bgr[:, :, 0] == 255

        gt_fisheye[vis_mask] = gt_fisheye[vis_mask] / 255.0 * (gt_max - gt_min) + gt_min

        img_path = img_path.replace('/data/', '/transform_data/')
        gt_path = gt_path.replace('/data/', '/transform_data/').replace('_depth.dpt', '_depth.npy')
        os.makedirs(os.path.split(img_path)[0], exist_ok=True)
        cv2.imwrite(img_path, fisheye_bgr)
        np.save(gt_path, gt_fisheye)
        # cv2.imwrite(gt_path, gt_fisheye)

        filename = os.path.split(img_path)[-1]
        img_path = img_path.replace(dataset_dir, '')
        img_path = img_path[1:] if img_path[0] == '/' else img_path
        img_path = os.path.join(prefix, img_path)
        depth_gt_path = img_path.replace('_rgb.png', 'depth.npy')
        txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'Matterport360'})

    save_file_name = 'train_public_Matterport360'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
