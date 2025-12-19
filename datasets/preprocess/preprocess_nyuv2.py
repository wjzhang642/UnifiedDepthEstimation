import io
import os
import os.path as osp
import json
import h5py
import numpy as np
from PIL import Image


def h5_loader(bytes_stream):
    # Reference: https://github.com/dwofk/fast-depth/blob/master/dataloaders/dataloader.py@L8-L13
    f = io.BytesIO(bytes_stream)
    h5f = h5py.file(f, 'r')
    rgb = np.array(h5f['rgb'])
    rgb = np.tranpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/NYUv2'
    prefix = 'public_dataset/NYUv2'

    for split in ['train', 'val']:
        split_path = osp.join(dataset_dir, split)
        if not osp.exists(split_path):
            continue

        txt_content = []
        h5_files = os.listdir(split_path)
        for file in h5_files:
            file_path = osp.join(split_path, file)
            with open(file_path, 'rb') as f_in:
                image, depth = h5_loader(f_in)

            image_path = file_path.replace('.h5', '.jpg')
            gt_path = file_path.replace('.h5', '.npy')
            image_name = osp.basename(image_path)
            gt_name = osp.basename(gt_path)

            Image.fromarray(image).save(image_path)
            np.save(gt_path, depth)

            img_path = osp.join(prefix, split, 'official', image_name)
            depth_gt_path = osp.join(prefix, split, 'official', gt_name)

            img_path = img_path.replace('\\', '/')
            depth_gt_path = depth_gt_path.replace('\\', '/')
            txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'NYUv2'})

        save_file_name = 'train_public_NYUv2_' + split
        save_file_path = osp.join(dataset_dir, save_file_name)
        print(f'Total {len(txt_content)} samples.')
        with open(save_file_path + '.txt', 'w') as f_out:
            for item in txt_content:
                item_str = json.dumps(item, ensure_ascii=False)
                f_out.write(item_str)
                f_out.write('\n')
