import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/HRWSI'
    prefix = 'public_dataset/HRWSI'
    part_name = 'HR-WSI'

    for split in ['train', 'val']:
        txt_content = []
        scene_path = osp.join(dataset_dir, part_name)
        img_dir = osp.join(scene_path, split, 'imgs')
        gt_dir = osp.join(scene_path, split, 'gts')

        image_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        gt_names = [i_name.replace('.jpg', '.png') for i_name in image_names]

        image_paths = [osp.join(img_dir, f_name) for f_name in image_names]
        gt_paths = [osp.join(gt_dir, f_name) for f_name in gt_names]

        for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
            assert osp.exists(image_path) and osp.exists(gt_path)
            img_path = osp.join(prefix, part_name, split, 'imgs', image_name)
            depth_gt_path = osp.join(prefix, part_name, split, 'gts', gt_name)

            img_path = img_path.replace('\\', '/')
            depth_gt_path = depth_gt_path.replace('\\', '/')
            txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'HRWSI'})

        save_file_name = 'train_public_HRWSI_' + split
        save_file_path = osp.join(dataset_dir, save_file_name)
        print(f'Total {len(txt_content)} samples.')
        with open(save_file_path + '.txt', 'w') as f_out:
            for item in txt_content:
                item_str = json.dumps(item, ensure_ascii=False)
                f_out.write(item_str)
                f_out.write('\n')
