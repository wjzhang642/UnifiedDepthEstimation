import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/DIODE'
    prefix = 'public_dataset/DIODE'

    for split in ['train', 'val']:
        split_path = osp.join(dataset_dir, split)
        if not osp.exists(split_path):
            continue

        txt_content = []
        for scene_type in ['indoors', 'outdoor']:
            type_path = osp.join(split_path, scene_type)
            scenes = os.listdir(type_path)

            for scene in scenes:
                scene_path = osp.join(type_path, scene)
                scans = os.listdir(scene_path)

                for scan in scans:
                    scan_path = osp.join(scene_path, scan)

                    image_names = [f for f in os.listdir(scan_path) if f.endswith('.png')]
                    gt_names = [i_name.replace('.png', '._depth.npy') for i_name in image_names]
                    mask_names = [i_name.replace('.png', '._depth_mask.npy') for i_name in image_names]

                    image_paths = [osp.join(scan_path, f_name) for f_name in image_names]
                    gt_paths = [osp.join(scan_path, f_name) for f_name in gt_names]
                    mask_paths = [osp.join(scan_path, f_name) for f_name in mask_names]

                    for image_name, gt_name, mask_name, image_path, gt_path, mask_path in zip(image_names, gt_names, mask_names, image_paths, gt_paths, mask_paths):
                        assert osp.exists(image_path) and osp.exists(gt_path) and osp.exists(mask_path)
                        img_path = osp.join(prefix, split, scene_type, scene, scan, image_name)
                        depth_gt_path = osp.join(prefix, split, scene_type, scene, scan, gt_name)

                        img_path = img_path.replace('\\', '/')
                        depth_gt_path = depth_gt_path.replace('\\', '/')
                        txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'DIODE'})

        save_file_name = 'train_public_DIODE_' + split
        save_file_path = osp.join(dataset_dir, save_file_name)
        print(f'Total {len(txt_content)} samples.')
        with open(save_file_path + '.txt', 'w') as f_out:
            for item in txt_content:
                item_str = json.dumps(item, ensure_ascii=False)
                f_out.write(item_str)
                f_out.write('\n')
