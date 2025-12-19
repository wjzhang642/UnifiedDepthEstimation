import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/KITTI360/data_2d_raw'
    prefix = 'public_dataset/KITTI360/data_2d_raw'
    type2data = {'pinhole': {'img_dirname': 'data_rect', 'folders': ['image_00', 'image_01'], 'cameras': ['pinhole_left', 'pinhole_right']},
                 'fisheye': {'img_dirname': 'data_rgb', 'folders': ['image_02', 'image_03'], 'cameras': ['fisheye_left', 'fisheye_right']}}

    scene_names = os.listdir(dataset_dir)
    scene_names = [s_name for s_name in scene_names if s_name.startswith('2013_05_28_drive')]
    for scene_name in scene_names:
        if scene_name in ['2013_05_28_drive_0008_sync', '2013_05_28_drive_0018_sync']:  # test set
            continue
    scene_path = osp.join(dataset_dir, scene_name)
    scene_id = scene_name.split('_'[-2])

    for data_type in type2data:
        txt_content = []
        img_dirname = type2data[data_type]['img_dirname']
        folders = type2data[data_type]['folders']
        cameras = type2data[data_type]['cameras']

        for folder, camera in zip(folders, cameras):
            folder_path = osp.join(scene_path, folder)
            if not osp.exists(folder_path):
                continue

            img_dir = osp.join(folder_path, img_dirname)
            gt_dir = osp.join(folder_path, 'depthmap')
            assert osp.exists(img_dir)
            assert osp.exists(gt_dir), f'{gt_dir} not exists!'

            image_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            gt_names = [i_name.replace('.png', '_depth.npz') for i_name in image_names]

            image_paths = [osp.join(img_dir, f_name) for f_name in image_names]
            gt_paths = [osp.join(gt_dir, f_name) for f_name in gt_names]

            for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
                assert osp.exists(image_path) and osp.exists(gt_path)
                img_path = osp.join(prefix, scene_name, folder, img_dirname, image_name)
                depth_gt_path = osp.join(prefix, scene_name, folder, 'depthmap', gt_name)

                img_path = img_path.replace('\\', '/')
                depth_gt_path = depth_gt_path.replace('\\', '/')
                txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'KITTI360'})

        save_file_name = 'train_public_KITTI360_' + data_type + '_' + scene_id
        save_file_path = osp.join(dataset_dir, save_file_name)
        print(f'Total {len(txt_content)} samples.')
        with open(save_file_path + '.txt', 'w') as f_out:
            for item in txt_content:
                item_str = json.dumps(item, ensure_ascii=False)
                f_out.write(item_str)
                f_out.write('\n')
