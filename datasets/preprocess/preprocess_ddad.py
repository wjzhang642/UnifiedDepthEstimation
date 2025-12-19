import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/DDAD'
    prefix = 'public_dataset/DDAD/ddad_train_val'

    train_txt_content = []
    val_txt_content = []

    ddad_train_val_dir = osp.join(dataset_dir, 'ddad_train_val')
    scenes = os.listdir(ddad_train_val_dir)
    scenes = [s for s in scenes if osp.isdir(osp.join(ddad_train_val_dir, s))]
    scenes = sorted(scenes)
    for idx, scene in enumerate(scenes):
        scene_path = osp.join(ddad_train_val_dir, scene)
        rgb_path = osp.join(scene_path, 'rgb')
        depth_path = osp.join(scene_path, 'depth')

        for cam in ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']:
            cam_rgb_path = osp.join(rgb_path, cam)
            cam_depth_path = osp.join(depth_path, cam)

            image_names = [f for f in os.listdir(cam_rgb_path) if f.endswith('.png')]
            gt_names = [i_name.replace('.png', '.npy') for i_name in image_names]

            image_paths = [osp.join(cam_rgb_path, f_name) for f_name in image_names]
            gt_paths = [osp.join(cam_depth_path, f_name) for f_name in gt_names]

            for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
                assert osp.exists(image_path) and osp.exists(gt_path)
                img_path = osp.join(prefix, scene, 'rgb', cam, image_name)
                depth_gt_path = osp.join(prefix, scene, 'depth', gt_name)

                img_path = img_path.replace('\\', '/')
                depth_gt_path = depth_gt_path.replace('\\', '/')

                if idx < 150:
                    train_txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'DDAD'})
                else:
                    val_txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'DDAD'})

    save_file_name = 'train_public_DDAD_train'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(train_txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in train_txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')

    save_file_name = 'train_public_DDAD_val'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(val_txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in val_txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
