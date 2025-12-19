import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/Hypersim'
    prefix = 'public_dataset/Hypersim'

    scenes = os.listdir(dataset_dir)
    txt_content = []
    for scene in scenes:
        scene_path = osp.join(dataset_dir, scene)
        if not osp.isdir(scene_path):
            continue
        scene_path = osp.join(scene_path, 'images')
        cam_path_list = [d for d in os.listdir(scene_path) if d.endswith('final_preview')]
        for cam_path in cam_path_list:
            img_dir = osp.join(scene_path, cam_path)
            gt_dir = img_dir.replace('final_preview', 'geometry_hdf5')

            image_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
            gt_names = [i_name.replace('color.jpg', 'depth_meters.hdf5') for i_name in image_names]

            image_paths = [osp.join(img_dir, f_name) for f_name in image_names]
            gt_paths = [osp.join(gt_dir, f_name) for f_name in gt_names]

            for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
                assert osp.exists(image_path) and osp.exists(gt_path)
                img_path = osp.join(prefix, scene, 'images', cam_path, image_name)
                depth_gt_path = osp.join(prefix, scene, 'images', cam_path.replace('final_preview', 'geometry_hdf5'), gt_name)

                img_path = img_path.replace('\\', '/')
                depth_gt_path = depth_gt_path.replace('\\', '/')
                txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'Hypersim'})

    save_file_name = 'train_public_Hypersim'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
