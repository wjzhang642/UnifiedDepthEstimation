import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/ETH3D'
    prefix = 'public_dataset/ETH3D'

    scenes = os.listdir(dataset_dir)
    scenes = [s for s in scenes if osp.isdir(osp.join(dataset_dir, s))]

    scenes = sorted(scenes)
    txt_content = []
    for idx, scene in enumerate(scenes):
        scene_path = osp.join(dataset_dir, scene)

        rgb_path = osp.join(scene_path, 'images', 'dslr_images')
        depth_path = osp.join(scene_path, 'ground_truth_depth', 'dslr_images')

        image_names = [f for f in os.listdir(rgb_path) if f.endswith('.JPG')]
        gt_names = [i_name for i_name in image_names]

        image_paths = [osp.join(rgb_path, f_name) for f_name in image_names]
        gt_paths = [osp.join(depth_path, f_name) for f_name in gt_names]

        for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
            assert osp.exists(image_path) and osp.exists(gt_path)
            img_path = osp.join(prefix, scene, 'images', 'dslr_images', image_name)
            depth_gt_path = osp.join(prefix, scene, 'ground_truth_depth', 'dslr_images', gt_name)

            img_path = img_path.replace('\\', '/')
            depth_gt_path = depth_gt_path.replace('\\', '/')
            txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'ETH3D'})

    save_file_name = 'train_public_ETH3D'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
