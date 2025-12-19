import os
import json
import os.path as osp


if __name__ == '__main__':
    dataset_dir = './data/BlendedMVS'
    prefix = 'public_dataset/BlendedMVS'
    txt_content = []

    scene_names = os.listdir(dataset_dir)
    for scene_name in scene_names:
        scene_path = osp.join(dataset_dir, scene_name)

        img_dir = osp.join(scene_path, 'blended_images')
        gt_dir = osp.join(scene_path, 'rendered_depth_maps')

        image_names = os.listdir(img_dir)
        image_names = [i_name for i_name in image_names if not i_name.endswith('_masked.jpg')]
        gt_names = [i_name.replace('.jpg', '.pfm') for i_name in image_names]

        image_paths = [osp.join(img_dir, f_name) for f_name in image_names]
        gt_paths = [osp.join(gt_dir, f_name) for f_name in gt_names]

        for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
            assert osp.exists(image_path) and osp.exists(gt_path)
            img_path = osp.join(prefix, scene_name, 'blended_images', image_name)
            depth_gt_path = osp.join(prefix, scene_name, 'rendered_depth_maps', gt_name)

            img_path = img_path.replace('\\', '/')
            depth_gt_path = depth_gt_path.replace('\\', '/')
            txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'BlendedMVS'})

    save_file_name = 'train_public_BlendedMVS'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
