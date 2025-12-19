import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/TartanAir'
    scene_names = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'carwelding', 'endofwolrd', 'gascola',
                   'hospital', 'japanesealley', 'neighborhood', 'ocean', 'office', 'office2', 'oldtown', 'seasidetown',
                   'seasonsforest', 'seasonsforest_winter', 'soulcity', 'westerndesert']
    prefix = 'public_dataset/TartanAir'

    txt_content = []
    for scene_name in scene_names:
        for mode in ['Easy', 'Hard']:
            data_dir = osp.join(dataset_dir, scene_name, mode)
            part_names = os.listdir(data_dir)
            part_paths = [osp.join(data_dir, p_name) for p_name in part_names]

            for dirt in ['left', 'right']:
                img_dirt = 'image_' + dirt
                gt_dirt = 'depth_' + dirt

                for part_name, part_path in zip(part_names, part_paths):
                    image_dir = osp.join(part_path, img_dirt)
                    gt_dir = osp.join(part_path, gt_dirt)

                    image_names = os.listdir(image_dir)
                    gt_names = [i_name.replace('.png', '._depth.npy') for i_name in image_names]

                    image_paths = [osp.join(image_dir, f_name) for f_name in image_names]
                    gt_paths = [osp.join(gt_dir, f_name) for f_name in gt_names]

                    for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
                        assert osp.exists(image_path) and osp.exists(gt_path)
                        img_path = osp.join(prefix, scene_name, mode, part_name, img_dirt, image_name)
                        depth_gt_path = osp.join(prefix, scene_name, mode, part_name, gt_dirt, gt_name)

                        img_path = img_path.replace('\\', '/')
                        depth_gt_path = depth_gt_path.replace('\\', '/')
                        txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'TartanAir'})

    save_file_name = 'train_public_TartanAir'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
