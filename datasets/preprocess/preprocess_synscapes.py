import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/Synscapes'
    prefix = 'public_dataset/Synscapes'

    data_dir = osp.join(dataset_dir, 'img')

    img_dir = osp.join(data_dir, 'rgb-2k')
    gt_dir = osp.join(data_dir, 'depth')

    image_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    gt_names = [i_name.replace('.png', '.exr') for i_name in image_names]

    image_paths = [osp.join(img_dir, f_name) for f_name in image_names]
    gt_paths = [osp.join(gt_dir, f_name) for f_name in gt_names]

    txt_content = []
    for image_name, gt_name, image_path, gt_path in zip(image_names, gt_names, image_paths, gt_paths):
        assert osp.exists(image_path) and osp.exists(gt_path)
        img_path = osp.join(prefix, 'img', 'rgb-2k', image_name)
        depth_gt_path = osp.join(prefix, 'img', 'depth', gt_name)

        img_path = img_path.replace('\\', '/')
        depth_gt_path = depth_gt_path.replace('\\', '/')
        txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'Synscapes'})

    save_file_name = 'train_public_Synscapes'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
