import os
import os.path as osp
import json
import glob


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/MultiFOV'
    prefix = 'public_dataset/MultiFOV'

    image_paths = glob.glob(os.path.join(dataset_dir, 'data', 'img', '*.png'), recursive=True)
    txt_content = []
    for img_path in image_paths:
        filename = os.path.split(img_path)[-1]
        img_path = img_path.replace(dataset_dir, '')
        img_path = img_path[1:] if img_path[0] == '/' else img_path
        img_path = os.path.join(prefix, img_path)
        depth_gt_path = osp.join(prefix, 'data/depth', filename.replace('.png', 'depth'))
        txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'MultiFOV'})

    save_file_name = 'train_public_MultiFOV'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
