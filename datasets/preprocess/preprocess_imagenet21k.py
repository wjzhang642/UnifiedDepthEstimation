import os.path as osp
import json
import glob


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/ImageNet-21K'
    prefix = 'public_dataset/ImageNet-21K'
    pseudo_label_prefix = 'depth_anything_pseudo/public_dataset/ImageNet-21K'

    txt_content = []
    image_paths = glob.glob(osp.join(dataset_dir, '**', '*.JPEG'), recursive=True)
    for img_path in image_paths:
        img_path = img_path.replace(dataset_dir, '')
        img_path = img_path[1:] if img_path[0] == '/' else img_path
        img_path = osp.join(prefix, img_path)
        depth_gt_path = osp.join(prefix, img_path)
        depth_gt_path = osp.join(pseudo_label_prefix, img_path.replace('.jpg', '.npy'))
        txt_content.append({'img_path': img_path, 'depth_gt_path': depth_gt_path, 'dataset_name': 'ImageNet-21K'})

    save_file_name = 'train_public_ImageNet-21K'
    save_file_path = osp.join(dataset_dir, save_file_name)
    split_num = 14
    print(f'Total {len(txt_content)} samples.')
    for num in range(split_num):
        with open(save_file_path + '_split' + str(num) + '.txt', 'w') as f_out:
            for item in txt_content[num::split_num]:
                item_str = json.dumps(item, ensure_ascii=False)
                f_out.write(item_str)
                f_out.write('\n')
