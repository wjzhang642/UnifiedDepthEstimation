import os
import os.path as osp
import json


if __name__ == '__main__':
    dataset_dir = './data/public_dataset/bdd100k'
    prefix = 'public_dataset/bdd100k'

    txt_content = []
    for split in ['train', 'val', 'test']:
        split_path = osp.join(dataset_dir, 'images', '100k', split)
        image_names = [f for f in os.listdir(split_path)]

        for image_name in image_names:
            img_path = osp.join(prefix, 'images', '100k', image_name)

            img_path = img_path.replace('\\', '/')
            txt_content.append({'img_path': img_path, 'dataset_name': 'bdd100k'})

    save_file_name = 'train_public_bdd100k'
    save_file_path = osp.join(dataset_dir, save_file_name)
    print(f'Total {len(txt_content)} samples.')
    with open(save_file_path + '.txt', 'w') as f_out:
        for item in txt_content:
            item_str = json.dumps(item, ensure_ascii=False)
            f_out.write(item_str)
            f_out.write('\n')
