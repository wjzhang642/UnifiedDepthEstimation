import os
import json
import numpy as np
import PIL.Image as pil


if __name__ == '__main__':
    """
    Some samples lack valid depth maps and should be filtered out.
    """
    dataset_root = './data'
    files = os.listdir(os.path.join(dataset_root, 'DIML'))

    for file in files:
        if '.txt' not in file:
            continue

        txt_path = os.path.join(dataset_root, 'DIML', 'file')
        txt_content = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        print(file)
        print('{} origin samples'.format(len(lines)))
        for line in lines:
            img_info = json.loads(line)

            abs_depth_gt_path = os.path.join(dataset_root.replace('public_dataset', ''), img_info['depth_gt_path'])
            with open(abs_depth_gt_path, 'rb') as f:
                with pil.open(f) as img:
                    depth_gt = np.array(img, dtype=np.float32) / 1000
            if (depth_gt > 0).sum() < 100:
                print((depth_gt > 0).sum())
                continue
            txt_content.append(img_info)
        print('{} filtered samples'.format(len(txt_content)))

        save_file_name = file.replace('.txt', '_clean.txt')
        save_file_path = os.path.join(dataset_root, save_file_name)
        with open(save_file_path + '.txt', 'w') as f_out:
            for item in txt_content:
                item_str = json.dumps(item, ensure_ascii=False)
                f_out.write(item_str)
                f_out.write('\n')