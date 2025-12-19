## Description
The metadata (e.g., img_path, depth_gt_path) from different datasets is extracted, formatted consistently, and saved into  `.txt` files.

## Data Preparation
Download the dataset(s) you wish to use and place them in the `./data/public_dataset/` directory.

## Converation
Run the corresponding script for your dataset. For example:
```bash
python preprocess_blendedmvs.py
```