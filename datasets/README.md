# datasets/

Place the dataset here in **YOLO format** (this folder is gitignored — data never goes to GitHub; on the company server, either copy the dataset in or point `configs/data.yaml` `path:` at a shared absolute location).

Expected layout:

```
datasets/
├── images/
│   ├── train/    img001.jpg, img002.jpg, ...
│   └── val/      img101.jpg, ...
└── labels/
    ├── train/    img001.txt, img002.txt, ...
    └── val/      img101.txt, ...
```

Each label `.txt` has one line per object:

```
<class_id> <cx> <cy> <w> <h>      # all coords normalized 0–1
```

Class IDs (must match `configs/data.yaml` — do not reorder):

| ID | Class |
|----|-------|
| 0  | drowning |
| 1  | swimming |
| 2  | person_out_of_water |

Have a flat export instead (images and .txt mixed in one folder)? Use `python utils/split_dataset.py --source <folder>` to build this layout, then validate with `python utils/check_dataset.py`.
