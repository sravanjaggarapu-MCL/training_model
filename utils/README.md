# utils/

Helper scripts run **from the repo root**:

| Script | Purpose | Run |
|---|---|---|
| `split_dataset.py` | Split a flat labeled export (images + .txt together) into the `datasets/images|labels/{train,val}` layout | `python utils/split_dataset.py --source /path/to/export` |
| `check_dataset.py` | Validate the dataset before training: image↔label pairing, label format, normalized coords, class balance | `python utils/check_dataset.py` |

Always run `check_dataset.py` before launching a server job — it exits non-zero on errors, so it can also be the first step of a training pipeline.
