# PoolGuard — Drowning Detection Model Training

Training repository for the PoolGuard (Swimming Pool Drowning Detection) YOLO model. Built to run locally for quick checks and on the company GPU servers for full training runs. The output of this repo — `weights/<name>_best.pt` — plugs directly into PoolGuard's `detector.py`.

The model detects three classes: **drowning**, **swimming**, **person_out_of_water**. In PoolGuard these map to the API contract events `DROWNING` (which can auto-deploy the rescue rod), `SWIMMING`, and `PERSON_DETECTED`.

## Repository Structure

```
repo/
├── train.py            Training entry point (Ultralytics API)
├── requirements.txt    Python dependencies
├── models/             Base pretrained checkpoints (auto-downloaded, gitignored)
├── datasets/           Your dataset in YOLO format (gitignored — see its README)
├── configs/
│   ├── data.yaml           Dataset paths + class list (IDs matter!)
│   └── train_defaults.yaml Team presets: quick_check / server_full / edge
├── weights/            Trained outputs: <run-name>_best.pt (gitignored)
├── utils/
│   ├── split_dataset.py    Flat labeled export → train/val layout
│   └── check_dataset.py    Validate dataset before burning GPU hours
└── README.md
```

Data, base models, and trained weights are all **gitignored** — this repo carries code and configs only, which is exactly what you want for cloning onto training servers.

## Setup

```bash
git clone <this-repo>
cd repo
pip install -r requirements.txt
```

On a company GPU server: if a specific CUDA build of PyTorch is required, install torch first (per pytorch.org), then run the requirements install — pip keeps the existing torch.

## Workflow

**1. Add the dataset.** Drop it into `datasets/` in YOLO format (layout in `datasets/README.md`). If your labeling tool exported a flat folder of images + `.txt` files together:

```bash
python utils/split_dataset.py --source /path/to/export --val 0.2
```

On the server you can alternatively leave `datasets/` empty and point `path:` in `configs/data.yaml` at a shared absolute dataset location.

**2. Validate before training.** Catches missing labels, bad coordinates, unknown class IDs, and class imbalance in seconds instead of failing a GPU job minutes in:

```bash
python utils/check_dataset.py
```

It exits non-zero on errors, so it works as the first step of an automated pipeline.

**3. Train.**

```bash
# Quick sanity run (small model, few epochs) — always do this first
python train.py --model yolov8n.pt --epochs 25 --name sanity

# Full server run
python train.py --model yolov8s.pt --epochs 150 --batch 32 --device 0 --name poolguard_v1

# YOLOv5 family instead
python train.py --model yolov5su.pt --epochs 150 --name poolguard_v5

# Resume an interrupted run
python train.py --resume
```

Key flags: `--device 0` (GPU) / `--device cpu`, `--batch` (halve it on CUDA out-of-memory), `--patience 30` (early stopping). Team presets are documented in `configs/train_defaults.yaml`.

**4. Collect the result.** Full logs, curves, and checkpoints land in `runs/<name>/`; the best checkpoint is auto-copied to `weights/<name>_best.pt`.

**5. Deploy to PoolGuard.** Copy the weights file next to PoolGuard's `detector.py`:

```bash
python detector.py --model poolguard_v1_best.pt
```

`detector.py` prints the model's class→API-event mapping at startup — verify it shows `drowning -> DROWNING` etc. on first run.

## Which base model?

| Base | Use when |
|---|---|
| `yolov8n.pt` | Dataset sanity checks, and the **final Raspberry Pi deployment** (nano is what stays real-time on Pi CPU) |
| `yolov8s.pt` | Main server training runs — accuracy/speed sweet spot |
| `yolov5su.pt` / `yolov5mu.pt` | If the team specifically wants the YOLOv5 family; trains with the identical command |

All of these are fine-tuned, not trained from scratch — pretrained weights + your pool dataset is both faster and more accurate than scratch training at this dataset size.

## Rules

- **Never reorder the class list in `configs/data.yaml`** — IDs are baked into every label file.
- Don't commit datasets or weights; git stays code-only.
- Name runs meaningfully (`--name poolguard_v2_more_night_data`) — the run name becomes the weights filename and your only clue later.
- The `drowning` class physically triggers rescue hardware in PoolGuard. Bias labeling and validation toward this class: it needs the most examples and the hardest review, not the fewest.
