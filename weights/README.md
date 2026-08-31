# weights/

Trained output checkpoints. After each run, `train.py` copies the best model here as `<run-name>_best.pt` (e.g. `poolguard_best.pt`).

Deploy into PoolGuard by copying the file next to `detector.py` and running:

```
python detector.py --model poolguard_best.pt
```

Contents are gitignored (weights are large); share final weights through your team's artifact storage, not git.
