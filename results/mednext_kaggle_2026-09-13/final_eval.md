# mednext final evaluation

- Input resolution: (64, 96, 112) (reduced from 96x128x144 for a free-tier T4)
- Epochs completed: 26 (stopped at deadline)
- Best val loss: 0.1474
- Eval set: validation (75 patients) -- no locally-available files are tagged dev_split=='test' in the official meta_data.csv, so this reuses val, not a held-out test set
- Eval L1 loss: 0.1474 ± 0.0568
