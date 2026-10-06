import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from nnunet_mednext import create_mednext_v1
import data_loader
import yaml
import argparse
import os

# Single source of truth for input/output resolution -- originally 96x128x144
# (the challenge's own dummy/full configs), reduced to fit a free-tier Colab
# T4's 15GB (repeatedly OOM'd at 96x128x144 even at the smallest model_id='S'
# with mixed precision -- see handoff.md for the full mitigation history).
# Restore the original fidelity on a >=24GB GPU by changing this one line;
# this constant overrides whatever in_size/out_size the config YAML has, so
# there is exactly one place resolution is decided.
INPUT_RESOLUTION = (64, 96, 112)

'''
In this script, we provide a basic (and simple) pipeline designed for successful execution.
There are numerous advanced AI methodologies and strategies that could potentially improve the model's performance. 
We encourage participants to explore these AI technologies independently. The organizers will not provide much support for these explorations.
Please note that discussions/questions about AI tech explorations are not supposed to be raised in the repository issues.

Reminder: The information provided in the meta files is crucial, as it directly impacts how the reference is created. 
An example of how to use these information are provided in the data_loader.py. 
If you have questions related to clinical backgrounds, feel free to start a discussion.
'''

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('cfig_path',  type = str)
parser.add_argument('--phase', default = 'train', type = str)
parser.add_argument('--resume', default=None, type=str,
                     help='checkpoint path to resume weights from (e.g. after a dropped Colab session)')
parser.add_argument('--epochs_override', type=int, default=None,
                     help="override cfig['num_epochs'] without editing the YAML -- useful when relaunching a "
                          "--resume run further than the config file's original target (the cosine LR schedule's "
                          "T_max is derived from this value, so bumping it also correctly extends the schedule).")
parser.add_argument('--deadline_epoch_seconds', type=float, default=None,
                     help='Unix epoch seconds (UTC) hard stop -- train-to-time, not train-to-epoch. '
                          'Checked before each epoch; whatever epoch was reached is the deliverable. '
                          'Best checkpoint (by val loss) is always what final eval uses, not last.')
args = parser.parse_args()

cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
if args.epochs_override is not None:
    print(f"[config] num_epochs overridden {cfig['num_epochs']} -> {args.epochs_override}")
    cfig['num_epochs'] = args.epochs_override
cfig['loader_params']['in_size'] = list(INPUT_RESOLUTION)
cfig['loader_params']['out_size'] = list(INPUT_RESOLUTION)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ data loader -----------------#
loaders = data_loader.GetLoader(cfig = cfig['loader_params'])
train_loader =loaders.train_dataloader()
val_loader = loaders.val_dataloader()

# ------------- Network ------------------ # 
model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
  num_classes = cfig['model_params']['out_channels'],
  model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
  kernel_size = cfig['model_params']['kernel_size'],   # 3x3x3 and 5x5x5 were tested in publication
  deep_supervision = cfig['model_params']['deep_supervision']   
).to(device)

# ------------ loss -----------------------#
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr':cfig['lr']}], lr=cfig['lr'])

# NOTE: this used to construct with last_epoch=cfig['num_epochs'] (e.g. 50)
# on every run, including fresh (non-resumed) ones -- last_epoch is a
# *resume position*, not a config echo, so a fresh run was always starting
# ~50 steps into its own schedule (negligible against T_max in the tens of
# thousands, so harmless there) but a --resume run reset right back to that
# same ~50-steps-in position regardless of how far training had actually
# progressed, i.e. every resumed session's LR jumped back up near its
# initial value instead of continuing to anneal. Same class of bug as
# dosenet3d's (see models/dosenet3d/train.py) -- fixed the same way: proper
# last_epoch=-1 on fresh construction, then scheduler.load_state_dict() to
# actually resume position when a full checkpoint is available.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * cfig['num_epochs'], last_epoch=-1)

criterion = nn.L1Loss()

# Checkpoints are saved below as {"model", "optimizer", "scheduler", "epoch"}
# so --resume can continue the LR schedule and optimizer momentum instead of
# restarting them. Old-format checkpoints (a bare model state_dict, no
# "model" key) still load fine -- model weights only, optimizer/scheduler
# start fresh, exactly the previous behaviour.
start_epoch = 0
resumed_val_loss = None
if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[resume] could not restore optimizer state: {e}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[resume] could not restore scheduler state: {e}")
        start_epoch = ckpt.get("epoch", 0)
        resumed_val_loss = ckpt.get("val_loss")
        print(f"Resumed full training state from {args.resume}: continuing at epoch {start_epoch + 1}")
    else:
        model.load_state_dict(ckpt)
        print(f"Resumed weights only (legacy checkpoint format, no optimizer/scheduler state) "
              f"from {args.resume} -- optimizer/schedule start fresh from epoch 1")

# Mixed precision: this model+resolution's peak activation memory sits right
# at a T4's ~14.5GB usable capacity (repeatedly OOMs by ~300MB at batch_size=1,
# model_id='S' even after ruling out fragmentation -- PYTORCH_CUDA_ALLOC_CONF
# didn't help since there wasn't meaningfully more reserved-but-unallocated
# memory to reclaim). AMP roughly halves activation memory for conv-heavy
# nets like this with negligible accuracy impact -- try that before further
# shrinking the model/resolution.
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# -----------Training loop --------------- #

nbatch_per_log = max(int(len(train_loader) / 20), 1)
best_val_loss = resumed_val_loss if resumed_val_loss is not None else float('inf')
history = []
stopped_for_deadline = False


def make_ckpt(epoch_num, val_loss):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch_num, "val_loss": val_loss}


for epoch in range(start_epoch, cfig['num_epochs']):
    if args.deadline_epoch_seconds is not None and time.time() >= args.deadline_epoch_seconds:
        print(f"\n[deadline] {time.time():.0f} >= {args.deadline_epoch_seconds:.0f} -- "
              f"stopping before epoch {epoch+1} (train-to-time, not train-to-epoch)")
        stopped_for_deadline = True
        break
    model.train()
    epoch_loss = 0
    for batch_idx, data_dict in enumerate(train_loader):
        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(data_dict['data'].to(device))

            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone())

            outputs = outputs * cfig['scale_out']

            loss = criterion(outputs, data_dict['label'].to(device))

        # Backward pass and optimization
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()
        
        if batch_idx % nbatch_per_log == 0:
        
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{cfig['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

        if batch_idx == 3 and epoch % 50 == 0:
            os.system('nvidia-smi')

        # Mid-epoch checkpoint: a single epoch here can take well over an
        # hour (heavy per-item 3D preprocessing + a big model), which is
        # longer than a free-tier Colab session reliably survives -- without
        # this, a disconnect mid-epoch loses every gradient update made so
        # far in that epoch, since save only otherwise happens at epoch end.
        checkpoint_every_batches = cfig.get('checkpoint_every_batches', 20)
        if checkpoint_every_batches and (batch_idx + 1) % checkpoint_every_batches == 0:
            os.makedirs(cfig['save_model_root'], exist_ok=True)
            torch.save(make_ckpt(epoch, best_val_loss), os.path.join(cfig['save_model_root'], 'mid_epoch_model.pth'))
            print(f"  [mid-epoch checkpoint @ batch {batch_idx+1}]")

    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)

    # Validation pass -- val_loader was built above but never previously used
    # (train/val split existed in the config but had no effect on training).
    model.eval()
    val_loss = 0.0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for data_dict in val_loader:
            outputs = model(data_dict['data'].to(device))
            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone())
            outputs = outputs * cfig['scale_out']
            val_loss += criterion(outputs, data_dict['label'].to(device)).item()
    avg_val_loss = val_loss / max(len(val_loader), 1)

    print(f"Epoch [{epoch+1}/{cfig['num_epochs']}] Completed: Avg Train Loss: {avg_epoch_loss:.4f}, "
          f"Avg Val Loss: {avg_val_loss:.4f}")

    os.makedirs(cfig['save_model_root'], exist_ok=True)
    torch.save(make_ckpt(epoch + 1, avg_val_loss), os.path.join(cfig['save_model_root'], 'last_model.pth'))
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(make_ckpt(epoch + 1, best_val_loss), os.path.join(cfig['save_model_root'], 'best_model.pth'))

    history.append({"epoch": epoch + 1, "train_loss": avg_epoch_loss, "val_loss": avg_val_loss})
    with open(os.path.join(cfig['save_model_root'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

status = "stopped at deadline" if stopped_for_deadline else "completed all epochs"
print(f"\nTraining {status}. Epochs completed: {len(history)}. Best val loss: {best_val_loss:.4f}")

# --- Loss curve (presentation artifact) ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs_x = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_x, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs_x, [h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("L1 loss"); plt.title("mednext loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfig['save_model_root'], 'loss_curve.png'), dpi=120)
    plt.close()
    print(f"Loss curve saved: {os.path.join(cfig['save_model_root'], 'loss_curve.png')}")
except Exception as e:
    print(f"[loss curve skipped] {e}")

# --- Final test-set eval using the BEST checkpoint (by val loss), not last ---
best_path = os.path.join(cfig['save_model_root'], 'best_model.pth')
if os.path.exists(best_path):
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model"] if isinstance(best_ckpt, dict) and "model" in best_ckpt else best_ckpt)
    print(f"Loaded best checkpoint for final eval: {best_path}")
# else: no training happened this run (e.g. an eval-only rerun) -- `model`
# already holds whatever --resume loaded above.
model.eval()

# NOTE: the official GDP-HMM meta_data.csv has train/valid/test dev_split
# values, but none of the locally-available files happen to be tagged
# 'test' -- only train/valid exist in this local subset. Re-using val_loader
# here rather than fabricating a test set; final_eval.md/json label this
# honestly as the validation set, not a held-out test set.
#
# Audited 2026-09-13 whether the canonical 50-patient test split
# (data_splits.json, used by dvhnet/dosenet3d) could substitute here for a
# cross-model-comparable number: it can't, cheaply. The official CSV's own
# 'test' bucket (147 patients) has ZERO overlap with the ~499 patients we
# actually downloaded (train_01.zip/train_02.zip) -- that bucket is a
# different cohort slice the organizers never distributed, which is exactly
# why no local file is ever tagged 'test'. Worse: 48 of the canonical 50
# test patients are already sitting in THIS CSV's own 'train' bucket (1 is
# 'valid', 1 is entirely absent) -- so filtering by canonical test-patient
# ID instead of dev_split would evaluate mostly on data this checkpoint was
# already trained on, producing a falsely optimistic number, not a fix.
# Getting a genuine canonical-test number for mednext would require
# retraining from scratch on a corrected split (excluding those 48 patients
# from training) -- a real option if cross-model comparability is ever
# needed, but not attempted here given the cost of a full retrain vs. just
# continuing to train the current, honestly-labeled val-based checkpoint.
test_loader = loaders.val_dataloader()
test_losses = []
with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
    for data_dict in test_loader:
        outputs = model(data_dict['data'].to(device))
        if cfig['act_sig']:
            outputs = torch.sigmoid(outputs.clone())
        outputs = outputs * cfig['scale_out']
        test_losses.append(criterion(outputs, data_dict['label'].to(device)).item())

if test_losses:
    import statistics
    final_summary = {
        "model": "mednext",
        "input_resolution": list(INPUT_RESOLUTION),
        "model_id": cfig['model_params']['model_id'],
        "eval_set": "validation (no locally-available files tagged dev_split=='test')",
        "n_eval_patients": len(test_losses),
        "eval_l1_loss_mean": statistics.mean(test_losses),
        "eval_l1_loss_std": statistics.pstdev(test_losses) if len(test_losses) > 1 else 0.0,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "stopped_for_deadline": stopped_for_deadline,
    }
    with open(os.path.join(cfig['save_model_root'], 'final_eval.json'), 'w') as f:
        json.dump(final_summary, f, indent=2)
    with open(os.path.join(cfig['save_model_root'], 'final_eval.md'), 'w') as f:
        f.write(f"# mednext final evaluation\n\n"
                f"- Input resolution: {INPUT_RESOLUTION} (reduced from 96x128x144 for a free-tier T4)\n"
                f"- Epochs completed: {len(history)} ({'stopped at deadline' if stopped_for_deadline else 'full run'})\n"
                f"- Best val loss: {best_val_loss:.4f}\n"
                f"- Eval set: validation ({len(test_losses)} patients) -- no locally-available files "
                f"are tagged dev_split=='test' in the official meta_data.csv, so this reuses val, not a held-out test set\n"
                f"- Eval L1 loss: {final_summary['eval_l1_loss_mean']:.4f} ± {final_summary['eval_l1_loss_std']:.4f}\n")
    print(f"Final eval written: {os.path.join(cfig['save_model_root'], 'final_eval.json')}")
