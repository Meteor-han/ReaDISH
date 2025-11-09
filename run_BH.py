import subprocess
import time

scripts = []
for init_lr, min_lr, accumulate_grad_batches in [
    (5e-3, 5e-4, 4), (1e-3, 1e-4, 4), (5e-4, 5e-5, 4),
    (5e-3, 5e-4, 8), (1e-3, 1e-4, 8), (5e-4, 5e-5, 8),
]:
    cmd = [
        "python", "downstream.py",
        "--devices", "0,1,2,3",
        "--check_val_every_n_epoch", "1",
        "--max_epochs", "200",
        "--batch_size", "4",
        "--accumulate_grad_batches", f"{accumulate_grad_batches}",
        "--num_workers", "8",
        "--init_lr", f"{init_lr}",
        "--min_lr", f"{min_lr}",
        "--warmup_lr", "1e-6",
        "--warmup_steps", "500",
        "--ds_name", "BH",
        "--repeat_times", "1",
        "--pred_type", "regression",
        "--init_checkpoint", "/checkpoint/last.ckpt",
        "--norm",
    ]
    scripts.append(cmd)

for script in scripts:
    subprocess.run(script, check=True)
    print(f"one finished.\n")
    time.sleep(100)
