import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_tower_dataset import TwoTowerDataset
from two_tower_model import TwoTowerModel, in_batch_softmax_loss


BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 0
TRAIN_SAMPLE_SIZE = None
RESUME = True

CHECKPOINT_DIR = "D:/projects/recoscale/two_tower/models/two_tower"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERACTIONS = "D:/projects/recoscale/two_tower/data/interactions_train"
USER_HISTORY = "D:/projects/recoscale/two_tower/data/user_history.pkl"

def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "loss_type": "in_batch_softmax",
        },
        path,
    )
    print(f"  Checkpoint saved - {path}")


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, use_amp):
    model.train()
    total_loss = 0.0
    total_steps = 0
    t0 = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)

    for step, batch in enumerate(pbar):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch)
            loss = in_batch_softmax_loss(logits)

        if torch.isnan(loss):
            print("NaN loss detected. Stopping training.")
            return total_loss / max(total_steps, 1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_steps += 1

        with torch.no_grad():
            targets = torch.arange(logits.size(0), device=logits.device)
            batch_acc = (logits.argmax(dim=1) == targets).float().mean().item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "batch_acc": f"{batch_acc:.3f}",
                "step/s": f"{total_steps / (time.time() - t0):.2f}",
            }
        )

        if step % 500 == 0:
            elapsed = time.time() - t0
            pbar.write(
                f"  Epoch {epoch} | Step {step} | Loss {loss.item():.4f} | "
                f"BatchAcc {batch_acc:.4f} | Elapsed {elapsed:.1f}s"
            )

    return total_loss / total_steps


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    use_amp = DEVICE.type == "cuda"
    print(f"Device: {DEVICE} | AMP: {use_amp}")
    print(f"Training sample size: {TRAIN_SAMPLE_SIZE}")

    print("Loading dataset...")
    dataset = TwoTowerDataset(
        INTERACTIONS,
        USER_HISTORY,
        sample_size=TRAIN_SAMPLE_SIZE,
    )
    print(f"Dataset size after filters: {len(dataset):,} interactions")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_amp,
        persistent_workers=False,
        drop_last=True,
    )

    model = TwoTowerModel().to(DEVICE)
    model.item_features = model.item_features.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if RESUME and checkpoints:
        latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint: {latest} (epoch {ckpt['epoch']})")
    elif checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s), but RESUME=False. Starting from scratch.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        avg_loss = train_one_epoch(model, loader, optimizer, scaler, DEVICE, epoch, use_amp)

        if avg_loss is None:
            break

        print(f"  Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pt")
        save_checkpoint(model, optimizer, scaler, epoch, avg_loss, ckpt_path)


if __name__ == "__main__":
    main()