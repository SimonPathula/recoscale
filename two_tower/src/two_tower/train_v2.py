import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from two_tower_model import TwoTowerModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERACTIONS = "D:/projects/recoscale/two_tower/data/interactions_train"
USER_HISTORY = "D:/projects/recoscale/two_tower/data/user_history.pkl"
ALL_ITEMS = "D:/projects/recoscale/two_tower/data/all_item_idxs.npy"
CHECKPOINT_DIR = "D:/projects/recoscale/two_tower/models/two_tower_hardneg"

SEED = 42
MAX_HISTORY = 50

BATCH_SIZE = 256
NUM_EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
TRAIN_SAMPLE_SIZE = None
RESUME = False

NUM_NEGATIVES = 64
NEGATIVE_POOL_SIZE = 192
POPULAR_POOL_SIZE = 200_000
POPULAR_FRACTION = 0.50

SAMPLED_LOSS_WEIGHT = 1.0
IN_BATCH_LOSS_WEIGHT = 0.25
TEMPERATURE = 0.07
GRAD_CLIP_NORM = 5.0


class HardNegativeTwoTowerDataset(Dataset):
    def __init__(
        self,
        interactions_path,
        user_history_path,
        all_items_path,
        sample_size=None,
        seed=SEED,
    ):
        self.rng = np.random.default_rng(seed)
        self.df = pd.read_parquet(interactions_path)

        if sample_size is not None and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        with open(user_history_path, "rb") as f:
            self.user_history = pickle.load(f)

        self.df = self.df[
            self.df["user_idx"].map(lambda user: len(self.user_history.get(int(user), [])) > 1)
        ].reset_index(drop=True)

        self.users = self.df["user_idx"].astype(np.int64).to_numpy()
        self.items = self.df["item_idx"].astype(np.int64).to_numpy()
        self.all_items = np.load(all_items_path).astype(np.int64)

        item_counts = self.df["item_idx"].value_counts()
        self.popular_items = item_counts.index.to_numpy(dtype=np.int64)[:POPULAR_POOL_SIZE]
        if len(self.popular_items) == 0:
            self.popular_items = self.all_items

    def __len__(self):
        return len(self.df)

    def _sample_candidates(self, source_items, count, blocked):
        selected = []
        attempts = 0

        while len(selected) < count and attempts < 8:
            need = count - len(selected)
            sample_size = max(need * 4, 128)
            sampled = self.rng.choice(source_items, size=sample_size, replace=True)

            for item in sampled:
                item = int(item)
                if item not in blocked:
                    selected.append(item)
                    if len(selected) == count:
                        break

            attempts += 1

        while len(selected) < count:
            item = int(self.all_items[self.rng.integers(0, len(self.all_items))])
            if item not in blocked:
                selected.append(item)

        return selected

    def _negative_pool(self, pos_item, seen_items):
        blocked = set(seen_items)
        blocked.add(int(pos_item))

        popular_count = int(NEGATIVE_POOL_SIZE * POPULAR_FRACTION)
        random_count = NEGATIVE_POOL_SIZE - popular_count

        negatives = []
        negatives.extend(self._sample_candidates(self.popular_items, popular_count, blocked))
        blocked.update(negatives)
        negatives.extend(self._sample_candidates(self.all_items, random_count, blocked))

        self.rng.shuffle(negatives)
        return np.asarray(negatives, dtype=np.int64)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos_item = int(self.items[idx])

        history = self.user_history[user]
        history = [(int(item), float(rating)) for item, rating in history if int(item) != pos_item]

        history_items = [item for item, _ in history][-MAX_HISTORY:]
        history_ratings = [rating for _, rating in history][-MAX_HISTORY:]
        seen_items = [item for item, _ in history]

        pad_len = MAX_HISTORY - len(history_items)
        history_items_padded = [0] * pad_len + history_items
        history_ratings_padded = [0.0] * pad_len + history_ratings
        history_mask = [0.0] * pad_len + [1.0] * len(history_items)

        return {
            "history_items": torch.tensor(history_items_padded, dtype=torch.long),
            "history_ratings": torch.tensor(history_ratings_padded, dtype=torch.float32),
            "history_mask": torch.tensor(history_mask, dtype=torch.float32),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "negative_pool": torch.tensor(self._negative_pool(pos_item, seen_items), dtype=torch.long),
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def sampled_softmax_loss(user_emb, pos_emb, neg_emb, temperature):
    pos_scores = (user_emb * pos_emb).sum(dim=1, keepdim=True)
    neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)
    logits = torch.cat([pos_scores, neg_scores], dim=1) / temperature
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets), logits


def in_batch_loss(user_emb, pos_emb, temperature):
    logits = user_emb @ pos_emb.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets), logits


def select_online_hard_negatives(model, user_emb, negative_pool, num_negatives):
    batch_size, pool_size = negative_pool.shape
    flat_pool = negative_pool.reshape(batch_size * pool_size)

    with torch.no_grad():
        pool_emb = model.encode_items(flat_pool).reshape(batch_size, pool_size, -1)
        scores = torch.bmm(pool_emb, user_emb.unsqueeze(-1)).squeeze(-1)
        hard_idx = scores.topk(k=num_negatives, dim=1).indices

    return torch.gather(negative_pool, dim=1, index=hard_idx)


def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "loss_type": "sampled_softmax_online_hard_negatives_plus_in_batch",
            "num_negatives": NUM_NEGATIVES,
            "negative_pool_size": NEGATIVE_POOL_SIZE,
            "popular_fraction": POPULAR_FRACTION,
            "temperature": TEMPERATURE,
        },
        path,
    )
    print(f"  Checkpoint saved - {path}")


def load_checkpoint_if_needed(model, optimizer, scaler):
    if not RESUME:
        return 1

    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not checkpoints:
        return 1

    latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"Resumed from checkpoint: {latest} (epoch {ckpt['epoch']})")
    return int(ckpt["epoch"]) + 1


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, use_amp):
    model.train()
    total_loss = 0.0
    total_sampled_loss = 0.0
    total_in_batch_loss = 0.0
    total_steps = 0
    t0 = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)

    for step, batch in enumerate(pbar):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            user_emb = model.encode_user(batch)
            pos_emb = model.encode_items(batch["pos_item"])

            hard_neg_ids = select_online_hard_negatives(
                model,
                user_emb.detach(),
                batch["negative_pool"],
                NUM_NEGATIVES,
            )

            neg_emb = model.encode_items(hard_neg_ids.reshape(-1))
            neg_emb = neg_emb.reshape(hard_neg_ids.size(0), hard_neg_ids.size(1), -1)

            sampled_loss, sampled_logits = sampled_softmax_loss(
                user_emb,
                pos_emb,
                neg_emb,
                TEMPERATURE,
            )
            batch_loss, batch_logits = in_batch_loss(user_emb, pos_emb, TEMPERATURE)
            loss = SAMPLED_LOSS_WEIGHT * sampled_loss + IN_BATCH_LOSS_WEIGHT * batch_loss

        if torch.isnan(loss):
            print("NaN loss detected. Stopping training.")
            return total_loss / max(total_steps, 1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_sampled_loss += float(sampled_loss.item())
        total_in_batch_loss += float(batch_loss.item())
        total_steps += 1

        with torch.no_grad():
            sampled_acc = (sampled_logits.argmax(dim=1) == 0).float().mean().item()
            targets = torch.arange(batch_logits.size(0), device=batch_logits.device)
            batch_acc = (batch_logits.argmax(dim=1) == targets).float().mean().item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "sampled": f"{sampled_loss.item():.4f}",
                "s_acc": f"{sampled_acc:.3f}",
                "b_acc": f"{batch_acc:.3f}",
                "step/s": f"{total_steps / (time.time() - t0):.2f}",
            }
        )

        if step % 500 == 0:
            elapsed = time.time() - t0
            pbar.write(
                f"  Epoch {epoch} | Step {step} | Loss {loss.item():.4f} | "
                f"SampledLoss {sampled_loss.item():.4f} | InBatchLoss {batch_loss.item():.4f} | "
                f"SampledAcc {sampled_acc:.4f} | InBatchAcc {batch_acc:.4f} | Elapsed {elapsed:.1f}s"
            )

    return {
        "loss": total_loss / total_steps,
        "sampled_loss": total_sampled_loss / total_steps,
        "in_batch_loss": total_in_batch_loss / total_steps,
    }


def main():
    seed_everything(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    use_amp = DEVICE.type == "cuda"
    print(f"Device: {DEVICE} | AMP: {use_amp}")
    print(f"Training sample size: {TRAIN_SAMPLE_SIZE}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(
        f"Negative mining: pool={NEGATIVE_POOL_SIZE}, selected={NUM_NEGATIVES}, "
        f"popular_fraction={POPULAR_FRACTION}"
    )

    print("Loading dataset...")
    dataset = HardNegativeTwoTowerDataset(
        INTERACTIONS,
        USER_HISTORY,
        ALL_ITEMS,
        sample_size=TRAIN_SAMPLE_SIZE,
        seed=SEED,
    )
    print(f"Dataset size after filters: {len(dataset):,} interactions")
    print(f"Train candidate items: {len(dataset.all_items):,}")
    print(f"Popular negative pool: {len(dataset.popular_items):,}")

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

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = load_checkpoint_if_needed(model, optimizer, scaler)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        metrics = train_one_epoch(model, loader, optimizer, scaler, DEVICE, epoch, use_amp)

        if metrics is None:
            break

        print(
            f"  Epoch {epoch} complete | Avg Loss: {metrics['loss']:.4f} | "
            f"Sampled Loss: {metrics['sampled_loss']:.4f} | "
            f"InBatch Loss: {metrics['in_batch_loss']:.4f}"
        )

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pt")
        save_checkpoint(model, optimizer, scaler, epoch, metrics, ckpt_path)


if __name__ == "__main__":
    main()