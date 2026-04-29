import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from two_tower.src.two_tower.two_tower_model import TwoTowerModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERACTIONS = "D:/projects/recoscale/data/export_data/interactions_test"
USER_HISTORY = "D:/projects/recoscale/data/export_data/user_history.pkl"
ALL_ITEMS = "D:/projects/recoscale/data/export_data/all_item_idxs.npy"
CHECKPOINT_DIR = "D:/projects/recoscale/models/two_tower_inbatch_sampled"

MAX_HISTORY = 50
K = 10
SAMPLE_USERS = 10000
NUM_NEGATIVES = 1000
SEED = 42


test = pd.read_parquet(INTERACTIONS)

with open(USER_HISTORY, "rb") as f:
    user_history = pickle.load(f)

all_items = np.load(ALL_ITEMS)

def latest_checkpoint():
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return os.path.join(CHECKPOINT_DIR, checkpoints[-1])


def get_user_embedding(model, user_idx):
    history = user_history.get(user_idx, [])
    if not history:
        return None, set()

    seen_items = {int(item) for item, _ in history}
    history_items = [int(x[0]) for x in history][-MAX_HISTORY:]
    history_ratings = [float(x[1]) for x in history][-MAX_HISTORY:]

    pad_len = MAX_HISTORY - len(history_items)
    batch = {
        "history_items": torch.tensor([[0] * pad_len + history_items], dtype=torch.long, device=DEVICE),
        "history_ratings": torch.tensor([[0.0] * pad_len + history_ratings], dtype=torch.float32, device=DEVICE),
        "history_mask": torch.tensor([[0.0] * pad_len + [1.0] * len(history_items)], dtype=torch.float32, device=DEVICE),
    }

    with torch.no_grad():
        return model.encode_user(batch), seen_items


def sample_negatives(gt_item, seen_items):
    negatives = []
    blocked = set(seen_items)
    blocked.add(gt_item)

    while len(negatives) < NUM_NEGATIVES:
        item = int(random.choice(all_items))
        if item in blocked:
            continue
        blocked.add(item)
        negatives.append(item)

    return negatives


def compute_ndcg(retrieved, ground_truth, k=10):
    for rank, item in enumerate(retrieved[:k]):
        if item == ground_truth:
            return 1.0 / np.log2(rank + 2)
    return 0.0


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ckpt_path = latest_checkpoint()
    model = TwoTowerModel().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.item_features = model.item_features.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Sampled candidates per user: 1 positive + {NUM_NEGATIVES} negatives")

    unique_users = test["user_idx"].drop_duplicates()
    sample_size = min(SAMPLE_USERS, len(unique_users))
    sample_users = unique_users.sample(sample_size, random_state=SEED)
    test_sample = test[test["user_idx"].isin(sample_users)]

    ndcg_scores = []
    hit_count = 0
    skipped_no_history = 0
    skipped_gt_in_history = 0

    for row in tqdm(test_sample.itertuples(index=False), total=len(test_sample)):
        user_idx = int(row.user_idx)
        gt_item = int(row.item_idx)

        user_emb, seen_items = get_user_embedding(model, user_idx)
        if user_emb is None:
            skipped_no_history += 1
            continue
        if gt_item in seen_items:
            skipped_gt_in_history += 1
            continue

        candidate_ids = [gt_item] + sample_negatives(gt_item, seen_items)
        candidate_ids_np = np.array(candidate_ids, dtype=np.int64)

        with torch.no_grad():
            item_ids = torch.from_numpy(candidate_ids_np).to(DEVICE)
            item_embs = model.encode_items(item_ids)
            scores = (user_emb * item_embs).sum(dim=1).detach().cpu().numpy()

        order = np.argsort(-scores)
        retrieved = [candidate_ids[i] for i in order[:K]]
        ndcg = compute_ndcg(retrieved, gt_item, K)

        ndcg_scores.append(ndcg)
        hit_count += int(ndcg > 0)

    if not ndcg_scores:
        raise RuntimeError("No users were evaluated.")

    print(f"\nSampled NDCG@{K} = {np.mean(ndcg_scores):.8f}")
    print(f"Sampled Hit Rate@{K} = {hit_count / len(ndcg_scores):.8f}")
    print(f"Evaluated users: {len(ndcg_scores)}")
    print(f"Skipped users without train history: {skipped_no_history}")
    print(f"Skipped test items already in train history: {skipped_gt_in_history}")


if __name__ == "__main__":
    main()
