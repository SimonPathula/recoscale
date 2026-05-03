import numpy as np
import pandas as pd
import pickle
import torch
import faiss
import os
import random
from tqdm import tqdm
from two_tower_model import TwoTowerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERACTIONS_TEST = "D:/projects/recoscale/two_tower/data/interactions_test"
USER_HISTORY = "D:/projects/recoscale/two_tower/data/user_history.pkl"
ALL_ITEMS = "D:/projects/recoscale/two_tower/data/all_item_idxs.npy"
CHECKPOINT_DIR = "D:/projects/recoscale/two_tower/models/two_tower"
INDEX_PATH = "D:/projects/recoscale/two_tower/models/faiss_index_fullbatch.bin"

MAX_HISTORY = 50
K = 10
RECALL_KS = [10, 50, 100, 500, 1000]
SAMPLE_USERS = 10000
FETCH_MULTIPLIER = 20
SEED = 42

test = pd.read_parquet(INTERACTIONS_TEST)

with open(USER_HISTORY, "rb") as f:
    user_history = pickle.load(f)

candidate_items = set(np.load(ALL_ITEMS).tolist())

index = faiss.read_index(INDEX_PATH)

model = TwoTowerModel().to(DEVICE)
model.item_features = model.item_features.to(DEVICE)


def latest_checkpoint():
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return os.path.join(CHECKPOINT_DIR, checkpoints[-1])

def get_user_embedding(user_idx):
    history = user_history.get(user_idx, [])
    if len(history) == 0:
        return None, set()

    history_items = [int(x[0]) for x in history][-MAX_HISTORY:]
    history_ratings = [float(x[1]) for x in history][-MAX_HISTORY:]
    seen_items = {int(x[0]) for x in history}

    pad_len = MAX_HISTORY - len(history_items)
    batch = {
        "history_items": torch.tensor([[0] * pad_len + history_items], dtype=torch.long, device=DEVICE),
        "history_ratings": torch.tensor([[0.0] * pad_len + history_ratings], dtype=torch.float32, device=DEVICE),
        "history_mask": torch.tensor([[0.0] * pad_len + [1.0] * len(history_items)], dtype=torch.float32, device=DEVICE),
    }

    with torch.no_grad():
        user_emb = model.encode_user(batch)

    return user_emb.cpu().numpy().astype("float32"), seen_items

def search_unseen(user_emb, seen_items, k=10):
    fetch_k = min(index.ntotal, max(k * FETCH_MULTIPLIER, k + len(seen_items)))

    while True:
        _, indices = index.search(user_emb, fetch_k)
        retrieved = [
            int(item)
            for item in indices[0]
            if item >= 0 and int(item) in candidate_items and int(item) not in seen_items
        ]

        if len(retrieved) >= k or fetch_k >= index.ntotal:
            return retrieved[:k]

        fetch_k = min(index.ntotal, fetch_k * 2)

def compute_ndcg(retrieved, ground_truth, k=10):
    for rank, item in enumerate(retrieved[:k]):
        if item == ground_truth:
            return 1.0 / np.log2(rank + 2)
    return 0.0

print("Sampling users...")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ckpt_path = latest_checkpoint()
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

unique_users = test["user_idx"].drop_duplicates()
sample_size = min(SAMPLE_USERS, len(unique_users))
sample_users = unique_users.sample(sample_size, random_state=SEED)
test_sample = test[test["user_idx"].isin(sample_users)]
eval_rows = test_sample.groupby("user_idx", group_keys=False).sample(n=1, random_state=SEED)

print(f"Evaluating {sample_size} sampled users...")
ndcg_scores = []
recall_hits = {k: 0 for k in RECALL_KS}
skipped_no_history = 0
gt_not_train_candidate = 0
skipped_gt_in_history = 0

for row in tqdm(eval_rows.itertuples(index=False), total=len(eval_rows)):
    user_idx = int(row.user_idx)
    gt_item = int(row.item_idx)
    if gt_item not in candidate_items:
        gt_not_train_candidate += 1
        continue

    user_emb, seen_items = get_user_embedding(user_idx)
    if user_emb is None:
        skipped_no_history += 1
        continue
    if gt_item in seen_items:
        skipped_gt_in_history += 1
        continue

    faiss.normalize_L2(user_emb)
    retrieved = search_unseen(user_emb, seen_items, max(RECALL_KS))

    ndcg = compute_ndcg(retrieved, gt_item, K)
    ndcg_scores.append(ndcg)
    for recall_k in RECALL_KS:
        recall_hits[recall_k] += int(gt_item in retrieved[:recall_k])

if not ndcg_scores:
    raise RuntimeError("No users were evaluated. Check user_history and test/candidate item overlap.")

print()
for recall_k in RECALL_KS:
    print(f"Recall@{recall_k} = {recall_hits[recall_k] / len(ndcg_scores):.4f}")
print(f"NDCG@{K} = {np.mean(ndcg_scores):.4f}")
print(f"Hit Rate@{K} = {sum(s > 0 for s in ndcg_scores) / len(ndcg_scores):.4f}")
print(f"Evaluated users: {len(ndcg_scores)}")
print(f"Skipped users without train history: {skipped_no_history}")
print(f"Evaluated test-only items outside train candidates: {gt_not_train_candidate}")
print(f"Skipped test items already in train history: {skipped_gt_in_history}")
