import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

TITLE_EMBEDDINGS_PATH = "D:/projects/recoscale/two_tower/data/title_embeddings.dat"
ITEM_FEATURES_PATH = "D:/projects/recoscale/two_tower/data/item_features_clean.npy"
TITLE_DIM = 384

def in_batch_softmax_loss(logits):
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(388, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        return self.net(x)


class UserTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        return self.net(x)


class TwoTowerModel(nn.Module):
    def __init__(self, temperature=0.07, title_chunk_rows=250_000, max_open_title_chunks=4):
        super().__init__()
        self.item_tower = ItemTower()
        self.user_tower = UserTower()
        self.temperature = temperature
        self.title_chunk_rows = title_chunk_rows
        self.max_open_title_chunks = max_open_title_chunks
        self._title_chunk_cache = OrderedDict()

        item_features = np.load(ITEM_FEATURES_PATH)
        self.total_items = item_features.shape[0]
        self._validate_title_embeddings()

        self.item_features = torch.from_numpy(item_features).float()

    def _validate_title_embeddings(self):
        row_bytes = TITLE_DIM * np.dtype("float32").itemsize
        file_bytes = os.path.getsize(TITLE_EMBEDDINGS_PATH)
        if file_bytes % row_bytes != 0:
            raise ValueError(
                f"{TITLE_EMBEDDINGS_PATH} has {file_bytes} bytes, which is not "
                f"a whole number of {TITLE_DIM}-dim float32 rows."
            )

        title_rows = file_bytes // row_bytes
        if title_rows != self.total_items:
            raise ValueError(
                f"title embeddings contain {title_rows:,} rows, but item features "
                f"contain {self.total_items:,} rows."
            )

    def _title_chunk(self, chunk_id):
        cached = self._title_chunk_cache.get(chunk_id)
        if cached is not None:
            self._title_chunk_cache.move_to_end(chunk_id)
            return cached

        start = chunk_id * self.title_chunk_rows
        rows = min(self.title_chunk_rows, self.total_items - start)
        offset_bytes = start * TITLE_DIM * np.dtype("float32").itemsize

        chunk = np.memmap(
            TITLE_EMBEDDINGS_PATH,
            dtype="float32",
            mode="r",
            offset=offset_bytes,
            shape=(rows, TITLE_DIM),
        )
        self._title_chunk_cache[chunk_id] = chunk

        if len(self._title_chunk_cache) > self.max_open_title_chunks:
            self._title_chunk_cache.popitem(last=False)

        return chunk

    def get_item_vecs(self, item_ids):
        item_ids_np = item_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        if item_ids_np.size:
            min_item = item_ids_np.min()
            max_item = item_ids_np.max()
            if min_item < 0 or max_item >= self.total_items:
                raise IndexError(
                    f"item_ids must be in [0, {self.total_items - 1}], got "
                    f"min={min_item}, max={max_item}"
                )

        titles = np.empty((item_ids_np.shape[0], TITLE_DIM), dtype=np.float32)
        chunk_ids = item_ids_np // self.title_chunk_rows

        for chunk_id in np.unique(chunk_ids):
            mask = chunk_ids == chunk_id
            local_rows = item_ids_np[mask] - (int(chunk_id) * self.title_chunk_rows)
            titles[mask] = self._title_chunk(int(chunk_id))[local_rows]

        title = torch.from_numpy(titles).to(item_ids.device)

        feature_ids = item_ids.to(self.item_features.device)
        feat = self.item_features[feature_ids].to(item_ids.device)

        return torch.cat([title, feat], dim=-1)

    def encode_items(self, item_ids):
        item_vecs = self.get_item_vecs(item_ids)
        item_embs = self.item_tower(item_vecs)
        return F.normalize(item_embs, dim=-1)

    def encode_user(self, batch):
        history_items = batch["history_items"]
        batch_size, history_len = history_items.shape
        history_flat = history_items.reshape(batch_size * history_len)

        history_vecs = self.get_item_vecs(history_flat)
        history_embs = self.item_tower(history_vecs).reshape(batch_size, history_len, 64)
        history_embs = F.normalize(history_embs, dim=-1)

        mask = batch["history_mask"]
        weights = batch["history_ratings"] * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        pooled = (weights.unsqueeze(-1) * history_embs).sum(dim=1)
        user_emb = self.user_tower(pooled)

        return F.normalize(user_emb, dim=-1)

    def forward(self, batch):
        user_emb = self.encode_user(batch)
        pos_emb = self.encode_items(batch["pos_item"])

        # Each row's positive item is the diagonal; other batch positives are negatives.
        return user_emb @ pos_emb.T / self.temperature
