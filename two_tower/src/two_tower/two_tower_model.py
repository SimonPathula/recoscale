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
    def __init__(self, temperature=0.07):
        super().__init__()
        self.item_tower = ItemTower()
        self.user_tower = UserTower()
        self.temperature = temperature

        item_features = np.load(ITEM_FEATURES_PATH)
        total_items = item_features.shape[0]

        self.title_embeddings  = np.memmap(
            TITLE_EMBEDDINGS_PATH,
            dtype="float32",
            mode="r",
            shape=(total_items, TITLE_DIM),
        )

        self.item_features = torch.from_numpy(
            np.load(ITEM_FEATURES_PATH)
        ).float()

    def get_item_vecs(self, item_ids):
        item_ids_np = item_ids.detach().cpu().numpy()

        title = torch.from_numpy(self.title_embeddings [item_ids_np]).to(item_ids.device)
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
