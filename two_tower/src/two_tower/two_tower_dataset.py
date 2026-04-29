import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
MAX_HISTORY = 50

class TwoTowerDataset(Dataset):
    def __init__(self, interactions_path, user_history_path, sample_size=None):

        self.df = pd.read_parquet(interactions_path)
        if sample_size is not None and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        with open(user_history_path, "rb") as f:
            self.user_history = pickle.load(f)

        self.df = self.df[
            self.df["user_idx"].map(lambda user: len(self.user_history.get(user, [])) > 1)
        ].reset_index(drop=True)

        self.users = self.df["user_idx"].values
        self.items = self.df["item_idx"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        user = self.users[idx]
        pos_item = self.items[idx]
        history = self.user_history[user]
        history = [(item, hist_rating) for item, hist_rating in history if item != pos_item]
        history_items = [x[0] for x in history]
        history_ratings = [x[1] for x in history]

        history_items = history_items[-MAX_HISTORY:]
        history_ratings = history_ratings[-MAX_HISTORY:]

        pad_len = MAX_HISTORY - len(history_items)

        history_ratings_padded = np.array([0.0] * pad_len + history_ratings, dtype=np.float32)  # (MAX_HISTORY,)

        mask = np.array([0.0] * pad_len + [1.0] * len(history_items), dtype=np.float32)

        history_items_padded = [0] * pad_len + history_items

        return {
            "history_items": torch.tensor(history_items_padded, dtype=torch.long),
            "history_ratings": torch.tensor(history_ratings_padded, dtype=torch.float32),
            "history_mask": torch.tensor(mask, dtype=torch.float32),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
        }
