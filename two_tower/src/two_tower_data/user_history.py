import pickle
import numpy as np
import pandas as pd

# Load all interactions_train chunks
interactions = pd.read_parquet("/mnt/d/projects/recoscale/two_tower/data/interactions_train")

interactions = interactions[interactions['rating'] >= 3]

# Group by user_idx → collect (item_idx, rating) per user
user_history = (
    interactions
    .groupby('user_idx')
    .apply(lambda g: list(zip(g['item_idx'], g['rating']))) #example [[123, 4], [234, 5], [345, 1]] - [[index, rating], [index, rating], ..]
    .to_dict()
)

# Save
with open("/mnt/d/projects/recoscale/two_tower/data/user_history.pkl", "wb") as f:
    pickle.dump(user_history, f)

all_item_idxs = interactions['item_idx'].unique()
np.save("/mnt/d/projects/recoscale/two_tower/data/all_item_idxs.npy", all_item_idxs)

