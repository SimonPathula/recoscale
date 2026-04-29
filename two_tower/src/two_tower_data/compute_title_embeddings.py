import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("D:/projects/recoscale/two_tower/data/item_features")
df = df.sort_values("item_idx").reset_index(drop=True)

def compute_title_embeddings(df, chunk_size=500000, batch_size=64, save_dir = "D:/projects/recoscale/two_tower/data/embeddings"):

    os.makedirs(save_dir, exist_ok=True)

    model = SentenceTransformer('all-MiniLM-L6-v2', device= "cuda")

    n = len(df)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f"Print Processing from {start} : {end}")

        chunk_df = df[start:end]
        titles = chunk_df['title'].fillna("").to_list()

        embeddings = model.encode(
            titles,
            batch_size = batch_size,
            device= 'cuda',
            show_progress_bar= True,
            convert_to_numpy= True
        )
        np.save(f"{save_dir}/emb_{start}_{end}.npy", embeddings)
        del embeddings
    print("Done")

compute_title_embeddings(df)