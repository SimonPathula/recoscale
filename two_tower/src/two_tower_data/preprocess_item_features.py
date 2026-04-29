import numpy as np
import pandas as pd

def preprocess_item_features(df):
    #sort with idem_idx
    df = df.sort_values("item_idx").reset_index(drop=True)
    
    #drop the columns
    df = df.drop(['store_encoded', 'parent_asin', 'title'], axis= 1)

    #normalization with log1p
    df["price"] = np.log1p(df["price"])
    df["rating_number"] = np.log1p(df["rating_number"])

    #scale raing
    df["average_rating"] = df["average_rating"] / 5.0

    df = df[[
    "price",
    "average_rating",
    "rating_number",
    "main_category" 
    ]]

    features = df.to_numpy(dtype=np.float32)

    print("First item_idx:", df.index[0])
    print("Last item_idx:", df.index[-1])
    print("Shape:", features.shape)
    print("Sample:", features[0])

    assert not np.isnan(features).any(), "NaNs found!"
    assert not np.isinf(features).any(), "Inf found!"

    np.save(
    "/mnt/d/projects/recoscale/two_tower/data/item_features_clean.npy",
    features
    )

if __name__ == "__main__":
    df = pd.read_parquet("/mnt/d/projects/recoscale/two_tower/data/item_features")
    preprocess_item_features(df)
    print("Done")
