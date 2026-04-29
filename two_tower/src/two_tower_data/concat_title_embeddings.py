import numpy as np
import os

save_dir = "D:/projects/recoscale/two_tower/data/embeddings"
output_path = "D:/projects/recoscale/two_tower/data/title_embeddings.dat"

files = sorted([f for f in os.listdir(save_dir) if f.startswith("emb_")])

#get total size + embedding dim
total_rows = 0
embedding_dim = None

for f in files:
    arr = np.load(os.path.join(save_dir, f))
    total_rows += arr.shape[0]
    if embedding_dim is None:
        embedding_dim = arr.shape[1]
    del arr 

print(f"Total rows: {total_rows}, dim: {embedding_dim}")

#create memmap
memmap = np.memmap(
    output_path,
    dtype="float32",
    mode="w+",
    shape=(total_rows, embedding_dim)
)

#fill memmap chunk by chunk
start = 0

for f in files:
    print(f"Processing {f}")

    chunk = np.load(os.path.join(save_dir, f))
    end = start + chunk.shape[0]

    memmap[start:end] = chunk

    start = end
    del chunk

#flush to disk
memmap.flush()

print("Done")