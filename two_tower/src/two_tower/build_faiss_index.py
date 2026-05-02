import numpy as np
import faiss

ITEM_EMBEDDINGS_PATH = "D:/projects/recoscale/two_tower/models/item_embeddings_inbatch_sampled.npy"
SAVE_PATH = "D:/projects/recoscale/two_tower/models/faiss_index_inbatch_sampled.bin"

item_embeddings = np.load(ITEM_EMBEDDINGS_PATH).astype("float32")

faiss.normalize_L2(item_embeddings)

dimension = 64
base_index = faiss.IndexFlatIP(dimension)
indexes = faiss.IndexIDMap2(base_index)
item_ids = np.arange(item_embeddings.shape[0], dtype=np.int64)
indexes.add_with_ids(item_embeddings, item_ids)

faiss.write_index(indexes, SAVE_PATH)
print("Index saved, total vectors:", indexes.ntotal)
