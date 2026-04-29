import os
import torch
import numpy as np
from tqdm import tqdm

from two_tower.src.two_tower.two_tower_model import TwoTowerModel

CHUNK_SIZE = 256
TOTAL_ITEMS = 7966869
CHECKPOINT_DIR = "D:/projects/recoscale/models/two_tower_inbatch_sampled"
SAVE_PATH = "D:/projects/recoscale/models/item_embeddings_inbatch_sampled"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
ckpt = torch.load(latest, map_location = DEVICE)   

title_embeddings = np.memmap(
    "D:/projects/recoscale/data/export_data/title_embeddings.dat",
    dtype= "float32",
    mode= "r",
    shape= (7966869, 384)
)

item_features = torch.from_numpy(
        np.load("D:/projects/recoscale/data/export_data/item_features_clean.npy")
    ).float()

def get_item_vecs(item_ids):
    item_ids_np = item_ids.detach().cpu().numpy()
    title = torch.from_numpy(title_embeddings[item_ids_np]).to(item_ids.device)
    feat = item_features[item_ids_np].to(item_ids.device)
    return torch.cat([title, feat], dim=-1)

def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

def generate_embeddings(model):
    all_embeddings = []

    for start in tqdm(range(0, TOTAL_ITEMS, CHUNK_SIZE), desc= "Generate embeddings"):
        end = min(start + CHUNK_SIZE, TOTAL_ITEMS)
        item_ids = torch.arange(start, end, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            emb = model.encode_items(item_ids)

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis= 0)

def main():
    model = TwoTowerModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Generating item embeddings...")
    embeddings = generate_embeddings(model)

    print(f"Final shape: {embeddings.shape}")
    assert embeddings.shape == (TOTAL_ITEMS, 64), "Shape mismatch!"

    np.save(SAVE_PATH, embeddings)


if __name__ == "__main__":
    main()
