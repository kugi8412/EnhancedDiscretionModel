"""Compare features between two trained SAEs via cosine similarity."""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from .model import SparseAutoencoder, TopKSparseAutoencoder

def main():
    ap = argparse.ArgumentParser(description="Compare dictionaries of two SAEs.")
    ap.add_argument("--sae_a", required=True, help="Path to first SAE checkpoint")
    ap.add_argument("--sae_b", required=True, help="Path to second SAE checkpoint")
    ap.add_argument("--output_dir", required=True, help="Directory to save results")
    ap.add_argument("--input_dim", type=int, default=256)
    ap.add_argument("--dict_size", type=int, default=512)
    ap.add_argument("--topk_sae", type=int, default=16, help="TopK value if using TopK SAE")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Załaduj model A
    if args.topk_sae > 0:
        model_a = TopKSparseAutoencoder(args.input_dim, args.dict_size, k=args.topk_sae)
    else:
        model_a = SparseAutoencoder(args.input_dim, args.dict_size)
    model_a.load_state_dict(torch.load(args.sae_a, map_location=device, weights_only=True))
    model_a = model_a.to(device).eval()

    # 2. Załaduj model B
    if args.topk_sae > 0:
        model_b = TopKSparseAutoencoder(args.input_dim, args.dict_size, k=args.topk_sae)
    else:
        model_b = SparseAutoencoder(args.input_dim, args.dict_size)
    model_b.load_state_dict(torch.load(args.sae_b, map_location=device, weights_only=True))
    model_b = model_b.to(device).eval()

    # 3. Wyciągnij wagi dekodera i je znormalizuj
    W_a = F.normalize(model_a.decoder.weight, dim=0) # [input_dim, dict_size]
    W_b = F.normalize(model_b.decoder.weight, dim=0)

    # 4. Oblicz macierz podobieństwa kosinusowego
    sim_matrix = (W_a.T @ W_b).cpu().detach().numpy() # [dict_size, dict_size]

    # 5. Znajdź najlepsze pary (Greedy matching)
    best_matches_for_a = np.argmax(sim_matrix, axis=1)
    best_sims_for_a = np.max(sim_matrix, axis=1)

    # Zapisz do CSV
    df = pd.DataFrame({
        "Feature_A": np.arange(args.dict_size),
        "Best_Match_Feature_B": best_matches_for_a,
        "Cosine_Similarity": best_sims_for_a
    })
    df = df.sort_values("Cosine_Similarity", ascending=False)
    csv_path = os.path.join(args.output_dir, "dictionary_matching.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Cross-SAE] Saved matching to {csv_path}")

    # 6. Narysuj histogram podobieństw
    plt.figure(figsize=(8, 5))
    sns.histplot(best_sims_for_a, bins=40, color="purple", kde=True)
    plt.title("Max Cosine Similarity Distribution between SAEs")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "similarity_distribution.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
