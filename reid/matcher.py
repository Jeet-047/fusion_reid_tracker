import torch
import torch.nn.functional as F

def cosine_distance(a, b):
    # a: [N, D], b: [M, D]
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return 1 - F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)

def match_embeddings(current_embs, id_manager, threshold=0.4):
    if len(id_manager.id_map) == 0:
        return [-1] * len(current_embs)
    memory = torch.stack(list(id_manager.id_map.values()))
    ids = list(id_manager.id_map.keys())
    dist = cosine_distance(current_embs, memory)
    matched = []
    for row in dist:
        min_val, min_idx = row.min(0)
        matched.append(ids[min_idx] if min_val < threshold else -1)
    return matched