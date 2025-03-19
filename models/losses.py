import torch
import torch.nn as nn


def contrastive_loss(anchor_emb, positive_emb, negative_embs, temperature=0.07):
    batch_size = anchor_emb.size(0)
    criterion = nn.CrossEntropyLoss()

    # Normalize embeddings
    anchor_emb = nn.functional.normalize(anchor_emb, dim=1)
    positive_emb = nn.functional.normalize(positive_emb, dim=1)
    negative_embs = nn.functional.normalize(negative_embs, dim=2)

    pos_sim = torch.bmm(anchor_emb.unsqueeze(1), positive_emb.unsqueeze(2)).squeeze(-1)  # [batch_size, 1]
    neg_sim = torch.bmm(negative_embs, anchor_emb.unsqueeze(2)).squeeze(-1)  # [batch_size, n_neg]

    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # [batch_size, 1 + n_neg]
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_emb.device)  # correct idx is 0
    return criterion(logits, labels)
