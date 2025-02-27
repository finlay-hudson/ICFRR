from typing import Optional

import torch
from torchmetrics.functional import retrieval_average_precision


def aps(pos_flags: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    aps = []
    for i in range(pos_flags.shape[0]):
        ap = retrieval_average_precision(scores[i], pos_flags[i])
        aps.append(ap)
    return torch.stack(aps).detach().cpu()


def apsak(pos_flags: torch.Tensor, sorted_idx: torch.Tensor, scores: torch.Tensor, device: torch.device,
          k: Optional[int] = None) -> torch.Tensor:
    if k is not None:
        if sorted_idx.ndim > 1:
            sorted_idx = sorted_idx[:, :k]
        else:
            sorted_idx = sorted_idx[:k]

    if sorted_idx.ndim > 1:
        sorted_scores = torch.stack([scores[i, id] for i, id in enumerate(sorted_idx)])
        sorted_pos_flags = torch.stack([pos_flags[i, id] for i, id in enumerate(sorted_idx)])
    else:
        sorted_scores = torch.stack([scores[id] for id in sorted_idx])
        sorted_pos_flags = torch.stack([pos_flags[id] for id in sorted_idx])
        sorted_scores = sorted_scores.unsqueeze(0)
        sorted_pos_flags = sorted_pos_flags.unsqueeze(0)

    aps_ = aps(sorted_pos_flags.to(device), sorted_scores.to(device))

    return aps_


def calc_metrics(pos_flags: torch.Tensor, sort_idx: torch.Tensor, scores: torch.Tensor, device: torch.device,
                 flags_already_sorted: bool = True) -> tuple[float, float, float, float, float]:
    if flags_already_sorted:
        aps_ = aps(pos_flags.to(device), scores.to(device))
    else:
        aps_ = apsak(pos_flags.clone(), sort_idx.clone(), scores.clone(), device)
    mAP_all = aps_.mean()

    if flags_already_sorted:
        aps_ = aps(pos_flags[:, :200].to(device), scores[:, :200].to(device))
    else:
        aps_ = apsak(pos_flags.clone(), sort_idx.clone(), scores.clone(), device, k=200)
    mAP_200 = aps_.mean()

    if flags_already_sorted:
        sorted_pos_flags = pos_flags
    else:
        sorted_pos_flags = torch.stack([pos_flags[i, idx] for i, idx in enumerate(sort_idx)])
    prec_25 = (sorted_pos_flags[:, :25].sum(1) / 25).mean()
    prec_100 = (sorted_pos_flags[:, :100].sum(1) / 100).mean()
    prec_200 = (sorted_pos_flags[:, :200].sum(1) / 200).mean()

    return mAP_all.item(), mAP_200.item(), prec_25.item(), prec_100.item(), prec_200.item()


def get_pos_matches(gt_labels_gallery: torch.Tensor, gt_labels_query: torch.Tensor,
                    reorder_gallery_by: Optional[torch.Tensor] = None):
    all_reranked_pos_matches = torch.zeros([len(gt_labels_query), len(gt_labels_gallery)], dtype=bool)
    for q in range(len(gt_labels_query)):
        if reorder_gallery_by is not None:
            all_reranked_pos_matches[q] = gt_labels_query[q] == gt_labels_gallery[reorder_gallery_by[q]]
        else:
            all_reranked_pos_matches[q] = gt_labels_query[q] == gt_labels_gallery

    return all_reranked_pos_matches
