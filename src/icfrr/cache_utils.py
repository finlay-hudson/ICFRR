from pathlib import Path
from typing import Optional

import torch


class Cacher:
    def __init__(self, cache_root: Path):
        self.cache_root_main = cache_root

    def load_caches(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("Loading from cache...")
        qg_dists = torch.load(
            self.cache_root_main / "qg_dists.pth")
        gg_dists = torch.load(
            self.cache_root_main / "gg_dists.pth")
        labels = None
        if (self.cache_root_main / "labels.pth").exists():
            labels = torch.load(self.cache_root_main / "labels.pth")
        print("Cache loaded")

        return qg_dists, gg_dists, labels

    def write_caches(self, qg_dists: torch.Tensor, gg_dists: torch.Tensor, labels: Optional[dict[str, torch.Tensor]]):
        if not self.cache_root_main.exists():
            self.cache_root_main.mkdir(parents=True, exist_ok=True)
        torch.save(qg_dists, self.cache_root_main / "qg_dists.pth")
        torch.save(gg_dists, self.cache_root_main / "gg_dists.pth")
        if labels is not None:
            torch.save({"query": labels["query"], "gallery": labels["gallery"]}, self.cache_root_main / "labels.pth")
