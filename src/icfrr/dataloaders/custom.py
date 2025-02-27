import json
from pathlib import Path
from typing import Optional

from torchvision.transforms import Compose

from src.icfrr.dataloaders.sbir import read_image


class CustomTestDataset:
    def __init__(self, root_dir: Path | str = 'custom', split: str = "query", transform: Optional[Compose] = None):
        self.root_dir = Path(root_dir) / split
        self.file_ls = [fn.name for fn in list((self.root_dir).glob("*"))]
        self.labels = None
        if (root_dir / "labels.json").exists():
            with open(root_dir / "labels.json") as f:
                labels = json.load(f)[split]
            self.labels = []
            for fn in self.file_ls:
                assert fn in labels, f"Missing a label for {fn}"
                self.labels.append(labels[fn])
        self.transform = transform

    def __len__(self):
        return len(self.file_ls)

    def __getitem__(self, idx: int):
        img = read_image(self.root_dir / self.file_ls[idx])

        if self.transform is not None:
            img = self.transform(img)

        label = -1
        if self.labels is not None:
            label = self.labels[idx]

        return img, label, idx
