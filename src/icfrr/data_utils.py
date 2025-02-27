from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.icfrr.dataloaders.custom import CustomTestDataset
from src.icfrr.dataloaders.officeHome import OfficeHomeDataset
from src.icfrr.dataloaders.sbir import SketchyDataset, TUBerlinDataset
from src.icfrr.utils import seed_worker


class DatasetName(Enum):
    tuberlin = "tuberlin"
    sketchy_zs1 = "sketchy_zs1"
    sketchy_zs2 = "sketchy_zs2"
    custom = "custom"


def get_datasets(data_root_main: Path, dataset: DatasetName, transforms: Optional[Compose] = None,
                 no_augs: bool = True, num_per_class: int = -1) -> tuple[int, dict]:
    if dataset == DatasetName.sketchy_zs1 or dataset == DatasetName.sketchy_zs2:
        zero_version = 'zeroshot1' if dataset == DatasetName.sketchy_zs1 else 'zeroshot2'
        num_classes = 104 if zero_version == 'zeroshot2' else 100
        data_root = data_root_main / "Sketchy"
        query_train = SketchyDataset(root_dir=data_root, split='train', zero_version=zero_version,
                                     transform=transforms, aug=False if no_augs else 'sketch',
                                     first_n_debug=num_per_class)
        gallery_train = SketchyDataset(root_dir=data_root, split='train', version='all_photo',
                                       zero_version=zero_version, transform=transforms,
                                       aug=None if no_augs else 'img', first_n_debug=num_per_class)
        gallery_test_zero = SketchyDataset(root_dir=data_root, split='zero', version='all_photo',
                                           zero_version=zero_version,
                                           transform=transforms, aug=False, first_n_debug=num_per_class)
        query_test_zero = SketchyDataset(root_dir=data_root, split='zero', zero_version=zero_version,
                                         transform=transforms, aug=False, first_n_debug=num_per_class)
    elif dataset == DatasetName.tuberlin:
        num_classes = 220
        data_root = data_root_main / "TUBerlin"
        query_train = TUBerlinDataset(root_dir=data_root, split='train', transform=transforms,
                                      aug=None if no_augs else 'sketch', first_n_debug=num_per_class)
        gallery_train = TUBerlinDataset(root_dir=data_root, split='train', version='ImageResized_ready',
                                        transform=transforms, aug=None if no_augs else 'img',
                                        first_n_debug=num_per_class)
        gallery_test_zero = TUBerlinDataset(root_dir=data_root, split='zero', version='ImageResized_ready',
                                            transform=transforms, aug=False, first_n_debug=num_per_class)
        query_test_zero = TUBerlinDataset(root_dir=data_root, split='zero',
                                          transform=transforms, aug=False, first_n_debug=num_per_class)
    elif dataset == DatasetName.custom:
        num_classes = -1
        query_train, gallery_train = None, None
        data_root = data_root_main / "custom"
        query_test_zero = CustomTestDataset(root_dir=data_root, split="query", transform=transforms)
        gallery_test_zero = CustomTestDataset(root_dir=data_root, split="gallery", transform=transforms)
    else:
        raise NotImplementedError(f"No compatibility for dataset: {dataset}")

    datasets = {"train": {"query": query_train, "gallery": gallery_train},
                "test": {"query": query_test_zero, "gallery": gallery_test_zero}}

    return num_classes, datasets


def get_dataloaders_test(test_datasets: Dict, batchsize: int, num_workers: int = 8) -> dict:
    dataloaders = {}
    for k in test_datasets:
        dataloaders[k] = DataLoader(dataset=test_datasets[k], batch_size=batchsize, shuffle=False,
                                    num_workers=num_workers, worker_init_fn=seed_worker)

    return dataloaders
