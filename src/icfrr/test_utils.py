from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch

from src.icfrr.cache_utils import Cacher
from src.icfrr.configs import RunConfigSettings
from src.icfrr.data_utils import DatasetName
from src.icfrr.models.model_cross import VisionTransformer


def load_checkpoint(resume_checkpoint: Path, device: torch.device) -> tuple[dict, RunConfigSettings]:
    loaded_ckpt = torch.load(resume_checkpoint, map_location=device)

    return loaded_ckpt, RunConfigSettings.model_validate(loaded_ckpt["args"])


def get_feats_scores_dists(cache_root_main: Path, device: torch.device, model: VisionTransformer, dataloaders: dict,
                           no_cache: bool = False, force_recache: bool = False):
    cacher = Cacher(cache_root_main)
    if not (force_recache or no_cache) and cache_root_main.exists():
        qg_dists, gg_dists, labels = cacher.load_caches()
    else:
        model = model.to(device)
        labels, features = {}, {}
        features["query"], labels["query"] = get_features(dataloaders["query"], device, model)
        features["gallery"], labels["gallery"] = get_features(dataloaders["gallery"], device, model)
        gg_dists = torch.cdist(features["gallery"], features["gallery"])
        qg_dists = torch.cdist(features["query"], features["gallery"])
        if not no_cache:
            cacher.write_caches(qg_dists, gg_dists, labels)

    return qg_dists, gg_dists, labels


@torch.no_grad()
def get_features(data_loader: DataLoader, device: torch.device, model: VisionTransformer,
                 cast_down: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # switch to evaluate mode
    model.eval()
    features_all = []
    targets_all = []
    num_prefix = model.num_prefix_tokens
    for i, data in enumerate(tqdm(data_loader)):
        input, target, idxs = data
        input = input.to(device)

        all_features, atts, vs = model.forward_features(None, input)
        features = all_features[1]
        features = features[:, num_prefix:, :].mean(1)

        features = torch.nn.functional.normalize(features)
        features = features.cpu().detach()
        if cast_down:
            features = features.to(torch.float16)

        features_all.append(features.reshape(input.size()[0], -1))
        targets_all.append(target.detach())

    features_all = torch.cat(features_all)
    targets_all = torch.cat(targets_all)

    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))

    return features_all, targets_all


def make_model_from_args(state_dict: dict, num_classes: int = 100):
    from src.icfrr.models import model_cross
    model = model_cross.vit_base_patch16_224(num_classes=num_classes, pretrained=True)
    model.head = torch.nn.Identity()
    state_dict = {k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    return model


def get_checkpoint(dataset: DatasetName, resume_dir_main: Path) -> Path:
    if dataset == DatasetName.tuberlin:
        resume_dir = "tuberlin"
    elif dataset == DatasetName.sketchy_zs1:
        resume_dir = "sketchy/zeroshot1"
    elif dataset == DatasetName.sketchy_zs2:
        resume_dir = "sketchy/zeroshot2"
    elif dataset == DatasetName.custom:
        resume_dir = "custom"
    else:
        raise NotImplementedError(f"No resume directory for dataset: {dataset}")
    resume_checkpoint = resume_dir_main / resume_dir / 'checkpoint.pth'
    return resume_checkpoint
