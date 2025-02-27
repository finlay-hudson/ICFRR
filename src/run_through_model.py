from argparse import Namespace

import torch

from src.icfrr.configs import get_model_run_args
from src.icfrr.data_utils import get_dataloaders_test, get_datasets, DatasetName
from src.icfrr.metric_utils import calc_metrics, get_pos_matches
from src.icfrr.test_utils import get_feats_scores_dists, make_model_from_args, load_checkpoint, get_checkpoint


def main(args: Namespace):
    dataset = DatasetName(args.dataset)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    loaded_ckpt, _ = load_checkpoint(get_checkpoint(dataset, args.pretrained_root), device)

    num_classes, datasets = get_datasets(args.data_root, dataset, loaded_ckpt["data_transform"],
                                         num_per_class=args.per_class)
    dataloaders_test = get_dataloaders_test(datasets["test"], args.batchsize)

    model = make_model_from_args(loaded_ckpt["model_state_dict_best"], num_classes).to(device).eval()

    qg_dists, gg_dists, labels = get_feats_scores_dists(args.cache_root / args.dataset / str(args.per_class),
                                                        device, model, dataloaders_test, args.no_cache,
                                                        args.force_cache)

    if labels is not None and not (labels["query"] == -1).all():
        # Calculate the original metrics before any reranking takes place
        orig_mAP, orig_mAP_200, orig_prec_25, orig_prec_100, orig_prec_200 = calc_metrics(
            get_pos_matches(labels["gallery"], labels["query"]), torch.argsort(qg_dists), -qg_dists, device,
            flags_already_sorted=False)
        print(
            f'Origional mAP@all: {orig_mAP:.5f}, map@200: {orig_mAP_200:.5f}, prec@25: {orig_prec_25:.5f}, prec@100: '
            f'{orig_prec_100:.5f}, prec@200: {orig_prec_200:.5f}')


if __name__ == '__main__':
    main(get_model_run_args())
