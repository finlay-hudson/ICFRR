from argparse import Namespace
from pathlib import Path

import torch
from tqdm import tqdm

from src.icfrr.configs import get_test_args
from src.icfrr.data_utils import get_datasets, DatasetName
from src.icfrr.metric_utils import calc_metrics, get_pos_matches
from src.icfrr.cache_utils import Cacher
from src.icfrr.plot_utils import plot_results_over_iterations
from src.icfrr.vis.visualise_reranks import visualise_ranks_for_queries, save_vis

torch.multiprocessing.set_sharing_strategy('file_system')


def _get_scores_based_off_im_dom(intersect_points: torch.Tensor, cross_dom_dists_indices: torch.Tensor, KG: int = 8,
                                 KQ: int = 3) -> torch.Tensor:
    '''
    Here we are saying where is every gallery domain image in the list of top images. But only taking into
    account KQ as if an image is too far down the cross domain list then we perceive it as a weaker/bad match and such
    do not take information from its KG ranks
    '''
    intersect_points_filtered = intersect_points[cross_dom_dists_indices[:KQ], :]

    '''
    Then we are basically suggesting that in the gallery to gallery domain, we only count the top KG of images 
    as being good/strong matches 
    '''
    weights = torch.linspace(1, 0, len(cross_dom_dists_indices)).to(intersect_points_filtered.device)
    weights[KG:] = 0

    '''
    Images <KQ will not get a score for all images within KQ due to not getting a score for a self image. Due to this 
    they have 1 less image to have their rank assessed and such has a 1 less divisor than all images >=KG
    '''
    weighted_scores = weights[intersect_points_filtered].mean(0)
    '''
    We then need to sort the weights by the query-gallery rankings to ensure we add the correct weights to the correct
    scores 
    '''
    return weighted_scores[cross_dom_dists_indices]


def reranking_without_each_element(dist_inds: torch.Tensor, dist_vals: torch.Tensor, intersect_points: torch.Tensor,
                                   KG: int = 8, KQ: int = 3, beta: float = 0.1, verbose: bool = False) -> torch.Tensor:
    reranked_dist_inds = torch.zeros_like(dist_inds)
    iter_line = enumerate(tqdm(dist_inds, desc="Reranking")) if verbose else enumerate(dist_inds)

    for row, cross_dom_dists_indices in iter_line:
        weighted_scores = _get_scores_based_off_im_dom(intersect_points, cross_dom_dists_indices, KG, KQ)
        weighted_scores *= beta
        reranked_all_top_patch_dists_values_i = dist_vals[row].to(weighted_scores.device) + weighted_scores

        reranked_vals_i, reranked_inds_i = reranked_all_top_patch_dists_values_i.sort(descending=True)
        reranked_dist_inds[row] = cross_dom_dists_indices.to(weighted_scores.device)[reranked_inds_i]

    return reranked_dist_inds


def _handle_permuting(args: Namespace, iter_num: int, top_dists_indices: torch.Tensor, results_dir: Path):
    (results_dir / "ranks").mkdir(exist_ok=True)
    torch.save(top_dists_indices, results_dir / "ranks" / f"reranking_iter_{iter_num}.pth")
    if len(args.query_idxs_to_vis) > 0:
        vis_dir = results_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        _, datasets = get_datasets(args.data_root, DatasetName(args.dataset), transforms=None,
                                   num_per_class=args.per_class)
        vis = visualise_ranks_for_queries(datasets, args.query_idxs_to_vis, top_dists_indices,
                                          args.num_gallery_ims_to_vis)
        save_vis(vis, vis_dir / f"reranking_iter_{iter_num}.png")


def main(args: Namespace):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    qg_dists, gg_dists, labels = Cacher(args.cache_root / args.dataset / str(args.per_class)).load_caches()

    maps_alls, maps_200s, prec_100s, prec_200s = None, None, None, None
    if labels is not None and not (labels["query"] == -1).all():
        # If there are ground truth labels, we can calculate metrics, however re-ranking itself does not need labels
        # Calculate the original metrics before any reranking takes place
        orig_mAP, orig_mAP_200, orig_prec_25, orig_prec_100, orig_prec_200 = calc_metrics(
            get_pos_matches(labels["gallery"], labels["query"]), torch.argsort(qg_dists), -qg_dists, device,
            flags_already_sorted=False)
        print(
            f'Origional mAP@all: {orig_mAP:.5f}, map@200: {orig_mAP_200:.5f}, prec@25: {orig_prec_25:.5f}, prec@100: '
            f'{orig_prec_100:.5f}, prec@200: {orig_prec_200:.5f}')
        maps_alls = [orig_mAP]
        maps_200s = [orig_mAP_200]
        prec_100s = [orig_prec_100]
        prec_200s = [orig_prec_200]

    top_cross_dom_dists = (-qg_dists).sort(descending=True)
    print("Scores sorted")

    all_gallery_dom_dists = gg_dists.to(device)
    if args.limited_memory:
        im_dom_intersect_points = torch.zeros_like(all_gallery_dom_dists, dtype=torch.int, device=device)
        for i in tqdm(range(all_gallery_dom_dists.shape[1]), desc="Sorting args"):
            im_dom_intersect_points[i] = all_gallery_dom_dists[:, i].argsort().argsort()
    else:
        im_dom_intersect_points = all_gallery_dom_dists.argsort(dim=1).argsort(dim=1)

    print("Intersect points shape: ", im_dom_intersect_points.shape)

    '''
    We know that the same image will always get the lowest distance so will be ranked 0. We do not want this same image 
    to be used so to ensure it carries no weighting we set the rank to -1, such that in the reranking this value will be
    >=KG so index -1 will get either a weighting of 0 or a very small amount, if KG == len(gallery). The rest of the 
    image rankings then also shift down by 1, such that the best ranked different image is at rank 0.
    '''
    im_dom_intersect_points -= 1

    top_dists_indices = top_cross_dom_dists.indices
    top_dists_values = top_cross_dom_dists.values
    print(f'KG: {args.KG}  KQ: {args.KQ} beta: {args.beta}')
    rerank_results_dir = args.rerank_results_dir / args.dataset / str(args.per_class)
    rerank_results_dir.mkdir(exist_ok=True, parents=True)
    _handle_permuting(args, 0, top_dists_indices, rerank_results_dir)
    for i in range(args.n_times):
        top_dists_indices = reranking_without_each_element(top_dists_indices, top_dists_values, im_dom_intersect_points,
                                                           KG=args.KG, KQ=args.KQ, beta=args.beta)
        _handle_permuting(args, i + 1, top_dists_indices, rerank_results_dir)

        if labels is not None and not (labels["query"] == -1).all():
            all_reranked_pos_matches = get_pos_matches(labels["gallery"], labels["query"], top_dists_indices)

            # If it is such a large dataset, can't fit all the tensors required for metrics onto the gpu
            device_for_metrics = torch.device("cpu") if args.cpu_as_metric_device else device
            reranked_maps_all, reranked_mAP_200, reranked_prec_25, reranked_prec_100, reranked_prec_200 = calc_metrics(
                all_reranked_pos_matches, top_dists_indices, top_dists_values, device=device_for_metrics)
            print(
                f'Reranked [{i + 1}/{args.n_times}], mAP@all: {reranked_maps_all:.5f}, map@200: {reranked_mAP_200:.5f}, '
                f'prec@25: {reranked_prec_25:.5f}, prec@100: {reranked_prec_100:.5f}, prec@200: {reranked_prec_200:.5f}'
            )

            maps_alls.append(reranked_maps_all)
            maps_200s.append(reranked_mAP_200)
            prec_100s.append(reranked_prec_100)
            prec_200s.append(reranked_prec_200)
        else:
            print(f'Reranked [{i + 1}/{args.n_times}]')

    if labels is not None and not (labels["query"] == -1).all():
        plot_results_over_iterations(args.n_times, maps_200s, maps_alls, prec_100s, prec_200s, every_x=1,
                                     save_plot=rerank_results_dir / "map_prec_ICFRR.png")


if __name__ == '__main__':
    main(get_test_args())
