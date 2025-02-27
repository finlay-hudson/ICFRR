from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from src.icfrr.configs import get_test_args
from src.icfrr.data_utils import get_datasets, DatasetName


def save_vis(vis: np.ndarray, out_fn: Path | str):
    Image.fromarray(vis).save(out_fn)


def add_border(frame: Image.Image | np.ndarray, border_ceof: int = 3, col: tuple = (255, 255, 255)):
    was_pil = False
    if isinstance(frame, Image.Image):
        was_pil = True
        frame = np.array(frame)
    new_frame = cv2.copyMakeBorder(frame.copy(), border_ceof, border_ceof, border_ceof, border_ceof,
                                   cv2.BORDER_CONSTANT, value=col)
    if was_pil:
        new_frame = Image.fromarray(new_frame)

    return new_frame


def visualise_ranks_for_queries(datasets: dict, query_ids_to_view: list, all_query_gallery_results: torch.Tensor,
                                num_gallery_to_view: int = 10):
    query_gallery_results = all_query_gallery_results[query_ids_to_view]
    results = []
    for query_idx, query_gallery_result in zip(query_ids_to_view, query_gallery_results):
        query_rank_vis = visualise_ranks(datasets, query_gallery_result, query_idx, num_gallery_to_view)
        results.append(query_rank_vis)

    return np.concatenate(results)


def visualise_ranks(datasets: dict, query_gallery_result: torch.Tensor, query_idx: int, num_gallery_to_view: int = 10,
                    border_thickness: int = 5):
    query_img = Image.open(datasets["test"]["query"].root_dir / datasets["test"]["query"].file_ls[query_idx])
    query_img = add_border(query_img, border_thickness, (0, 0, 0))
    query_label = datasets["test"]["query"].labels[query_idx] if datasets["test"]["query"].labels is not None else None
    gallery_imgs = []
    for ind in query_gallery_result[:num_gallery_to_view]:
        gallery_label = None
        if datasets["test"]["gallery"].labels is not None and query_label is not None:
            gallery_label = datasets["test"]["gallery"].labels[ind]
        img = Image.open(datasets["test"]["gallery"].root_dir / datasets["test"]["gallery"].file_ls[ind])
        if gallery_label is not None and query_label is not None:
            border_col = (0, 255, 0) if gallery_label == query_label else (255, 0, 0)
        else:
            border_col = (127, 127, 127)
        img = add_border(img, border_thickness, border_col)
        gallery_imgs.append(img)

    return np.concatenate([query_img] + gallery_imgs, axis=1)


def main(args: Namespace):
    dataset = DatasetName(args.dataset)
    num_gallery_to_view = 20
    query_ids_to_view = [1887, 1856, 2333, 1803, 2369, 526, 866, 1085, 1688]
    num_classes, datasets = get_datasets(args.data_root, dataset, None, None,
                                         num_per_class=args.per_class)

    result_dir = args.rerank_results_dir / args.dataset / str(args.per_class) / "ranks"
    result_files = sorted(list(result_dir.glob("*.pth")))
    assert len(result_files), f"No results files in {result_dir}"
    vis_dir = result_dir.parent / "vis"
    vis_dir.mkdir(exist_ok=True)

    print(f"Saving visualisation to {vis_dir}")
    for result_file in result_files:
        iter_num = int(result_file.stem.split("_")[-1])
        rerank_inds = torch.load(result_file)

        vis = visualise_ranks_for_queries(datasets, query_ids_to_view, rerank_inds, num_gallery_to_view)
        _, datasets = get_datasets(args.data_root, DatasetName(args.dataset), transforms=None,
                                   num_per_class=args.per_class)
        save_vis(vis, vis_dir / f"reranking_iter_{iter_num}.png")


if __name__ == "__main__":
    main(get_test_args())
