import argparse

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union


class DefaultConfig(BaseModel):
    dataset: str = "sketchy"
    zero_version: str = "zeroshot"
    batchsize: int = 32
    eval_period: int = 20
    weight_decay: float = 5e-4
    tri_lambda: float = 1.0
    tri_margin: float = 0.2
    feat_loss_function: str = "triple"
    mean_feats: bool = True
    no_augs: bool = False
    ce_lambda: float = 1.0
    classes_per_epoch: int = 2
    per_class: int = -1
    debug: bool = False

    def __getitem__(self, item):
        return getattr(self, item)


class WandbSettings(BaseModel):
    use: bool = False
    entity: str | None = None
    project_name: str = "ICFRR"


class DefaultSettings(BaseModel):
    wandb_settings: WandbSettings = WandbSettings()
    epochs: int = 100
    parallel: bool = False
    op_root: Union[Path, str] = None
    gpu_id: int = 1
    print_freq: int = -1
    num_workers: int = 4
    lr: float = 0.0001
    base_lr_mult: float = 0.1
    data_root: Path = Path("data")
    op_name: Optional[str] = None

    def __getitem__(self, item):
        return getattr(self, item)

    class Config:
        # To allow torch.device
        arbitrary_types_allowed = True


class RunConfigSettings(DefaultConfig, DefaultSettings):
    pass


def get_model_run_args(external_parsed_args=None):
    parser = shared_args()
    parser.add_argument('--pretrained_root', default="pretrained", type=Path,
                        help='Parent directory of all pretrained models')
    parser.add_argument('--no_cache', action='store_true', help='whether to cache data')
    parser.add_argument('--force_cache', action='store_true', help='whether to force recalculation and caching of data')

    return parser.parse_args(external_parsed_args)


def get_test_args(external_parsed_args=None):
    parser = shared_args()
    parser.add_argument('--KG', default=256, type=int,
                        help='limit of how many of domB to domB we are counting as a strong match')
    parser.add_argument('--KQ', default=250, type=int, help='how many of domA-domB matches we count as strong')
    parser.add_argument('--beta', default=0.5, type=float, help='factor of effect of the rerank')
    parser.add_argument('--n_times', default=5, type=int, help='how many times to run the reranking')
    parser.add_argument("--limited_memory", action='store_true',
                        help='If you have limited memory, try enabling this to chunk up some operations')
    parser.add_argument("--cpu_as_metric_device", action='store_true',
                        help="to calculate metrics of huge datasets, sometimes cpu memory is larger so should be used")
    parser.add_argument("--rerank_results_dir", default="results", type=Path,
                        help='Directory to store reranking results')
    parser.add_argument("--query_idxs_to_vis", nargs='+', type=int, default=[],
                        help='Idxs of the query ids for reranking visualisation, if left blank no visualisation occurs')
    parser.add_argument("--num_gallery_ims_to_vis", default=20, type=int,
                        help='If visualising with query_idxs_to_vis, how many of the top query-gallery results to see')

    return parser.parse_args(external_parsed_args)


def shared_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=32, type=int, metavar='N', help='batchsize to use')
    parser.add_argument('--gpu_id', default=0, type=int, metavar='N', help='gpu id to use')
    parser.add_argument('--per_class', default=-1, type=int, metavar='N', help='number of ims per class (-1 is all)')
    parser.add_argument('--dataset', default="tuberlin", type=str, help='dataset to use, options in DatasetName')
    parser.add_argument('--data_root', default="datasets", type=Path, help='Parent directory of all datasets')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--cache_root', default="cache", type=Path, help='Parent directory to store caches')

    return parser
