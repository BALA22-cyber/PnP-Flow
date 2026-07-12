import argparse
import os
from datetime import datetime
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from pnpflow.dataloaders import DataLoaders
from pnpflow.train_flow_matching import FLOW_MATCHING
from pnpflow.utils import define_model, load_cfg_from_cfg_file, merge_cfg_from_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training entrypoint")
    cfg = load_cfg_from_cfg_file("./config/main_config.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument(
        "--no_cap",
        action="store_true",
        help="Train over the full DataLoader instead of stopping after 21 batches per epoch.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Path to a training_state_latest.pt checkpoint created by this training entrypoint.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Existing run directory name to continue when --resume-checkpoint is supplied.",
    )
    args = parser.parse_args()
    cfg.no_cap = args.no_cap
    cfg.resume_checkpoint = args.resume_checkpoint
    if args.run_name is not None:
        cfg.run_name = args.run_name

    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + f"config/dataset_config/{cfg.dataset}.yaml"
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_config_file = cfg.root + f"config/method_config/{cfg.method}.yaml"
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]

    return cfg


def setup_distributed(args):
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("main_ddp.py must be launched with torchrun.")

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    args.device = torch.device("cuda", args.local_rank)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_run_name(args):
    run_name = getattr(args, "run_name", None)
    if run_name in [None, "None", ""] and args.rank == 0:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.dataset}_{args.model}"

    obj = [run_name]
    dist.broadcast_object_list(obj, src=0)
    args.run_name = obj[0]


def make_distributed_train_loader(train_loader, args):
    drop_last = getattr(train_loader.batch_sampler, "drop_last", False)
    sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=drop_last,
    )
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        collate_fn=train_loader.collate_fn,
        num_workers=train_loader.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def main():
    args = parse_args()
    setup_distributed(args)
    broadcast_run_name(args)

    if args.seed is not None:
        seed = args.seed + args.rank
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True

    if args.rank == 0:
        print(
            f"DDP training on {args.world_size} GPU(s); "
            f"local_rank={args.local_rank}; device={args.device}; run_name={args.run_name}",
            flush=True,
        )

    model, _ = define_model(args)
    model = model.to(args.device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if not args.train:
        raise ValueError("main_ddp.py currently supports training only. Use main.py for eval/inference.")

    args.batch_size = args.batch_size_train
    data_loaders = DataLoaders(
        args.dataset, args.batch_size_train, args.batch_size_train).load_data()
    data_loaders["train"] = make_distributed_train_loader(data_loaders["train"], args)

    if args.model == "ot":
        generative_method = FLOW_MATCHING(model, args.device, args)
    else:
        raise ValueError("main_ddp.py currently supports model ot training only.")

    try:
        generative_method.train(data_loaders)
        if args.rank == 0:
            print("Training done!", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
