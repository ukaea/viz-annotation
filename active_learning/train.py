"""
@authors: 
    Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk)
"""

import os
import sys
import yaml
import math
import json
import hydra
import shutil
import random
import datetime
import traceback
import torch.multiprocessing as mp
import numpy as np
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer
from utils.logger import create_logger
from utils.misc import set_random_seed

import torch
from torch.distributed import destroy_process_group

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True

def copy_src(root_src_dir, root_dst_dir, overwrite=True):
    print(f"Copying source: {root_src_dir}")
    for src_dir, dirs, files in os.walk(root_src_dir):
        if '__pycache__' in src_dir:
            continue
        if '.ipynb_checkpoints' in src_dir:
            continue
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if 'cpython' in file:
                continue
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.exists(dst_file):
                if overwrite:
                    shutil.copy(src_file, dst_file)
            else:
                shutil.copy(src_file, dst_file)
        
def ddp_setup(rank: int, world_size: int, backend="nccl"):
    # set master address if not set
    if os.getenv("MASTER_ADDR", None) is None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"  # "localhost"
        os.environ["MASTER_PORT"] = "29500"  # "12355"

    # torch.distributed.init_process_group(backend="nccl",
    # init_method='env://', rank=rank, world_size=world_size)
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=36000),
    )
    torch.cuda.set_device(rank)

    # Try to use all memory allocated for caching - this is to prevent cuda
    # out of memory which is caused by another training process
    torch.cuda.set_per_process_memory_fraction(0.99, rank)
    print(
        f"DDP Initialized - Master :{os.environ['MASTER_ADDR']}, "
        f"Port:{os.environ['MASTER_PORT']}, "
        f"rank={rank}, world_size={world_size}"
    )


def run(local_rank, world_size, cfg):
    
    print(f"Running {local_rank}/{world_size} ......")

    
    if cfg.optim.lr_adjust:
        if cfg.optim.lr_adjust_rule == "sqrt_wrt_1024":
            cfg.optim.lr = round(
                cfg.optim.lr * math.sqrt(cfg.train.batch_size * world_size / 1024), 6
            )
        else:
            cfg.optim.lr = cfg.optim.lr * world_size * cfg.train.batch_size / 16

    if cfg.train.ddp:
        ddp_setup(local_rank, world_size)
        print(
            f"DDP initialized for rank {local_rank}/{world_size}: ",
            torch.distributed.is_initialized(),
        )

    if not cfg.exp.name:
        cfg.exp.name = f"exp_{datetime.datetime.now().timestamp()}"
        
    # Don't overwrite exp.name as it cause exp.name name too long in case of for loop.
    exp_name = "_".join([
            cfg.exp.name,
            f"{cfg.net.type}",
            f"h{cfg.net.hidden_dim}",
            f"lr{cfg.optim.lr:.4f}",
            f"e{cfg.train.epochs}",
            f"b{cfg.train.batch_size*world_size}",
            f"s{cfg.data.train_samples}",
            f"c{cfg.data.context_len}",
            # f"decay{cfg.optim.weight_decay}"
    ])
    if cfg.net.detection:
        exp_name += f"_elm_det"
        
    if cfg.data.class_balance:
         exp_name+= f"_{cfg.data.class_balance}"
        
    if cfg.train.single_batch:
        exp_name += "_single_batch"

    print("Experiment:", exp_name)
    kfold_results = []
    for k in range(cfg.data.n_folds):
        cfg.data.curr_fold = k
        
        cfg.exp.exp_dir = f"{cfg.exp.log_dir}/session{cfg.exp.session}/{exp_name}/fold{cfg.data.curr_fold+1}"
    
        # Create log directories
        if not cfg.train.debug:
    
            if not cfg.exp.overwrite:
                if os.path.exists(cfg.exp.exp_dir):
                    print(f"Experiment {cfg.exp.exp_dir} already exists. Skipping ... !!!")
                    return 
                    
            os.makedirs(cfg.exp.exp_dir, exist_ok=True)
    
            # save config
            with open(os.path.join(cfg.exp.exp_dir, "cfg.yaml"), "w") as f:
                yaml.dump(OmegaConf.to_yaml(cfg), f)
    
            # Save the command
            with open(cfg.exp.exp_dir + "/command.txt", "w") as f:
                f.write(" ".join(sys.argv))
    
            # Backup src scripts
            copy_src("../src", f"{cfg.exp.exp_dir}/src/")
    
        # create logger 
        trace_func = print if cfg.train.debug else create_logger(cfg.exp.exp_dir + f"/train_log.txt").info
        
        # Create trainer
        trainer = Trainer(cfg=cfg, 
                          rank=local_rank, 
                          world_size=world_size, 
                          trace_func=trace_func, 
                          exp_name=exp_name,
                          )
    
        if local_rank == 0:
            trace_func(f"Result will be logged in : {cfg.exp.exp_dir}")
            trace_func(trainer.network)
            trace_func(f"Training parameters: {trainer.num_params}")
    
        trainer.train()
        # kfold_results.append(trainer.best_metrics_results)
        
        if cfg.train.ddp:
            destroy_process_group()

    # print("***** K-fold average results*******")
    # avg_results = defaultdict(float)
    # for i, results in enumerate(kfold_results):
    #     print(f"Fold-{i:02}", {k: int(val) if k in ['tp', 'fp', 'tn', 'fn'] else round(val, 3) for k, val in results.items()})
    #     for k, val in results.items():
    #         avg_results[k] += val
    #         if k not in ['tp', 'fp', 'tn', 'fn'] and (i==len(kfold_results)-1):
    #             avg_results[k] /= (i+1)
    # print(f"Average", {k: int(val) if k in ['tp', 'fp', 'tn', 'fn'] else round(val, 3) for k, val in avg_results.items()})  
    # with open(os.path.dirname(cfg.exp.exp_dir) + '/kfold_results.json', 'w') as f:
    #     json.dump(avg_results, f)

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    
    # set seed
    set_random_seed(cfg.rng.seed)
    
    # Print current config
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.train.ddp:
        # Spawn ddp processes either use (1) torch distributed run or
        # (2) use mp.spawn
        if cfg.train.dist_run:
            # os.environ is set during distributed run
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            run(local_rank, world_size, cfg)

        else:
            world_size = torch.cuda.device_count()
            mp.spawn(run, args=(world_size, cfg), nprocs=world_size)

    else:
        run(0, 1, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting from training early because of KeyboardInterrupt")
        sys.exit()
    except Exception as e:
        print(e)
        traceback.print_exc()
