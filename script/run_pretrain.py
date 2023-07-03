import os
import sys
import csv
import tqdm
import math
import pprint
import shutil
import logging
import argparse
import numpy as np

import torch

import torchdrug
from torchdrug import core, datasets, tasks, models, layers, utils
import torchdrug.data as td_data
from torchdrug.utils import comm, cuda

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from protst import dataset, data, model, task, util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("-s", "--save_step", type=int, default=1,
                        help="the interval of saving checkpoint.")

    return util.proceed_parser(parser)

def build_solver(cfg, logger, output_dir, old_dir):
    # build task model
    torch.cuda.set_device(comm.get_rank())
    torch.cuda.empty_cache()
    task = core.Configurable.load_config_dict(cfg.task)
    
    # fix text_model
    fix_text_model = cfg.get("fix_text_model", False)
    if fix_text_model:
        for name, p in task.text_model.named_parameters():
            if "text_mlp" not in name:
                p.requires_grad = False
    fix_protein_model = cfg.get("fix_protein_model", False)
    if fix_protein_model:
        for name, p in task.protein_model.named_parameters():
            if "graph_mlp" not in name and "residue_mlp" not in name:
                p.requires_grad = False

    # build dataset
    if cfg.dataset['class'] in ["UniProtSeqText"]:
        cfg.dataset['text_tokenizer'] = task.text_model.tokenizer

    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        train_set, valid_set, test_set = _dataset.split(['train', 'valid', cfg.test_split])
    else:
        train_set, valid_set, test_set = _dataset.split()
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    # build solver
    if not "lr_ratio" in cfg:
        cfg.optimizer.params = task.parameters()
    else:
        params = []
        head_params = []
        params.append({"params": task.protein_model.graph_mlp.parameters(), "lr": cfg.optimizer.lr})
        head_params.extend(list(task.protein_model.graph_mlp.parameters()))
        params.append({"params": task.protein_model.residue_mlp.parameters(), "lr": cfg.optimizer.lr})
        head_params.extend(list(task.protein_model.residue_mlp.parameters()))
        params.append({"params": task.text_model.text_mlp.parameters(), "lr": cfg.optimizer.lr})
        head_params.extend(list(task.text_model.text_mlp.parameters()))
        params.append({"params": task.text_model.word_mlp.parameters(), "lr": cfg.optimizer.lr})
        head_params.extend(list(task.text_model.word_mlp.parameters()))
        if cfg.task.get("mlm_weight", 0) > 0:
            params.append({"params": task.mlm_head.parameters(), "lr": cfg.optimizer.lr})
            head_params.extend(list(task.mlm_head.parameters()))
        if cfg.task.get("mmp_weight", 0) > 0:
            params.append({"params": task.fusion_model.parameters(), "lr": cfg.optimizer.lr})
            head_params.extend(list(task.fusion_model.parameters()))
            params.append({"params": task.mmp_protein_head.parameters(), "lr": cfg.optimizer.lr})
            head_params.extend(list(task.mmp_protein_head.parameters()))
            params.append({"params": task.mmp_text_head.parameters(), "lr": cfg.optimizer.lr})
            head_params.extend(list(task.mmp_text_head.parameters()))
        backbone_params = list(set(task.parameters()) - set(head_params))
        params.append({"params": backbone_params, "lr": cfg.optimizer.lr * cfg.lr_ratio})
        cfg.optimizer.params = params
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if not "scheduler" in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)

    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint, load_optimizer=False)

    return solver

def train_and_validate(cfg, solver, step=1):

    if not cfg.train.num_epoch > 0:
        return

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)

if __name__ == "__main__":
    args, vars = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config, context=vars)
    util.set_seed(args.seed)
    
    old_dir = os.getcwd()
    output_dir = util.create_working_directory(cfg)
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)
        shutil.copyfile(args.config, os.path.basename(args.config))

    solver = build_solver(cfg, logger, output_dir, old_dir)
    train_and_validate(cfg, solver, step=args.save_step)