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

def build_solver(cfg, logger):
    # build dataset
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        train_set, valid_set, test_set = _dataset.split(['train', 'valid', cfg.test_split])
    else:
        train_set, valid_set, test_set = _dataset.split()
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    # build task model
    if cfg.task["class"] in ["PropertyPrediction", "InteractionPrediction"]:
        cfg.task.task = _dataset.tasks
    elif cfg.task["class"] == "MultipleBinaryClassification":
        cfg.task.task = [_ for _ in range(len(_dataset.tasks))]
    task = core.Configurable.load_config_dict(cfg.task)

    # build solver
    if not "lr_ratio" in cfg:
        cfg.optimizer.params = task.parameters()
    else:
        if hasattr(task, "preprocess"):
            _ = task.preprocess(train_set, valid_set, test_set)
        cfg.optimizer.params = [
            {'params': task.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
            {'params': task.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if not "scheduler" in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)
    if cfg.get("checkpoint") is not None:
        checkpoint = os.path.expanduser(cfg.checkpoint)
        state = torch.load(checkpoint, map_location=solver.device)
        model_dict = solver.model.state_dict()
        keys = [k for k in state['model'].keys()]
        for k in keys:
            if "protein_model" in k:
                new_k = k.replace("protein_model", "model")
                state['model'][new_k] = state['model'][k]
        for k in keys:
            if "protein_model" in k or "text_model" in k or k not in model_dict:
                state['model'].pop(k)
        solver.model.load_state_dict(state['model'], strict=False)
    
    # fix the protein encoder and the protein projection layer
    fix_encoder = cfg.get("fix_encoder", False)
    if fix_encoder:
        for name, p in task.model.named_parameters():
            p.requires_grad = False

    return solver

def train_and_validate(cfg, solver):
    step = math.ceil(cfg.train.num_epoch / 10)
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        if "test_batch_size" in cfg:
            solver.batch_size = cfg.test_batch_size
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        solver.batch_size = cfg.engine.batch_size

        score = []
        for k, v in metric.items():
            if k.startswith(cfg.eval_metric):
                if "root mean squared error" in cfg.eval_metric:
                    score.append(-v)
                else:
                    score.append(v)
        score = sum(score) / len(score)
        if score > best_score:
            best_score = score
            best_epoch = solver.epoch
            solver.save("best_valid_epoch.pth")

    solver.load("best_valid_epoch.pth")
    return solver, best_epoch

def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
    solver.model.split = "valid"
    solver.evaluate("valid")
    solver.model.split = "test"
    solver.evaluate("test")

if __name__ == "__main__":
    args, vars = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config, context=vars)
    util.set_seed(args.seed)

    output_dir = util.create_working_directory(cfg)
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)
        shutil.copyfile(args.config, os.path.basename(args.config))
    os.chdir(output_dir)

    solver = build_solver(cfg, logger)
    solver, best_epoch = train_and_validate(cfg, solver)
    if comm.get_rank() == 0:
        logger.warning("Best epoch on valid: %d" % best_epoch)
    test(cfg, solver)