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
from torch.utils import data as torch_data

import torchdrug
from torchdrug import core, datasets, tasks, models, layers, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from protst import dataset, data, model, task, util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="./config/few_shot/few_shot.yaml")
    parser.add_argument("-s", "--shot", type=int, help="shot number (number of training samples per class)",
                        default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=0)

    return util.proceed_parser(parser)

def sample_train_set(dataset, train_set, shot=1):
    target_field = dataset.target_fields[0]
    label2indices = {}
    for idx, sample in enumerate(train_set):
        label = sample[target_field]
        if label not in label2indices:
            label2indices[label] = [idx]
        else:
            label2indices[label].append(idx)

    sampled_indices = []
    for label, indices in label2indices.items():
        indices = np.array(indices)
        if len(indices) >= shot:
            indices_ = np.random.choice(indices, size=shot, replace=False)
        else:
            indices_ = np.random.choice(indices, size=shot, replace=True)
        sampled_indices.append(indices_)
    sampled_indices = np.concatenate(sampled_indices)
    sampled_train_set = torch_data.Subset(train_set, sampled_indices)

    return sampled_train_set


def build_solver(cfg, logger):
    # build dataset
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        train_set, valid_set, test_set = _dataset.split(['train', 'valid', cfg.test_split])
    else:
        train_set, valid_set, test_set = _dataset.split()
    train_set = sample_train_set(_dataset, train_set, cfg.shot)
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

def train_and_test(cfg, solver):
    solver.model.split = "train"
    solver.train(**cfg.train)
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
    solver.model.split = "test"
    solver.evaluate("test")

if __name__ == "__main__":
    args, vars = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config, context=vars)
    cfg.shot = args.shot
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
    train_and_test(cfg, solver)