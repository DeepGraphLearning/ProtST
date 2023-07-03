import os
import sys
import csv
import tqdm
import math
import tqdm
import pprint
import shutil
import logging
import argparse
import numpy as np

import torch
from torch.utils import data as torch_data

import torchdrug
from torchdrug import core, datasets, tasks, models, layers, utils
import torchdrug.data as td_data
from torchdrug.utils import comm, cuda

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

def build_data(cfg, logger):
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        train_set, valid_set, test_set = _dataset.split(['train', 'valid', cfg.test_split])
    else:
        train_set, valid_set, test_set = _dataset.split()
    train_set = sample_train_set(_dataset, train_set, cfg.shot)
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    train_dataloader = td_data.DataLoader(train_set, cfg.engine.batch_size, shuffle=False)
    train_dataloader = tqdm.tqdm(train_dataloader)
    test_dataloader = td_data.DataLoader(test_set, cfg.engine.batch_size, shuffle=False)
    test_dataloader = tqdm.tqdm(test_dataloader)
    return train_dataloader, test_dataloader, _dataset.target_fields[0]

def build_model(cfg, logger, device):
    model = core.Configurable.load_config_dict(cfg.task.model)
    if cfg.get("checkpoint") is not None:
        checkpoint = os.path.expanduser(cfg.checkpoint)
        pretrained_model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
        model_state = {}
        for k in pretrained_model_state.keys():
            if k.startswith("protein_model."):
                model_state[k[14:]] = pretrained_model_state[k]
        model.load_state_dict(model_state, strict=False)
        logger.warning(pprint.pformat(model_state.keys()))

    for name, p in model.named_parameters():
        p.requires_grad = False
    model = model.to(device)
    model.eval()
    return model

def train_set_embedding(cfg, logger, device, train_dataloader, target_field, model):

    # get training sample embeddings for each class
    train_embeddings = {}
    for batch in train_dataloader:
        batch = cuda(batch, device=device)
        
        graph = batch["graph"]
        output = model(graph, graph.residue_feature.float())
        protein_features = output["graph_feature"]

        targets = batch[target_field]
        for i in range(targets.shape[0]):
            if targets[i].cpu().item() not in train_embeddings:
                train_embeddings[targets[i].cpu().item()] = [protein_features[i]]
            else:
                train_embeddings[targets[i].cpu().item()].append(protein_features[i])

    class_embeddings = []
    for i in range(len(train_embeddings)):
        embeddings = torch.stack(train_embeddings[i], dim=0)  # [S, D]
        class_embeddings.append(embeddings)
    class_embeddings = torch.stack(class_embeddings, dim=0)  # [C, S, D]

    return class_embeddings


def nonparam_few_shot_eval(cfg, test_dataloader, target_field, model, class_embeddings, logger):

    # get predictions and targets
    preds, targets = [], []
    for batch in test_dataloader:
        batch = cuda(batch, device=device)
        target = batch[target_field]
        targets.append(target)

        graph = batch["graph"]
        output = model(graph, graph.residue_feature.float())
        protein_feature = output["graph_feature"]  # [N, D]
        dist = ((protein_feature.unsqueeze(1).unsqueeze(1) - class_embeddings.unsqueeze(0)) ** 2).sum(-1)  # [N, C, S]
        pred = torch.exp(-dist).sum(-1)  # [N, C]
        preds.append(pred)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    accuracy = (preds.argmax(dim=-1) == targets).float().mean().item()
    logger.warning("Zero-shot accuracy: %.6f" % accuracy)

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
    
    device = torch.device("cuda:0")
    train_dataloader, test_dataloader, target_field = build_data(cfg, logger)
    model = build_model(cfg, logger, device)
    class_embeddings = train_set_embedding(cfg, logger, device, train_dataloader, target_field, model)
    nonparam_few_shot_eval(cfg, test_dataloader, target_field, model, class_embeddings, logger)
