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

def build_test_data(cfg, logger):
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        _, _, test_set = _dataset.split(['train', 'valid', cfg.test_split])
    else:
        _, _, test_set = _dataset.split()
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#test: %d" % len(test_set))
    
    dataloader = td_data.DataLoader(test_set, cfg.batch_size, shuffle=False)
    dataloader = tqdm.tqdm(dataloader)

    return dataloader, _dataset.target_fields[0]

def build_protein_model(cfg, logger, device):
    protein_model = core.Configurable.load_config_dict(cfg.protein_model)
    if cfg.get("checkpoint") is not None:
        checkpoint = os.path.expanduser(cfg.checkpoint)
        model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
        protein_model_state = {}
        for k in model_state.keys():
            if k.startswith("protein_model."):
                protein_model_state[k[14:]] = model_state[k]
        protein_model.load_state_dict(protein_model_state, strict=False)
        logger.warning(pprint.pformat(protein_model_state.keys()))

    for name, p in protein_model.named_parameters():
        p.requires_grad = False
    protein_model = protein_model.to(device)
    protein_model.eval()
    return protein_model


def build_text_model(cfg, logger, device):
    text_model = core.Configurable.load_config_dict(cfg.text_model)
    if cfg.get("checkpoint") is not None:
        checkpoint = os.path.expanduser(cfg.checkpoint)
        model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
        text_model_state = {}
        for k in model_state.keys():
            if k.startswith("text_model."):
                text_model_state[k[11:]] = model_state[k]
        text_model.load_state_dict(text_model_state, strict=False)
        logger.warning(pprint.pformat(text_model_state.keys()))
    
    for name, p in text_model.named_parameters():
        p.requires_grad = False
    text_model = text_model.to(device)
    text_model.eval()
    return text_model

def fetch_labels(cfg, logger, id_field="id"):
    # fetch label descriptions
    label_description = os.path.expanduser(cfg.label.path)
    with open(label_description, "r") as fin:
        reader = csv.reader(fin, delimiter="\t")
        fields = next(reader)
        label_dict = {}
        for values in reader:
            label_text = []
            for field, value in zip(fields, values):
                if field == id_field:
                    label_id = int(value)
                if field in cfg.label.field:
                    label_text.append(value)
            label_dict[label_id] = " ".join(label_text)
    labels = [label_dict[i] for i in range(len(label_dict))]

    return labels

def label_embedding(cfg, labels, text_model):
    # embed label descriptions
    label_feature = []
    for label in labels:
        label_token = text_model.tokenizer.encode(label, max_length=cfg.label.max_length, 
                                                  truncation=True, add_special_tokens=False)
        label_token = [text_model.cls_idx] + label_token
        label_token = torch.tensor(label_token, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = label_token != text_model.pad_idx
        model_output = text_model(None, input_ids=label_token, attention_mask=attention_mask)
        
        label_feature.append(model_output["text_feature"])
    label_feature = torch.cat(label_feature, dim=0)
    label_feature = label_feature / label_feature.norm(dim=-1, keepdim=True)

    return label_feature

def fetch_logit_scale(cfg, logger, device):

    checkpoint = os.path.expanduser(cfg.checkpoint)
    model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
    logit_scale = model_state["logit_scale"]
    logit_scale.requires_grad = False
    logit_scale = logit_scale.to(device)
    logit_scale = logit_scale.exp()

    return logit_scale

def zero_shot_eval(cfg, logger, device, 
                   test_dataloader, target_field, protein_model, logit_scale, label_feature):
    
    # get prediction and target
    preds, targets = [], []
    for batch in test_dataloader:
        batch = cuda(batch, device=device)
        
        target = batch[target_field]
        targets.append(target)

        graph = batch["graph"]
        output = protein_model(graph, graph.residue_feature.float())
        protein_feature = output["graph_feature"]
        protein_feature = protein_feature / protein_feature.norm(dim=-1, keepdim=True)
        pred = logit_scale * protein_feature @ label_feature.t()
        preds.append(pred)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    accuracy = (preds.argmax(dim=-1) == targets).float().mean().item()
    logger.warning("Zero-shot accuracy: %.6f" % accuracy)

if __name__ == "__main__":
    args, vars = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config, context=vars)

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    device = torch.device("cuda:0")

    test_dataloader, target_field = build_test_data(cfg, logger)
    protein_model = build_protein_model(cfg, logger, device)
    text_model = build_text_model(cfg, logger, device)
    logit_scale = fetch_logit_scale(cfg, logger, device)

    labels = fetch_labels(cfg, logger)
    label_feature = label_embedding(cfg, labels, text_model)
    zero_shot_eval(cfg, logger, device, 
                   test_dataloader, target_field, protein_model, logit_scale, label_feature)
