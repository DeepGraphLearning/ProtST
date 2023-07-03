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

from protst import dataset, model, task, util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="./config/t2p_retrieval/go_mf.yaml")
    parser.add_argument("-o", "--output_file", help="output file for retrieval results",
                        default="./t2p_go_mf.txt")
    parser.add_argument("-k", "--topk", help="topk for retrieval",
                        type=int, default=20)

    return util.proceed_parser(parser)

def build_dataset(cfg, logger):
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#samples: %d" % len(_dataset))

    return _dataset

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

def fetch_logit_scale(cfg, logger):
    checkpoint = os.path.expanduser(cfg.checkpoint)
    model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
    logit_scale = model_state["logit_scale"]
    logit_scale.requires_grad = False

    logit_scale = logit_scale.to(device)
    logit_scale = logit_scale.exp()
    return logit_scale

def prompt_embedding(cfg, text_model):
    # fetch prompts
    prompt_file = os.path.expanduser(cfg.prompt.path)
    prompts = []
    with open(prompt_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            prompts.append(line)

    # embed prompts
    prompt_feature = []
    for prompt in prompts:
        prompt_token = text_model.tokenizer.encode(prompt, max_length=cfg.prompt.max_length, truncation=True,
                                                   add_special_tokens=False)
        prompt_token = [text_model.cls_idx] + prompt_token
        prompt_token = torch.tensor(prompt_token, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = prompt_token != text_model.pad_idx
        model_output = text_model(None, input_ids=prompt_token, attention_mask=attention_mask)
        prompt_feature.append(model_output["text_feature"])
    prompt_feature = torch.cat(prompt_feature, dim=0)
    prompt_feature = prompt_feature / prompt_feature.norm(dim=-1, keepdim=True)

    return prompt_feature

def get_prediction(cfg, logger, device, _dataset, protein_model, logit_scale, prompt_feature):
    dataloader = td_data.DataLoader(_dataset, cfg.batch_size, shuffle=False)
    dataloader = tqdm.tqdm(dataloader)

    # embed proteins
    preds = []
    for batch in dataloader:
        batch = cuda(batch, device=device)
        graph = batch["graph"]
        output = protein_model(graph, graph.residue_feature.float())
        protein_feature = output["graph_feature"]
        protein_feature = protein_feature / protein_feature.norm(dim=-1, keepdim=True)
        pred = logit_scale * prompt_feature @ protein_feature.t()
        preds.append(pred)
    preds = torch.cat(preds, dim=-1)

    return preds

if __name__ == "__main__":
    args, vars = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config, context=vars)
    cfg.prompt.path = os.path.realpath(cfg.prompt.path)
    
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    device = torch.device("cuda:0")
    _dataset = build_dataset(cfg, logger)
    protein_model = build_protein_model(cfg, logger, device)
    text_model = build_text_model(cfg, logger, device)
    logit_scale = fetch_logit_scale(cfg, logger)

    prompt_feature = prompt_embedding(cfg, text_model)
    preds = get_prediction(cfg, logger, device, _dataset, protein_model, logit_scale, prompt_feature)

    # get retrieval results
    assert preds.shape[1] == len(_dataset.pdb_files)
    _, topk_indices = torch.topk(preds, args.topk, dim=-1)
    with open(args.output_file, "w") as fout:
        for prompt_id in range(topk_indices.shape[0]):
            line = "\t".join([_dataset.pdb_files[i] for i in topk_indices[prompt_id]]) + "\n"
            fout.write(line)
    logger.warning("Output file: %s" % args.output_file)
