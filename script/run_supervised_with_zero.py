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
    parser.add_argument("-sc", "--supervised_config", help="yaml configuration file for supervised predictor",
                        required=True)
    parser.add_argument("-zc", "--zero_config", help="yaml configuration file for zero-shot predictor",
                        required=True)
    parser.add_argument("-sch", "--supervised_checkpoint", help="checkpoint for supervised predictor",
                        default="/home/tiger/scratch/torchprotein_output/subloc/best_valid_epoch.pth")
    parser.add_argument("-e", "--ensemble_scheme", help="scheme to ensemble supervised and zero-shot predictors",
                        default="logit", choices=["logit", "prob"])
    parser.add_argument("-a", "--alpha", help="the weight of zero-shot predictor",
                        type=float, default=1.0)

    args, unparsed = parser.parse_known_args()
    s_vars = util.detect_variables(args.supervised_config)
    z_vars = util.detect_variables(args.zero_config)
    vars = {*s_vars, *z_vars}
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs() if v != "null"}
    return args, vars

def build_test_set(cfg, logger):
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
    return _dataset, dataloader, _dataset.target_fields[0]

def build_supervised_model(cfg, _dataset, logger, device):
    if cfg.task["class"] in ["PropertyPrediction", "InteractionPrediction"]:
        cfg.task.task = _dataset.tasks
    elif cfg.task["class"] == "MultipleBinaryClassification":
        cfg.task.task = [_ for _ in range(len(_dataset.tasks))]
    model = core.Configurable.load_config_dict(cfg.task)
    if hasattr(model, "preprocess"):
        _ = model.preprocess(_dataset, None, None)
    assert cfg.get("checkpoint") is not None
    if cfg.get("checkpoint") is not None:
        checkpoint = os.path.expanduser(cfg.checkpoint)
        model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(model_state, strict=True)
        logger.warning(pprint.pformat(model_state.keys()))

    for name, p in model.named_parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model

def build_protein_model(cfg, logger, device):
    protein_model = core.Configurable.load_config_dict(cfg.protein_model)
    assert cfg.get("checkpoint") is not None
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
    protein_model.to(device)
    protein_model.eval()
    return protein_model

def build_text_model(cfg, logger, device):
    text_model = core.Configurable.load_config_dict(cfg.text_model)
    assert cfg.get("checkpoint") is not None
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
                if "prompt" in cfg and field in cfg.prompt:
                    processed_value = ""
                    if "prefix" in cfg.prompt[field]:
                        processed_value += cfg.prompt[field].prefix + " "
                    processed_value += value
                    if "suffix" in cfg.prompt[field]:
                        processed_value += " " + cfg.prompt[field].suffix
                else:
                    processed_value = value
                if field in cfg.label.field:
                    label_text.append(processed_value)
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

    zero_checkpoint = os.path.expanduser(cfg.checkpoint)
    model_state = torch.load(zero_checkpoint, map_location=torch.device("cpu"))["model"]
    logit_scale = model_state["logit_scale"]
    logit_scale.requires_grad = False

    logit_scale = logit_scale.to(device)
    logit_scale = logit_scale.exp()
    return logit_scale

def ensemble_eval(cfg, logger, device, test_dataloader, target_field, 
                  supervised_model, protein_model, logit_scale, label_feature, 
                  ensemble_scheme="logit", alpha=1.0):
    
    # get prediction and target
    preds, targets = [], []
    for batch in test_dataloader:
        batch = cuda(batch, device=device)
        
        target = batch[target_field]
        targets.append(target)

        # get predictions by two ways and ensemble them
        graph = batch["graph"]
        output = protein_model(graph, graph.residue_feature.float())
        protein_feature = output["graph_feature"]
        protein_feature = protein_feature / protein_feature.norm(dim=-1, keepdim=True)
        zero_pred = logit_scale * protein_feature @ label_feature.t()
        supervised_pred = supervised_model.predict(batch)
        if ensemble_scheme == "logit":
            pred = zero_pred * alpha + supervised_pred
        elif ensemble_scheme == "prob":
            zero_pred = torch.softmax(zero_pred, dim=-1)
            supervised_pred = torch.softmax(supervised_pred, dim=-1)
            pred = zero_pred * alpha + supervised_pred
        else:
            raise ValueError("Unknown ensemble scheme `%s`" % ensemble_scheme)
        preds.append(pred)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    accuracy = (preds.argmax(dim=-1) == targets).float().mean().item()
    logger.warning("Ensemble accuracy: %.6f" % accuracy)

if __name__ == "__main__":
    args, vars = parse_args()
    args.supervised_config = os.path.realpath(args.supervised_config)
    supervised_cfg = util.load_config(args.supervised_config, context=vars)
    supervised_cfg.checkpoint = args.supervised_checkpoint
    args.zero_config = os.path.realpath(args.zero_config)
    zero_cfg = util.load_config(args.zero_config, context=vars)
    
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Supervised config file: %s" % args.supervised_config)
        logger.warning(pprint.pformat(supervised_cfg))
        logger.warning("\nZero-shot config file: %s" % args.zero_config)
        logger.warning(pprint.pformat(zero_cfg))

    device = torch.device("cuda:0")

    _dataset, test_dataloader, target_field = build_test_set(zero_cfg, logger)
    supervised_model = build_supervised_model(supervised_cfg, _dataset, logger, device)
    protein_model = build_protein_model(zero_cfg, logger, device)
    text_model = build_text_model(zero_cfg, logger, device)
    logit_scale = fetch_logit_scale(zero_cfg, logger, device)
    
    labels = fetch_labels(zero_cfg, logger)
    label_feature = label_embedding(zero_cfg, labels, text_model)
    ensemble_eval(zero_cfg, logger, device, test_dataloader, target_field,
                  supervised_model, protein_model, logit_scale, 
                  label_feature, args.ensemble_scheme, args.alpha)
