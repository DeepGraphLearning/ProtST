import os
import time
import logging
import argparse
import numpy as np

import yaml
import easydict
import jinja2
from jinja2 import meta

import torch
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm

def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def meshgrid(dict):
    if len(dict) == 0:
        yield {}
        return

    key = next(iter(dict))
    values = dict[key]
    sub_dict = dict.copy()
    sub_dict.pop(key)

    if not isinstance(values, list):
        values = [values]
    for value in values:
        for result in meshgrid(sub_dict):
            result[key] = value
            yield result


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    if context is not None:
        template = jinja2.Template(raw_text)
        instance = template.render(context)
        configs = easydict.EasyDict(yaml.safe_load(instance))
    else:
        configs = easydict.EasyDict(yaml.safe_load(raw_text))

    return configs


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger

def proceed_parser(parser):

    args, unparsed = parser.parse_known_args()
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}
    return args, vars

def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    if cfg.task["class"] in ["ProtST", "ProtSTMMP"]:
        output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"], cfg.dataset["class"],
                              cfg.task.protein_model["class"] + "_" + cfg.task.text_model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"], cfg.dataset["class"],
                              cfg.task.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    comm.synchronize()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
    if comm.get_rank() != 0:
        if local_rank is not None and comm.get_rank() != local_rank:
            pass
        else:
            with open(file_name, "r") as fin:
                output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    if local_rank is not None and comm.get_rank() != local_rank:
        pass
    else:
        os.chdir(output_dir)
    return output_dir
