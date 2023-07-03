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
from functools import partial

import torch

import torchdrug
from torchdrug import core, datasets, tasks, models, layers
from torchdrug.utils import comm, cuda

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from protst import dataset, model, task, util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="~/scratch/protein-datasets/",
                        help="dataset storing direction")
    return parser.parse_known_args()[0]

def prepare_all_datasets(dataset_dir):
    text_model = model.HuggingFaceModel()
    _, text_tokenizer_abs = text_model._build_from_huggingface("PubMedBERT-abs", local_files_only=False)
    _, text_tokenizer_full = text_model._build_from_huggingface("PubMedBERT-full", local_files_only=False)

    all_datasets = [partial(dataset.UniProtSeqText, dataset_dir, text_tokenizer_abs, atom_feature=None, bond_feature=None, seq_lazy=True, text_lazy=True),
                      partial(dataset.UniProtSeqText, dataset_dir, text_tokenizer_full, atom_feature=None, bond_feature=None, seq_lazy=True, text_lazy=True),
                      partial(dataset.Reaction, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.BinaryLocalization, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.SubcellularLocalization, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.BetaLactamase, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(dataset.AAV, dataset_dir, keep_mutation_region=True, atom_feature=None, bond_feature=None),
                      partial(dataset.Thermostability, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.Fluorescence, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.Stability, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.EnzymeCommission, dataset_dir, atom_feature=None, bond_feature=None),
                      partial(datasets.GeneOntology, dataset_dir, branch="BP", atom_feature=None, bond_feature=None),
                      partial(datasets.GeneOntology, dataset_dir, branch="MF", atom_feature=None, bond_feature=None),
                      partial(datasets.GeneOntology, dataset_dir, branch="CC", atom_feature=None, bond_feature=None)]

    for prep_dataset in all_datasets:
        _dataset = prep_dataset()
        print("\n", _dataset)
        
        if hasattr(_dataset, "split"):
            train_set, valid_set, test_set = _dataset.split()
            print("#train: %d, #valid: %d, #test: %d\n" % (len(train_set), len(valid_set), len(test_set)))
        del _dataset

if __name__ == "__main__":
    args = parse_args()
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    prepare_all_datasets(args.dataset_dir)
