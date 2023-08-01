import os
import csv
import glob
from tqdm import tqdm
import lmdb
import pickle
from collections import defaultdict

import torch
from torch.utils import data as torch_data

import torchdrug.data as td_data
from torchdrug import utils
from torchdrug.core import Registry as R

from protst import data

@R.register("datasets.UniProtSeqText")
class UniProtSeqText(data.ProteinDataset):

    urls = {
        "swiss_prot": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/uniprotdata/uniprot_sprot_filtered.tsv",
        "trembl": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/uniprotdata/uniprot_sprot_tremblrep_2.5M.tsv"
    }
    md5s = {
        "swiss_prot": "3dedbad606e014a6bbd802245aae54de",
        "trembl": "b3c5f34a32e570b97658aef998812d00"
    }
    seq_field = "Sequence"
    text_fields = ["ProteinName", "Function", "SubcellularLocation", "Similarity"]
    text_field2acronym = {"ProteinName": "prot_name", "Function": "function",
                          "SubcellularLocation": "subloc", "Similarity": "similarity"}

    def __init__(self, path, text_tokenizer, version="swiss_prot", max_text_length=128,
                 verbose=1, seq_lazy=False, text_lazy=False, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.verbose = verbose
        self.seq_lazy = seq_lazy
        self.text_lazy = text_lazy

        url = self.urls[version]
        save_file = "uniprotqeqtext_%s_%s" % (version, os.path.basename(url))
        tsv_file = os.path.join(path, save_file)
        if not os.path.exists(tsv_file):
            tsv_file = utils.download(url, path, save_file=save_file, md5=self.md5s[version])
        #tsv_file = utils.download(url, path, md5=self.md5s[version])
        self.load_tsv(tsv_file, **kwargs)

    def load_tsv(self, tsv_file, transform=None, **kwargs):
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            if self.verbose:
                reader = iter(tqdm(reader, "Loading %s" % tsv_file, utils.get_line_count(tsv_file)))
            fields = next(reader)
            sequences = []
            texts = defaultdict(list)
            for i,values in enumerate(reader):
                for field, value in zip(fields, values[:len(fields)]):  # to handle rows with unexpected columns
                    if field == self.seq_field:
                        sequences.append(value)
                    elif field in self.text_fields:
                        if not self.text_lazy:
                            value = torch.tensor(self.text_tokenizer.encode(value, max_length=self.max_text_length, truncation=True, add_special_tokens=False))
                        texts[self.text_field2acronym[field]].append(value)
        if not self.seq_lazy and not self.text_lazy:
            attributes = texts
        else:
            attributes = None
        self.load_sequence(sequences, defaultdict(list), attributes=attributes, transform=transform,
                           lazy=self.seq_lazy, verbose=self.verbose, **kwargs)
        # register padding index when `text_lazy` is not used
        self.pad_idx = self.text_tokenizer.convert_tokens_to_ids(self.text_tokenizer._pad_token)
        if not self.seq_lazy:
            for i in range(len(self.data)):
                with self.data[i].graph():
                    self.data[i].pad_idx = torch.tensor(self.pad_idx)
        self.texts = texts

    def get_item(self, index):
        if self.seq_lazy:
            protein = data.Protein.from_sequence(self.sequences[index], **self.kwargs)
            with protein.graph():
                protein.pad_idx = torch.tensor(self.pad_idx)
        else:
            protein = self.data[index].clone()
        if self.text_lazy:
            attributes = {v: torch.tensor(self.text_tokenizer.encode(self.texts[v][index],
                                                                     max_length=self.max_text_length,
                                                                     truncation=True, add_special_tokens=False))
                          for v in self.text_field2acronym.values()}
        else:
            if self.seq_lazy:
                attributes = {v: self.texts[v][index] for v in self.text_field2acronym.values()}
            else:
                attributes = None

        if attributes is not None:
            with protein.graph():
                for k, v in attributes.items():
                    setattr(protein, k, v)
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    # Minghao: Since we use this dataset only for pretraining, we can remove the split function.
    #          We can send (dataset, None, None) as three splits to the solver.
    #          I will re-write a ``script/pretrain.py`` for pretraining runs.
    def split(self,):
        #lengths = [int(0.8 * len(self)), int(0.1 * len(self))]
        #lengths += [len(self) - sum(lengths)]
        lengths = [len(self), 0, 0]

        offset = 0
        splits = []
        for num_sample in lengths:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        
        #splits = torch_data.random_split(self, lengths)
        return splits


@R.register("datasets.Reaction")
class Reaction(data.ProteinDataset):
    """
    Reaction labels for a set of proteins determined by Enzyme Commission annotations.

    Statistics:
        - #Train: 29,215
        - #Valid: 2,562
        - #Test: 5,651

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/Reaction.zip"
    md5 = "d1c20341616ad3890349247017a34d68"
    splits = ["train", "valid", "test"]
    target_fields = ["label"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "Reaction/Reaction_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

@R.register("td_datasets.AAV")
class AAV(data.ProteinDataset):

    url = "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip"
    md5 = "cabdd41f3386f4949b32ca220db55c58"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]
    region = slice(474, 674)

    def __init__(self, path, split="two_vs_many", keep_mutation_region=True, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, 'aav')
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ['des_mut', 'low_vs_high', 'mut_des', 'one_vs_many', 'sampled', 'seven_vs_many', 'two_vs_many']

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)
        if keep_mutation_region:
            for i in range(len(self.data)):
                self.data[i] = self.data[i][self.region]
                self.sequences[i] = self.sequences[i][self.region]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

@R.register("td_datasets.Thermostability")
class Thermostability(data.ProteinDataset):

    url = "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip"
    md5 = "0f8b1e848568f7566713d53594c0ca90"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]

    def __init__(self, path, split="human_cell", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, 'thermostability')
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ['human', 'human_cell', 'mixed_split']

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

@R.register("td_datasets.EnzymeCommission")
class EnzymeCommission(data.ProteinDataset):
    """
    A set of proteins with their 3D structures and EC numbers, which describes their
    catalysis of biochemical reactions.

    Statistics (test_cutoff=0.95):
        - #Train: 15,011
        - #Valid: 1,664
        - #Test: 1,840

    Parameters:
        path (str): the path to store the dataset
        test_cutoff (float, optional): the test cutoff used to split the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://zenodo.org/record/6622158/files/EnzymeCommission.zip"
    md5 = "33f799065f8ad75f87b709a87293bc65"
    processed_file = "enzyme_commission.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

    def __init__(self, path, branch=None, test_cutoff=0.95, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if test_cutoff not in self.test_cutoffs:
            raise ValueError("Unknown test cutoff `%.2f` for EnzymeCommission dataset" % test_cutoff)
        self.test_cutoff = test_cutoff

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(utils.extract(zip_file), "EnzymeCommission")
        pkl_file = os.path.join(path, self.processed_file)

        csv_file = os.path.join(path, "nrPDB-EC_test.csv")
        pdb_ids = []
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin, delimiter=",")
            idx = self.test_cutoffs.index(test_cutoff) + 1
            _ = next(reader)
            for line in reader:
                if line[idx] == "0":
                    pdb_ids.append(line[0])

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in ["train", "valid", "test"]:
                split_path = utils.extract(os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        if len(pdb_ids) > 0:
            self.filter_pdb(pdb_ids)

        tsv_file = os.path.join(path, "nrPDB-EC_annot.tsv")
        pdb_ids = [os.path.basename(pdb_file).split("_")[0] for pdb_file in self.pdb_files]
        self.load_annotation(tsv_file, pdb_ids)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("valid"), splits.count("test")]

    def filter_pdb(self, pdb_ids):
        pdb_ids = set(pdb_ids)
        sequences = []
        pdb_files = []
        data = []
        for sequence, pdb_file, protein in zip(self.sequences, self.pdb_files, self.data):
            if os.path.basename(pdb_file).split("_")[0] in pdb_ids:
                continue
            sequences.append(sequence)
            pdb_files.append(pdb_file)
            data.append(protein)
        self.sequences = sequences
        self.pdb_files = pdb_files
        self.data = data

    def load_annotation(self, tsv_file, pdb_ids):
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            _ = next(reader)
            tasks = next(reader)
            task2id = {task: i for i, task in enumerate(tasks)}
            _ = next(reader)
            pos_targets = {}
            for pdb_id, pos_target in reader:
                pos_target = [task2id[t] for t in pos_target.split(",")]
                pos_target = torch.tensor(pos_target)
                pos_targets[pdb_id] = pos_target

        # fake targets to enable the property self.tasks
        self.targets = task2id
        self.pos_targets = []
        for pdb_id in pdb_ids:
            self.pos_targets.append(pos_targets[pdb_id])

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        indices = self.pos_targets[index].unsqueeze(0)
        values = torch.ones(len(self.pos_targets[index]))
        item["targets"] = utils.sparse_coo_tensor(indices, values, (len(self.tasks),)).to_dense()
        return item