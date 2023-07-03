import csv
import math
from tqdm import tqdm
from collections import defaultdict

import torch
from torchdrug import utils
import torchdrug.data as td_data


class Protein(td_data.Protein):

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_residues = []
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_residue = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        view = graphs[0].view
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_residues.append(graph.num_residue)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = torch.where(v != -1, v + num_cum_node, -1)
                    elif type == "edge reference":
                        v = torch.where(v != -1, v + num_cum_edge, -1)
                    elif type == "residue reference":
                        v = torch.where(v != -1, v + num_cum_residue, -1)
                    elif type == "graph reference":
                        v = torch.where(v != -1, v + num_graph, -1)
                data_dict[k].append(v)
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_cum_residue += graph.num_residue
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        
        for k in data_dict.keys():
            if not any([type == "graph" for type in meta_dict[k]]):
                continue
            if not all(len(x.shape) == 2 for x in data_dict[k]):
                continue
            lengths = [x.shape[1] for x in data_dict[k]]
            max_length = max(lengths)
            if min(lengths) != max_length:
                for i in range(len(data_dict[k])):
                    data_dict[k][i] = torch.cat([data_dict[k][i], torch.ones(max_length - data_dict[k][i].shape[1], dtype=torch.long, device=data_dict[k][i].device).unsqueeze(0) * graph[0].pad_idx.item()], dim=-1)        

        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(edge_list, edge_weight=edge_weight, num_relation=graphs[0].num_relation,
                               num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=view,
                               meta_dict=meta_dict, **data_dict)


class PackedProtein(td_data.PackedProtein, Protein):
    unpacked_type = Protein
    _check_attribute = Protein._check_attribute
    
Protein.packed_type = PackedProtein

class ProteinDataset(td_data.ProteinDataset):

    def load_csv(self, csv_file, sequence_field="sequence", target_fields=None, verbose=0, **kwargs):
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            train, valid, test = [], [], []
            _sequences = []
            _targets = defaultdict(list)
            for i, values in enumerate(reader):
                for field, value in zip(fields, values):
                    if field == sequence_field:
                        _sequences.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        _targets[field].append(value)
                    elif field == "set":
                        if value == "train":
                            train.append(i)
                        elif value == "test":
                            test.append(i)
                    elif field == "validation":
                        if value == "True":
                            valid.append(i)

        valid_set = set(valid)
        sequences = [_sequences[i] for i in train if i not in valid_set] \
                    + [_sequences[i] for i in valid] \
                    + [_sequences[i] for i in test]
        targets = defaultdict(list)
        for key, value in _targets.items():
            targets[key] = [value[i] for i in train if i not in valid_set] \
                           + [value[i] for i in valid] \
                           + [value[i] for i in test]
        self.load_sequence(sequences, targets, verbose=verbose, **kwargs)
        self.num_samples = [len(train) - len(valid), len(valid), len(test)]