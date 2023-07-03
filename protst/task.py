import copy
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop
from torch_scatter import scatter_min

from torchdrug import core, layers, models, tasks, metrics
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from protst import model


@R.register("tasks.ProtST")
class ProtST(tasks.Task, core.Configurable):

    def __init__(self, protein_model, text_model, protein2text=True, text2protein=True, mlm_weight=0,
                 mask_rate=0.15, num_mlp_layer=2, activation="relu", global_contrast=False):
        super(ProtST, self).__init__()
        self.protein_model = protein_model
        self.text_model = text_model
        self.protein2text = protein2text
        self.text2protein = text2protein
        self.mlm_weight = mlm_weight
        self.mask_rate = mask_rate
        self.global_contrast = global_contrast

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if mlm_weight > 0:
            self.mlm_head = layers.MLP(protein_model.output_dim,
                                       [protein_model.output_dim] * (num_mlp_layer - 1) + [constant.NUM_AMINO_ACID])

    def mask_protein(self, graph):
        num_samples = (graph.num_residues * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        residue_index = (torch.rand(num_sample, device=self.device) * graph.num_residues[sample2graph]).long()
        residue_index = residue_index + (graph.num_cum_residues - graph.num_residues)[sample2graph]

        mlm_target = graph.residue_type[residue_index]
        if isinstance(self.protein_model, models.ESM):
            mask_id = self.protein_model.alphabet.get_idx("<mask>")
        elif isinstance(self.protein_model, model.PretrainProtBert):
            mask_id = self.protein_model.tokenizer.mask_token_id
        else:
            mask_id = 0
        with graph.residue():
            graph.residue_feature[residue_index] = 0
            graph.residue_type[residue_index] = mask_id

        return graph, mlm_target, residue_index

    def alignment_pred_and_target(self, protein_output, text_output):
        protein_feature = protein_output["graph_feature"]
        protein_feature = protein_feature / protein_feature.norm(dim=-1, keepdim=True)
        text_feature = text_output["text_feature"]
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        if self.global_contrast:
            local_batch_size = protein_feature.shape[0]
            all_protein_feature = all_gather_with_backprop(protein_feature)
            all_protein_feature = torch.cat(all_protein_feature, dim=0)
            all_text_feature = all_gather_with_backprop(text_feature)
            all_text_feature = torch.cat(all_text_feature, dim=0)
            pred_protein2text = logit_scale * protein_feature @ all_text_feature.t()
            pred_text2protein = logit_scale * text_feature @ all_protein_feature.t()
            target_protein2text = local_batch_size * comm.get_rank() + torch.arange(
                local_batch_size, device=pred_protein2text.device)
            target_text2protein = local_batch_size * comm.get_rank() + torch.arange(
                local_batch_size, device=pred_text2protein.device)
        else:
            pred_protein2text = logit_scale * protein_feature @ text_feature.t()
            pred_text2protein = pred_protein2text.t()
            target_protein2text = torch.arange(protein_feature.shape[0], device=pred_protein2text.device)
            target_text2protein = torch.arange(text_feature.shape[0], device=pred_text2protein.device)

        preds = {"protein2text": pred_protein2text, "text2protein": pred_text2protein}
        targets = {"protein2text": target_protein2text, "text2protein": target_text2protein}
        return preds, targets

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        # mask partial residue tokens if MLM is applied
        if self.mlm_weight > 0:
            graph, mlm_target, residue_index = self.mask_protein(graph)

        protein_output = self.protein_model(graph, graph.residue_feature.float(), all_loss, metric)
        text_output = self.text_model(graph, all_loss, metric)

        # prediction and target for protein-text alignment
        preds, targets = self.alignment_pred_and_target(protein_output, text_output)

        # prediction and target for MLM
        if self.mlm_weight > 0:
            residue_feature = protein_output["residue_feature"]
            residue_feature = residue_feature[residue_index]
            mlm_pred = self.mlm_head(residue_feature)
            preds["mlm"] = mlm_pred
            targets["mlm"] = mlm_target

        return preds, targets

    def evaluate(self, pred, target):
        metric = {}
        acc_name = tasks._get_metric_name("acc")
        if self.protein2text:
            accuracy = (pred["protein2text"].argmax(dim=-1) == target["protein2text"]).float().mean()
            metric["Protein2Text." + acc_name] = accuracy
        if self.text2protein:
            accuracy = (pred["text2protein"].argmax(dim=-1) == target["text2protein"]).float().mean()
            metric["Text2Protein." + acc_name] = accuracy
        if self.mlm_weight > 0:
            accuracy = (pred["mlm"].argmax(dim=-1) == target["mlm"]).float().mean()
            metric["MLM." + acc_name] = accuracy

        return metric

    def get_loss(self, pred, target, all_loss, metric):
        ce_name = tasks._get_criterion_name("ce")
        if self.protein2text:
            loss = F.cross_entropy(pred["protein2text"], target["protein2text"])
            metric["Protein2Text." + ce_name] = loss
            all_loss += loss
        if self.text2protein:
            loss = F.cross_entropy(pred["text2protein"], target["text2protein"])
            metric["Text2Protein." + ce_name] = loss
            all_loss += loss
        if self.mlm_weight > 0:
            loss = F.cross_entropy(pred["mlm"], target["mlm"])
            metric["MLM." + ce_name] = loss
            all_loss += loss * self.mlm_weight

        return all_loss, metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))
        all_loss, metric = self.get_loss(pred, target, all_loss, metric)

        return all_loss, metric


@R.register("tasks.ProtSTMMP")
class ProtSTMMP(ProtST):

    def __init__(self, protein_model, text_model, fusion_model, protein2text=True, text2protein=True, mlm_weight=0,
                 mask_rate=0.15, num_mlp_layer=2, activation="relu", global_contrast=False, mmp_weight=0):
        super(ProtSTMMP, self).__init__(protein_model, text_model, protein2text, text2protein, mlm_weight,
                                          mask_rate, num_mlp_layer, activation, global_contrast)
        self.fusion_model = fusion_model
        self.mmp_weight = mmp_weight

        if mmp_weight > 0:
            self.mmp_protein_head = layers.MLP(fusion_model.hidden_dim, [fusion_model.hidden_dim]
                                               * (num_mlp_layer - 1) + [constant.NUM_AMINO_ACID])
            self.mmp_text_head = layers.MLP(fusion_model.hidden_dim, [fusion_model.hidden_dim]
                                            * (num_mlp_layer - 1) + [text_model.tokenizer.vocab_size])

    def mask_text(self, input_ids):
        mask = torch.rand_like(input_ids.to(dtype=torch.float32)) < self.mask_rate
        is_special = (input_ids == self.text_model.cls_idx) | (input_ids == self.text_model.sep_idx) | \
                     (input_ids == self.text_model.pad_idx)
        mask = mask & (~is_special)
        text_target = input_ids[mask]
        input_ids[mask] = self.text_model.mask_idx
        return input_ids, text_target, mask

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        # mask partial residue tokens if MLM or MMP is applied
        if self.mlm_weight > 0 or self.mmp_weight > 0:
            graph, protein_target, residue_index = self.mask_protein(graph)
        # mask partial word tokens if MMP is applied
        input_ids, attention_mask = self.text_model._combine_attributes(graph)
        if self.mmp_weight > 0:
            input_ids, text_target, text_mask = self.mask_text(input_ids)

        protein_output = self.protein_model(graph, graph.residue_feature.float(), all_loss, metric)
        text_output = self.text_model(graph, all_loss, metric, input_ids, attention_mask)
        fusion_output = self.fusion_model(graph, protein_output["residue_feature"], text_output["word_feature"],
                                          attention_mask, all_loss, metric)

        # prediction and target for protein-text alignment
        preds, targets = self.alignment_pred_and_target(protein_output, text_output)

        # prediction and target for MLM
        if self.mlm_weight > 0:
            residue_feature = protein_output["residue_feature"]
            residue_feature = residue_feature[residue_index]
            mlm_pred = self.mlm_head(residue_feature)
            preds["mlm"] = mlm_pred
            targets["mlm"] = protein_target

        # prediction and target for multimodal mask prediction
        if self.mmp_weight > 0:
            residue_feature = fusion_output["residue_feature"]
            residue_feature = residue_feature[residue_index]
            mmp_protein_pred = self.mmp_protein_head(residue_feature)
            preds["mmp_protein"] = mmp_protein_pred
            targets["mmp_protein"] = protein_target

            word_feature = fusion_output["word_feature"]
            word_feature = word_feature[text_mask]
            mmp_text_pred = self.mmp_text_head(word_feature)
            preds["mmp_text"] = mmp_text_pred
            targets["mmp_text"] = text_target

        return preds, targets

    def evaluate(self, pred, target):
        metric = {}
        metric.update(super(ProtSTMMP, self).evaluate(pred, target))

        acc_name = tasks._get_metric_name("acc")
        if self.mmp_weight > 0:
            accuracy = (pred["mmp_protein"].argmax(dim=-1) == target["mmp_protein"]).float().mean()
            metric["MMP.Protein." + acc_name] = accuracy
            accuracy = (pred["mmp_text"].argmax(dim=-1) == target["mmp_text"]).float().mean()
            metric["MMP.Text." + acc_name] = accuracy

        return metric

    def get_loss(self, pred, target, all_loss, metric):
        ce_name = tasks._get_criterion_name("ce")
        if self.mmp_weight > 0:
            loss = F.cross_entropy(pred["mmp_protein"], target["mmp_protein"])
            metric["MMP.Protein." + ce_name] = loss
            all_loss += loss * self.mmp_weight
            loss = F.cross_entropy(pred["mmp_text"], target["mmp_text"])
            metric["MMP.Text." + ce_name] = loss
            all_loss += loss * self.mmp_weight

        return all_loss, metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))
        all_loss, metric = super(ProtSTMMP, self).get_loss(pred, target, all_loss, metric)
        all_loss, metric = self.get_loss(pred, target, all_loss, metric)

        return all_loss, metric
