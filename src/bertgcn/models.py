import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .gcn_models.torch_gcn import GCN


class BertClassifier(nn.Module):
    """Basic BERT classifier for clinical text classification."""

    def __init__(self, pretrained_model="roberta_base", nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(nn.Module):
    """BertGCN hybrid model combining BERT and GCN for clinical text classification."""

    def __init__(
        self,
        pretrained_model,
        nb_class=20,
        mix_factor=0.7,
        gcn_layers=2,
        n_hidden=200,
        dropout=0.5,
    ):
        super(BertGCN, self).__init__()
        self.mix_factor = mix_factor
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers - 1,
            activation=F.elu,
            dropout=dropout,
        )

    def forward(self, graph, idx):
        input_ids = graph.ndata["input_ids"][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids)[0][:, 0]
            graph.ndata["cls_feats"][idx] = cls_feats
        else:
            cls_feats = graph.ndata["cls_feats"][idx]
        cls_logit = self.classifier(cls_feats)
        gcn_logit = self.gcn(
            graph.ndata["cls_feats"], graph, graph.edata["edge_weight"]
        )[idx]
        bert_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.mix_factor + bert_pred * (1 - self.mix_factor)
        pred = torch.log(pred)
        return pred

    def explain_forward(self, doc_feats, graph, target_id2, doc_mask, interpret_mode):
        """Forward pass for model interpretation."""
        batch_size = doc_feats.size(0)

        # Depending on if we come from ferret or directly from shap/captum:
        if len(doc_feats.shape) == 2:
            batch_size = int(len(doc_feats) / doc_mask.sum())
        else:
            batch_size = doc_feats.size(0)

        cls_feats = graph.ndata["cls_feats"]

        gcn_logits = list()
        logits = list()
        doc_feats = doc_feats.view(batch_size, doc_mask.sum(), -1)

        if interpret_mode == "gcn_bert":
            cls_logit = self.classifier(cls_feats[0])
            # we detach the results to follow gradients back only to the GCN
            cls_pred = torch.nn.Softmax(dim=-1)(cls_logit).detach()

        for i in range(batch_size):
            cls_feats_ = cls_feats.clone()
            cls_feats_[doc_mask] = doc_feats[i]
            gcn_logit = self.gcn(cls_feats_, graph, graph.edata["edge_weight"])[
                target_id2
            ]
            if interpret_mode == "gcn_only":
                gcn_logits.append(gcn_logit)
            elif interpret_mode == "gcn_bert":
                gcn_pred = torch.nn.Softmax(dim=-1)(gcn_logit)
                pred = (gcn_pred + 1e-10) * self.mix_factor + cls_pred * (
                    1 - self.mix_factor
                )
                logit = torch.log(pred)
                logits.append(logit)

        if interpret_mode == "gcn_only":
            pred = torch.stack(gcn_logits)
        elif interpret_mode == "gcn_bert":
            pred = torch.stack(logits)
        return pred
