import torch
import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .torch_gat import GAT
from .torch_gcn import GCN


class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model="roberta_base", nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model, nb_class=20, mix_factor=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.mix_factor = mix_factor
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
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
        gcn_logit = self.gcn(graph.ndata["cls_feats"], graph, graph.edata["edge_weight"])[idx]
        bert_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.mix_factor + bert_pred * (1 - self.mix_factor)
        pred = th.log(pred)
        # pred = cls_logit + gcn_logit
        return pred

    # def explain_forward(self, graph, target_id2, mask):
    def explain_forward(self, doc_feats, graph, target_id2, doc_mask):
        batch_size = int(len(doc_feats) / doc_mask.sum())

        cls_feats = graph.ndata["cls_feats"]

        gcn_logits = list()
        doc_feats = doc_feats.view(batch_size, doc_mask.sum(), -1)
        for i in range(batch_size):
            cls_feats_ = cls_feats.clone()
            cls_feats_[doc_mask] = doc_feats[i]
            gcn_logit = self.gcn(cls_feats_, graph, graph.edata["edge_weight"])[target_id2]
            gcn_logits.append(gcn_logit)

        pred = torch.stack(gcn_logits)
        return pred


class BertGAT(th.nn.Module):
    def __init__(
        self, pretrained_model="roberta_base", nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5
    ):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
            num_layers=gcn_layers - 1,
            in_dim=self.feat_dim,
            num_hidden=n_hidden,
            num_classes=nb_class,
            heads=[heads] * (gcn_layers - 1) + [1],
            activation=F.elu,
            feat_drop=dropout,
            attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata["input_ids"][idx], g.ndata["attention_mask"][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata["cls_feats"][idx] = cls_feats
        else:
            cls_feats = g.ndata["cls_feats"][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata["cls_feats"], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
