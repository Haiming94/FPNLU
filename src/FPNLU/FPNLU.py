# coding=utf-8

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from src.Models.Measurement import Measurement as mm
from src.Models.Linear import Linear
from src.Models.GlobalMixture import GlobalMixture as Mixture
from src.Models.KLDistance import KLDistance as KL

class FPNLU(BertPreTrainedModel):

  def __init__(self, config, num_labels, arg):
    super(FPNLU, self).__init__(config)
    self.num_labels = num_labels
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    self.mixture = Mixture()
    self.measurement = mm(config.hidden_size)

    self.kl = KL(arg)

    self.apply(self.init_bert_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    """
    Args:
      input_ids: 词对应的 id
      token_type_ids: 区分句子，0 为第一句，1表示第二句
      attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
    """
    encoded_layers, _ = self.bert(
      input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    # encoded_layers: [batch_size, seq_len, bert_dim=768]
    
    encoded_layers = self.dropout(encoded_layers)

    # encoded_layers = encoded_layers.permute(0, 2, 1)
    # encoded_layers: [batch_size, bert_dim=768, seq_len]

    mixture_layers = self.mixture(encoded_layers)
    # encoded_layers: [batch_size, bert_dim=768, bert_dim=768]

    mm_layers = self.measurement(mixture_layers)
    # encoded_layers: [batch_size, bert_dim=768]
    
    logits = self.kl(mm_layers)
    # logits: [batch_size, output_dim]

    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      return loss
    else:
      return logits