# coding=utf-8

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class FPNLU(BertPreTrainedModel):

  def __init__(self, config, num_labels, arg):
    super(FPNLU, self).__init__(config)
    self.num_labels = num_labels
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, num_labels)
    self.kernel = torch.nn.Parameter(torch.eye(arg.hidden_size))
    self.sigmoid = torch.nn.Sigmoid()

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
    # pooled_output: [batch_size, bert_dim=768]
    
    encoded_layers = self.dropout(encoded_layers)
    mixture_layers = self.GlobalMixture(encoded_layers)
    # encoded_layers: [batch_size, bert_dim=768, bert_dim=768]

    # mixture_layers = torch.bmm(pooled_output.unsqueeze(2), pooled_output.unsqueeze(1)) # [batch_size, bert_dim, bert_dim]
    mm_layers = self.RealMeasurement(mixture_layers)
    # encoded_layers: [batch_size, bert_dim=768]
    mm_layers = self.sigmoid(mm_layers)
    
    logits = self.classifier(mm_layers)

    return logits

  
  def KLDistance(self, inputs):
    input_size = list(inputs.size())
    distances = []
    for i in range(self.label_w.shape[0]):
      target = self.label_w[i].unsqueeze(0).expand(input_size[0], -1)
      distance = -1 * torch.mean(self.kl(self.softmax(inputs), self.softmax(target)), 1)
      distance = distance.unsqueeze(1)
      distances.append(distance)
    output = torch.cat(distances, 1)
    return output


  def L2Norm(self, inputs):
    output = torch.sqrt(0.00001 + torch.sum(inputs**2, dim=-1))
    return output


  def GlobalMixture(self, inputs):
    input_a = torch.unsqueeze(inputs, dim=-1)
    input_b = torch.unsqueeze(inputs, dim=-2)

    # tensor outer product
    outer = torch.matmul(input_a, input_b)  # [b, l, d, d]

    # L2-nom
    position_weight = self.L2Norm(inputs)
    position_weight = torch.unsqueeze(torch.unsqueeze(position_weight, dim=-1), dim=-1)

    output = outer.float() * position_weight.float()
    output = torch.sum(output, dim=1)

    return output


  def RealMeasurement(self, inputs, measure_operayor=None):
    kernel = self.kernel
    if measure_operayor is None:
      kernel = kernel.unsqueeze(-1)
    
    projrctor = torch.matmul(kernel, kernel.transpose(1, 2))
    output = torch.matmul(torch.flatten(inputs, start_dim=-2, end_dim=-1), torch.flatten(projrctor, start_dim=-2, end_dim=-1).t())
    
    return output