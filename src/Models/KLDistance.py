import torch
import torch.nn as nn

class KLDistance(nn.Module):

  def __init__(self, arg):
    super(KLDistance, self).__init__()
    self.softmax = nn.Softmax(dim=1)
    self.kl = nn.KLDivLoss(size_average=False, reduce=False)
    # self.label_w = nn.Parameter(torch.Tensor(len(arg.label_list), arg.hidden_size), requires_grad=True)
    # nn.init.xavier_uniform_(self.label_w)
    # nn.init.uniform_(self.label_w, -1, 1)

    self.W = nn.Linear(arg.hidden_size, 2)
    
    # self.label_w1 = nn.Parameter(torch.Tensor(arg.hidden_size), requires_grad=True)
    # self.label_w2 = nn.Parameter(torch.Tensor(arg.hidden_size), requires_grad=True)
    # nn.init.uniform_(self.label_w1, -1, 1)
    # nn.init.uniform_(self.label_w2, -1, 1)

    # self.PD = nn.PairwiseDistance(p=2)

    self.label_emb = nn.Embedding(len(arg.label_list), arg.hidden_size)

  def forward(self, inputs, labels):
    input_size = list(inputs.size())
    distances = []
    inputs = self.W(inputs)
    for i in range(self.label_emb.weight.shape[0]):
      target = self.W(self.label_emb.weight[i]).unsqueeze(0).expand(input_size[0], -1)
      distance = - torch.sum(self.kl(self.softmax(inputs), self.softmax(target)), 1)
      # distance = self.PD(self.softmax(inputs), self.softmax(target))
      distance = distance.unsqueeze(1)
      distances.append(distance)

    # target1 = self.label_w1.unsqueeze(0).expand(input_size[0], -1)
    # distance1 =  -1 * torch.mean(self.kl(self.softmax(inputs), self.softmax(target1)), 1)
    # distance1 = distance1.unsqueeze(1)

    # target2 = self.label_w2.unsqueeze(0).expand(input_size[0], -1)
    # distance2 =  -1 * torch.mean(self.kl(self.softmax(inputs), self.softmax(target2)), 1)
    # distance2 = distance2.unsqueeze(1)

    # distances = [distance1, distance2]
    logits = torch.cat(distances, 1)
    # return output
    
    label_dis = self.W(self.label_emb(labels.long()))
    loss = torch.sum(self.kl(self.softmax(inputs), self.softmax(label_dis)))
    # loss = torch.sum(self.PD(self.softmax(inputs), self.softmax(label_dis)))
    
    return loss, logits