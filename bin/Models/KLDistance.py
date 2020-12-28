import torch
import torch.nn as nn

class KLDistance(nn.Module):

  def __init__(self, arg):
    super(KLDistance, self).__init__()
    self.softmax = nn.Softmax(dim=1)
    self.kl = nn.KLDivLoss(size_average=False, reduce=False)
    self.label_w = nn.Parameter(torch.Tensor(len(arg.label_list), arg.hidden_size))
    nn.init.uniform_(self.label_w, 0, 1)

  def forward(self, inputs, ):
    input_size = list(inputs.size())
    distances = []
    for target in self.label_w:
      target = target.unsqueeze(0).expand(input_size[0], -1)
      distance = 0 - torch.mean(self.kl(self.softmax(inputs), self.softmax(target)), 1)
      distance = distance.unsqueeze(1)
      distances.append(distance)
    
    output = torch.cat(distances, 1)

    return output
