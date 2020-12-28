import torch
import torch.nn.functional as F
import torch.nn as nn

class Measurement(nn.Module):

  def __init__(self, hidden_size):
    super(Measurement, self).__init__()
    self.softmax = nn.Softmax(dim=1)
    self.operator = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    nn.init.uniform_(self.operator, 0, 1)

  
  def forward(self, inputs):
    """
    this is a measurement function

    inputs:[batch_size, dim, dim]
    output:[batch_size, dim]
    """
    mm = torch.matmul(self.operator, inputs)
    mm = torch.matmul(mm, self.operator.t())
    mm = torch.diagonal(mm, dim1=1, dim2=2)
    output = self.softmax(mm)

    return output
