import torch
import torch.nn as nn

class GlobalMixture(nn.Module):
  
  def __init__(self,):
    super(GlobalMixture, self).__init__()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs, ):
    """
    this is a function for global mixture matrix,
    input word-level representation [batch_size, length, dim]
    output sentence-level mixture matrix [batch_size, dim, dim]

    inputs: [batch_size, length, dim]
    output: [batch_size, dim, dim]
    """

    # origin_inputs = list(inputs.size())

    # input_a = inputs.view(-1, origin_inputs[-1])
    # input_b = input_a
    input_a = torch.unsqueeze(inputs, dim=-1)
    input_b = torch.unsqueeze(inputs, dim=-2)

    # tensor outer product
    outer = torch.matmul(input_a, input_b)  # [b, l, d, d]
    # outer = outer.view(origin_inputs[0], origin_inputs[1], origin_inputs[-1], origin_inputs[-1])

    # L2-nom
    position_weight = self.L2Norm(inputs)
    position_weight = torch.unsqueeze(torch.unsqueeze(position_weight, dim=-1), dim=-1)

    output = outer.float() * position_weight.float()
    output = torch.sum(output, dim=1)

    return output
  

  def L2Norm(self, inputs):
    output = torch.sqrt(0.00001 + torch.sum(inputs**2, dim=-1))
    return output


def test():
    mixture = GlobalMixture()
    a = torch.randn(10, 12, 6)
    b = torch.randn(10, 2)
    mix = mixture(a)
    print(mix.size(), mix)


if __name__ == '__main__':
    test()
