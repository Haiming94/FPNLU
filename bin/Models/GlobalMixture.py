import torch
import torch.nn as nn

class GlobalMixture(nn.Module):

  def __init__(self,):
    super(GlobalMixture, self).__init__()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inputs):
    """
    this is a function for global mixture matrix,
    input word-level representation [batch_size, length, dim]
    output sentence-level mixture matrix [batch_size, dim, dim]

    inputs: [batch_size, length, dim]
    output: [batch_size, dim, dim]
    """

    origin_inputs = list(inputs.size())

    input_a = inputs.view(-1, origin_inputs[-1])
    input_b = input_a

    # tensor outer product
    outer = torch.bmm(input_a.unsqueeze(2), input_b.unsqueeze(1))
    outer = outer.view(origin_inputs[0], origin_inputs[1], origin_inputs[-1], origin_inputs[-1])

    # L2-nom
    inputs_norm = torch.norm(inputs, dim=2)
    position_weight = self.softmax(inputs_norm)

    # weight value of each position
    position_weight = position_weight.unsqueeze(2).unsqueeze(3)
    position_weight = position_weight.expand(origin_inputs[0], origin_inputs[1], origin_inputs[-1], origin_inputs[-1])

    output = torch.mul(outer, position_weight)
    output = torch.sum(output, dim=1).squeeze()

    return output


def test():
    mixture = GlobalMixture()
    a = torch.randn(10, 12, 6)
    b = torch.randn(10, 2)
    mix = mixture(a)
    print(mix.size(), mix)


# if __name__ == '__main__':
#     test()
