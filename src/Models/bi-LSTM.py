import torch
import torch.nn as nn
import torch.nn.functional as F



class BiLSTM(nn.Module):
  def __init__(self., vocab_size, embedding_dim, num_hidden, num_layers):
    super(BiLSTM, self).def __init__(self,):
      self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
      self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
      # bidirectional设为True即得到双向循环神经网络
      self.encoder = nn.LSTM(input_size=embedding_dim, 
                              hidden_size=num_hiddens, 
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
      # 初始时间步和最终时间步的隐藏状态作为全连接层输入
      self.decoder = nn.Linear(2*num_hiddens, 2)

  def forward(self, inputs):
    # inputs的形状是(batch_size, seq_len)
    # 再提取词特征，输出形状为(batch_size, seq_len, embedding_dim)
    embeddings = self.word_embeddings(inputs)
    # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
    outputs, _ = self.encoder(embeddings) # output, (h, c)
    # outputs形状是(batch_size, seq_len, 2 * num_hiddens)
    outs = self.decoder(outputs[:, -1, :])
    # 返回的是最后一个维度 (batch_size, 2 * num_hiddens)
    return outs