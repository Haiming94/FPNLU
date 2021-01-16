import os
import numpy as np


class InputExample(object):
  """单句子分类的 Example 类"""

  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class DataLoader():
  def __init__(self, opt):
    self.data_path = opt.data_path
    self.glove_path = opt.glove_path
    self.config = opt
    self.all_words = []

    data = []
    for file in [opt.data_path+'train.tsv', opt.data_path+'dev.tsv', opt.data_path+'test.tsv']:
      print(file)
      data_list = self.load_tsv_dataset(file)
      data.append(data_list)
    
    self.common_word = self.word_dict()
    embed = self.read_emb(opt.glove_path, self.common_word)

    for examples in data:
      exs = self.word_to_id(examples)

  def load_tsv_dataset(self, filename, ):
    dataList = []
    lines = self.read_tsv(filename)
    for (i, line) in enumerate(lines):
      guid = i+1
      text_a = line[0]
      text_a = self.text_process(text_a)
      label = int(line[1])
      dataList.append((guid, text_a, label))
    return dataList

  def read_tsv(self, filename):
    with open(filename, "r", encoding='utf-8') as f:
      reader = csv.reader(f, delimiter="\t")
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def text_process(self, text):
    text = re.split('\s|-',text.lower().strip())
		self.all_words += text
		return text

  def word_dict(self):
    counter=collections.Counter(self.all_words)	
		vocab=len(counter)
		vocab_size=vocab-2
		common_word=dict(counter.most_common(vocab_size))		

		c=2
		for key in common_word:
			common_word[key]=c
			c+=1
		print('c:',c)
		return common_word            

  def word_to_id(self, examples):
    examples_list = []
    label_list = []
    for example in examples:
      guid, text_a, label = example
      label_list.append(label)
      text_a = [self.common_word[w] if w in self.common_word else 1 for w in text_a]
      examples_list.append(InputExample(guid, text_a, label=label))
    return examples_list, label_list

  def read_emb(self, embed_path, word2id):
    word_to_embed  = {}
    with open(embed_path, 'r', ) as f:
      lines = f.readlines()
      for line in lines:
        line = line.split()
        word = line[0]
        embed = line[1:]
        embed = [float(num) for num in embed]
        word_to_embed[word] = embed
    
    embed = [np.random.normal(0, 0.1, dim).tolist(), np.random.normal(0, 0.1, dim).tolist()]

    missing = 0
    find = 0
    for n,w in sorted(zip(self.common_word.values(), self.common_word.keys())):
      try:
        embed.append(word_to_embed[w])
        find += 1
      except KeyError:
        embed.append(np.random.normal(0, 0.1, dim).tolist())
        missing += 1
    print("missing word {}. ".format(missing))
    return embed

    # word2id
    # id2word = {ix:w for w,ix in word2id.items()}
    # id2emb = {}
    # for ix in range(len(word2id)):
    #   if id2word[ix] in word_embed:
    #     id2emb[ix] = word_embed[id2word[ix]]
    #   else:
    #     id2emb[ix] = [0.0] * 100
    # data = [id2emb[ix] for ix in range(len(word2id))]
    # return data

# numpy_embed = get_numpy_word_embed(word2ix)
# embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed)).to('cuda')
