# coding=utf-8
import os
import random
import time

from src.Models.bi-LSTM import *
from src.Models.gloveVector import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id)
                          if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


def main(config, model_times):

  # output path
  if not os.path.exists(config.output_dir + model_times):
    os.makedirs(config.output_dir + model_times)

  # load embedding
  glove = GloveVector(config.glove_path, config.glove_dim)
  glove_emb = glove.reader()

  # gpu
  gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
  device, n_gpu = get_device(gpu_ids[0])
  if n_gpu > 1:
      n_gpu = len(gpu_ids)

  config.train_batch_size = 1

  # seed
  random.seed(config.seed)
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(config.seed)
  
  # Train & dev
  if config.do_train:
    train_dataloader, train_examples_len, label_list = load_data(
      config.task_list, config.data_dir, tokenizer, config.max_seq_length, config.train_batch_size, "train")
    dev_dataloader = 1

    print("model name is {} .".format(config.model_name))
    model = models.setup(config)