# coding=utf-8
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

from src.train_evalute_kl import evaluate, evaluate_save, train
from src.Utils.load_datatsets import load_data
from src.Utils.utils import get_device


def main(config, model_times):
  
  # 输出路径
  if not os.path.exists(config.output_dir + model_times):
    os.makedirs(config.output_dir + model_times)

  if not os.path.exists(config.cache_dir + model_times):
    os.makedirs(config.cache_dir + model_times)

  # Bert 模型输出文件
  output_model_file = os.path.join(config.output_dir, model_times, WEIGHTS_NAME)  
  output_config_file = os.path.join(config.output_dir, model_times, CONFIG_NAME)

  # 设备准备
  gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
  device, n_gpu = get_device(gpu_ids[0])  
  if n_gpu > 1:
    n_gpu = len(gpu_ids)

  config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

  # 设定随机种子 
  random.seed(config.seed)
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(config.seed)

  # 数据准备
  tokenizer = BertTokenizer.from_pretrained(
    config.bert_vocab_file, do_lower_case=config.do_lower_case)  # 分词器选择

  # num_labels = len(label_list)

  # Train and dev
  if config.do_train:

    train_dataloader, train_examples_len, label_list = load_data(
      config.task_list, config.data_dir, tokenizer, config.max_seq_length, config.train_batch_size, "train")
    dev_dataloader, _, label_list = load_data(
      config.task_list, config.data_dir, tokenizer, config.max_seq_length, config.dev_batch_size, "dev")
    
    num_train_optimization_steps = int(
      train_examples_len / config.train_batch_size / config.gradient_accumulation_steps) * config.num_train_epochs

    config.print_step = num_train_optimization_steps // config.num_train_epochs // 2
    print(config.print_step)

    num_labels = len(label_list)

    # 模型准备
    print("model name is {}".format(config.model_name))
    if config.model_name == "BertOrigin":
      from src.BertOrigin.BertOrigin import BertOrigin
      model = BertOrigin.from_pretrained(
        config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)

    elif config.model_name == "BertATT":
      from src.BertATT.BertATT import BertATT
      model = BertATT.from_pretrained(
          config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
    
    elif config.model_name == 'FPNLU':
      from src.FPNLU.FPNLU import FPNLU
      config.label_list = label_list
      model = FPNLU.from_pretrained(
        config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels, arg=config)
    elif config.model_name == 'FPNLUs':
      from src.FPNLUs.FPNLU import FPNLU
      config.label_list = label_list
      model = FPNLU.from_pretrained(
        config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels, arg=config)

    model.to(device)

    if n_gpu > 1:
      model = torch.nn.DataParallel(model,device_ids=gpu_ids)

    """ 优化器准备 """
    # param_optimizer = list(model.named_parameters())
    # param_optimizer = list(model.parameters())

    #     optimizer_grouped_parameters = [
    #     {'params': param_optimizer, 'weight_decay': 0.01},
    # ]

    """ 优化器准备 """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer_grouped_parameters_n = [
    #     {'params': [n for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [n for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=config.learning_rate,
                        warmup=config.warmup_proportion,
                        t_total=num_train_optimization_steps, 
                        eps=1e-8)

    """ 损失函数准备 """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train(config.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer, criterion, config.gradient_accumulation_steps, 
          device, label_list, output_model_file, output_config_file, config.log_dir, config.print_step, config.early_stop)

  # """ Test """

  # # test 数据
  # test_dataloader, _ = load_data(
  #   config.task_list, config.data_dir, tokenizer, config.max_seq_length, config.test_batch_size, "test", label_list)

  # # 加载模型 
  # bert_config = BertConfig(output_config_file)

  # if config.model_name == "BertOrigin":
  #   from src.BertOrigin.BertOrigin import BertOrigin
  #   model = BertOrigin(bert_config, num_labels=num_labels)
  # elif config.model_name == 'FPNLU':
  #   from src.FPNLU.FPNLU import FPNLU
  #   config.label_list = label_list
  #   model = FPNLU(bert_config, num_labels=num_labels, arg=config)

  # elif config.model_name == "BertATT":
  #   from src.BertATT.BertATT import BertATT
  #   model = BertATT(bert_config, num_labels=num_labels)

  # elif config.model_name == 'FPNLUs':
  #   from src.FPNLUs.FPNLU import FPNLU
  #   config.label_list = label_list
  #   model = FPNLU(bert_config, num_labels=num_labels, arg=config)


  # model.load_state_dict(torch.load(output_model_file))
  # model.to(device)

  # # 损失函数准备
  # criterion = nn.CrossEntropyLoss()
  # criterion = criterion.to(device)

  # # test the model
  # test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds = evaluate_save(
  #   model, test_dataloader, criterion, device, label_list)
  # print("-------------- Test -------------")
  # print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} % | AUC:{test_auc}')

  # for label in label_list:
  #   print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
  #       label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
  # print_list = ['macro avg', 'weighted avg']

  # for label in print_list:
  #   print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
  #       label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
