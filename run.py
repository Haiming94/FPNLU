# coding=utf-8
from src import main



if __name__ == "__main__":

  
  model_name = "BertOrigin" #  FPNLU  BertOrigin   FPNLUs   BertATT
  # task_list = ["cola", "mnli", "mrpc", "sst-2",
  #               "sts-b", "qqp", "qnli", "rte", "wnli"]
  task_list = ["cola", ]
  data_dir = './data/glue_data/'
  output_dir = "./result/output/"
  cache_dir = "./result/cache/"
  log_dir = "./result/log/"

  # bert-base
  bert_vocab_file = "/Data/wuhaiming/data/bert_pretrain_base_cased/vocab.txt"
  bert_model_dir = "/Data/wuhaiming/data/bert_pretrain_base_cased"

  if model_name == "FPNLU":
    from src.FPNLU import args
  elif model_name == "BertOrigin":
    from src.BertOrigin import args
  elif model_name == "BertATT":
        from src.BertATT import args
  elif model_name == "FPNLUs":
    from src.FPNLUs import args

  config = args.get_args(data_dir, output_dir, cache_dir,
                          bert_vocab_file, bert_model_dir, log_dir)
  
  config.task_list = task_list

  # train
  config.do_train = True

  main(config, config.save_name)
