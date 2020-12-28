# coding=utf-8
from bin import main


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 5,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


if __name__ == "__main__":

  model_name = "FPNLU" #  FPNLU  BertOrigin
  # task_list = ["cola", "mnli", "mrpc", "sst-2",
  #               "sts-b", "qqp", "qnli", "rte", "wnli"]
  task_list = ["cola"]
  task_label_nums = [glue_tasks_num_labels[t] for t in task_list]
  label_list = list(range(sum(task_label_nums)))
  label_list = [str(l) for l in label_list]
  data_dir = './data/glue_data/'
  output_dir = "./result/output/"
  cache_dir = "./result/cache/"
  log_dir = "./result/log/"

  # bert-base
  bert_vocab_file = "/Data/wuhaiming/data/bert_pretrain_base_cased/vocab.txt"
  bert_model_dir = "/Data/wuhaiming/data/bert_pretrain_base_cased"

  if model_name == "FPNLU":
    from bin.FPNLU import args
  elif model_name == "BertOrigin":
    from bin.BertOrigin import args

  config = args.get_args(data_dir, output_dir, cache_dir,
                          bert_vocab_file, bert_model_dir, log_dir)
  
  config.task_list = task_list

  # train
  config.do_train = True

  main(config, config.save_name, label_list)
