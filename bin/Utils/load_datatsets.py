import os
import sys
import csv

from .Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors
from .Classifier_utils import glue_processors


"""
glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
"""

glue_tasks_num_labels = {
    "cola": 2,  # 0, 1
    "mnli": 3,  # 2, 3, 4
    "mrpc": 2,  # 5, 6
    "sst-2": 2, # 7, 8
    "sts-b": 5, # 9 10 11 12 13 
    "qqp": 2,   # 14 15
    "qnli": 2,  # 16 17
    "rte": 2,   # 18 19
    "wnli": 2,  # 20 21
}

glue_dir = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mnli-mm": "MNLI",
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
}


def read_tsv(filename):
    with open(filename, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def load_tsv_dataset(filename, set_type):
    """
    文件内数据格式: sentence  label
    """
    examples = []
    lines = read_tsv(filename)
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = i
        text_a = line[0]
        label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def load_data(task_list, data_dir, tokenizer, max_length, batch_size, data_type, label_list, format_type=0):
    
    if format_type == 0:
        load_func = load_tsv_dataset

    if data_type == "train":
        examples = []
        for t in task_list:
            print(t)
            Processor = glue_processors[t]()
            task_dir = os.path.join(data_dir, glue_dir[t])
            task_examples = Processor.get_train_examples(task_dir)
        examples += task_examples
    elif data_type == "dev":
        examples = []
        for t in task_list:
            Processor = glue_processors[t]()
            dev_dir = os.path.join(data_dir, glue_dir[t])
            dev_examples = Processor.get_dev_examples(dev_dir)
        examples += dev_examples
        # examples = load_func(dev_file, data_type)
    elif data_type == "test":
        examples = []
        for t in task_list:
            Processor = glue_processors[t]()
            test_dir = os.path.join(data_dir, glue_dir[t])
            test_examples = Processor.get_test_examples(test_dir)
        examples += test_examples
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)

    dataloader = convert_features_to_tensors(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len

