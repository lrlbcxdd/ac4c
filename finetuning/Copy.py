import torch
from termcolor import colored
from transformers import AutoTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
import transformers
from datasets import load_dataset, load_metric, DatasetDict, concatenate_datasets, Dataset
import torch.utils.data as Data
import datasets
import random
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel
import evaluate
import numpy as np
import os
from evaluateMetric.glue import Glue
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import time
from torch.nn.utils.weight_norm import weight_norm

class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)

    def __getitem__(self, i):
        return self.data[i], self.label[i]

    def __len__(self):
        return len(self.label)

index = 0
batchsize = 8

train_dataset_0 = DatasetDict.from_csv(
{'train': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/train/{2 * index}.csv'})
train_dataset_1 = DatasetDict.from_csv(
{'train': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/train/{2 * index + 1}.csv'})
train_dataset = concatenate_datasets([train_dataset_0['train'], train_dataset_1['train']])

valid_dataset_0 = DatasetDict.from_csv(
{'valid': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/valid/{2 * index}.csv'})
valid_dataset_1 = DatasetDict.from_csv(
{'valid': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/valid/{2 * index + 1}.csv'})
valid_dataset = concatenate_datasets([valid_dataset_0['valid'], valid_dataset_1['valid']])

test_dataset_0 = DatasetDict.from_csv(
{'test': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/test/{2 * index}.csv'})
test_dataset_1 = DatasetDict.from_csv(
{'test': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/test/{2 * index + 1}.csv'})

test_dataset = concatenate_datasets([test_dataset_0['test'], test_dataset_1['test']])

dataset = DatasetDict()
dataset['train'] = train_dataset
dataset['test'] = test_dataset
dataset['valid'] = valid_dataset

trainDatas = RNADataset(train_dataset['data'], train_dataset['label'])
validDatas = RNADataset(valid_dataset['data'], valid_dataset['label'])
testDatas = RNADataset(test_dataset['data'], test_dataset['label'])

train_loader = Data.DataLoader(trainDatas, batch_size=batchsize, shuffle=True)
valid_loader = Data.DataLoader(validDatas, batch_size=batchsize, shuffle=True)
test_loader = Data.DataLoader(testDatas, batch_size=batchsize, shuffle=True)

for i, (data, labels) in enumerate(train_loader):
    batch_sentences_partial = [seq[491:1511] for seq in list(data)]
    for j in batch_sentences_partial:
        print(len(j.split()))