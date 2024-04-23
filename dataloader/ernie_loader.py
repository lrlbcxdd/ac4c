import pandas as pd
import torch
from models import ErniedataSet
from torch.utils.data import DataLoader
import numpy as np


def seq_to_index(sequences):
    '''
    input:
    sequences: list of string (difference length)

    return:
    rna_index: numpy matrix, shape like: [len(sequences), max_seq_len+2]
    rna_len_lst: list of length
    '''

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums, max_len + 2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j + 1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j + 1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j + 1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j + 1] = 6
            else:
                rna_index[i][j + 1] = 3
        rna_index[i][rna_len_lst[i] + 1] = 2  # add 'eos' token
    rna_index[:, 0] = 0  # add 'cls' token
    return rna_index, rna_len_lst

def process_data(sequences,length):
    new_sequences = []
    for sequence in sequences:
        center_point = len(sequence) // 2  # 计算序列的中心点位置
        new_seq = sequence[center_point - length:center_point + length + 1]  # 取中心点左侧length个字符
        new_sequences.append(new_seq)
    return new_sequences


def ernie_load_ac4c_data():
    train_file = r"/mnt/sdb/home/lrl/code/ac4c/data/new_test_data/nofold_data/train.csv"
    val_file = r"/mnt/sdb/home/lrl/code/ac4c/data/new_test_data/nofold_data/val.csv"
    test_file = r"/mnt/sdb/home/lrl/code/ac4c/data/new_test_data/nofold_data/test.csv"

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    train_seq, train_label = list(train_data['Sequence']), list(train_data['Label'])
    val_seq, val_label = list(val_data['Sequence']), list(val_data['Label'])
    test_seq, test_label = list(test_data['Sequence']), list(test_data['Label'])

    # 0 - 206
    # max_len = 1001

    half_length = 100

    train_seq = process_data(train_seq,half_length)
    test_seq = process_data(test_seq,half_length)
    val_seq = process_data(val_seq,half_length)

    train_seq, train_len = seq_to_index(train_seq)
    val_seq, val_len = seq_to_index(val_seq)
    test_seq, test_len = seq_to_index(test_seq)

    train_dataset = ErniedataSet.MyDataset(train_seq,train_label,train_len)
    val_dataset = ErniedataSet.MyDataset(val_seq, val_label,val_len)
    test_dataset = ErniedataSet.MyDataset(test_seq, test_label,test_len)

    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=16,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True,drop_last=True)

    return train_dataloader , val_dataloader ,test_dataloader


def load_ac4c_data(index):
    label_enc = {v:k for k,v in enumerate('NATCG')} # Z:0

    train_file0 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/train/{2*index}.csv"
    train_file1 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/train/{2*index+1}.csv"
    val_file0 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/val/{2*index}.csv"
    val_file1 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/val/{2*index+1}.csv"
    test_file0 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/test/{2*index}.csv"
    test_file1 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/test/{2*index+1}.csv"

    train_data0 = pd.read_csv(train_file0)
    train_data1 = pd.read_csv(train_file1)
    val_data0 = pd.read_csv(val_file0)
    val_data1 = pd.read_csv(val_file1)
    test_data0 = pd.read_csv(test_file0)
    test_data1 = pd.read_csv(test_file1)

    train_seq  = list(train_data0['data']) + (list(train_data1['data']))
    train_label = list(train_data0['label']) + (list(train_data1['label']))
    val_seq = list(val_data0['data'])+(list(val_data1['data']))
    val_label =  list(val_data0['label']) + list(val_data1['label'])
    test_seq  = list(test_data0['data'])+(list(test_data1['data']))
    test_label = list(test_data0['label'])  +  (list(test_data1['label']))

    half_length = 100

    train_seq = process_data(train_seq,half_length)
    test_seq = process_data(test_seq,half_length)
    val_seq = process_data(val_seq,half_length)

    # 0 - 206
    # max_len = 1001
    train_seq, train_len = seq_to_index(train_seq)
    val_seq, val_len = seq_to_index(val_seq)
    test_seq, test_len = seq_to_index(test_seq)

    train_dataset = ErniedataSet.MyDataset(train_seq,train_label,train_len)
    val_dataset = ErniedataSet.MyDataset(val_seq, val_label,val_len)
    test_dataset = ErniedataSet.MyDataset(test_seq, test_label,test_len)

    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True,drop_last=True)

    return train_dataloader , val_dataloader ,test_dataloader

