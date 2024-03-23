import pandas as pd
import torch
from models import MydataSet
from torch.utils.data import DataLoader


def process_data(sequences,length):
    new_sequences = []
    for sequence in sequences:
        center_point = len(sequence) // 2  # 计算序列的中心点位置
        new_seq = sequence[center_point - length:center_point + length + 1]  # 取中心点左侧40个字符
        new_sequences.append(new_seq)
    return new_sequences


def load_ac4c_data():
    label_enc = {v:k for k,v in enumerate('ZATCG')} # Z:0

    train_file = r"/mnt/sdb/home/lrl/code/ac4c/data/no_fold/train.csv"
    val_file = r"/mnt/sdb/home/lrl/code/ac4c/data/no_fold/val.csv"
    test_file = r"/mnt/sdb/home/lrl/code/ac4c/data/no_fold/test.csv"

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    train_seq , train_label = list(train_data['Sequence']) , list(train_data['Label'])
    val_seq, val_label = list(val_data['Sequence']), list(val_data['Label'])
    test_seq, test_label = list(test_data['Sequence']), list(test_data['Label'])

    # 0 - 206
    half_length = 200

    train_seq = process_data(train_seq,half_length)
    test_seq = process_data(test_seq,half_length)
    val_seq = process_data(val_seq,half_length)


    train_seq = [torch.tensor([label_enc[i] for i in seq])for seq in train_seq]
    val_seq = [torch.tensor([label_enc[i] for i in seq])for seq in val_seq]
    test_seq = [torch.tensor([label_enc[i] for i in seq])for seq in test_seq]

    train_dataset = MydataSet.MyDataSet(train_seq,train_label)
    val_dataset = MydataSet.MyDataSet(val_seq, val_label)
    test_dataset = MydataSet.MyDataSet(test_seq, test_label)

    train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=True)

    return train_dataloader , val_dataloader ,test_dataloader , half_length * 2 + 1

