import csv
import random
import pandas as pd

# def read_fasta_file(file_path):
#     sequences = []
#     headers = []
#
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#         for i in range(0,len(lines),2):
#             line = lines[i].strip()
#
#             if line.startswith('>'):
#                 header = line[1:]
#                 headers.append(header)
#                 sequences.append(lines[i+1].strip())
#
#     return headers, sequences


def write_to_csv(sequences):
    data = []
    for sequence in sequences:
        entry = [sequence, 0]
        data.append(entry)
    with open('./all_data/minus_2/balance_output.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


# fasta_file = 'D:\课题\\final_fw\\neg.fasta'
# headers, sequences = read_fasta_file(fasta_file)

fake_data = pd.read_csv("./all_data/minus_2/minus_neg.csv")
sequences = list(fake_data['Sequence'])
sequences = random.sample(sequences,1850)
write_to_csv(sequences)


def split_dataset_nofold(sequences, labels, train_ratio, val_ratio, test_ratio):
    # 根据标签将数据分为正例和负例
    positive_sequences = [sequences[i] for i, label in enumerate(labels) if label == 1]
    negative_sequences = [sequences[i] for i, label in enumerate(labels) if label == 0]

    positive_size = len(positive_sequences)
    negative_size = len(negative_sequences)

    # 计算划分后每个集合的样本数量
    train_positive_size = int(positive_size * train_ratio)
    train_negative_size = int(negative_size * train_ratio)
    val_positive_size = int(positive_size * val_ratio)
    val_negative_size = int(negative_size * val_ratio)
    test_positive_size = positive_size - train_positive_size - val_positive_size
    test_negative_size = negative_size - train_negative_size - val_negative_size

    # 随机选取正例和负例样本的索引
    positive_indices = list(range(positive_size))
    negative_indices = list(range(negative_size))
    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    # 划分训练集、验证集和测试集的索引
    train_positive_indices = positive_indices[:train_positive_size]
    train_negative_indices = negative_indices[:train_negative_size]
    val_positive_indices = positive_indices[train_positive_size:(train_positive_size + val_positive_size)]
    val_negative_indices = negative_indices[train_negative_size:(train_negative_size + val_negative_size)]
    test_positive_indices = positive_indices[(train_positive_size + val_positive_size):]
    test_negative_indices = negative_indices[(train_negative_size + val_negative_size):]

    # 根据索引获取划分后的数据
    train_data = [(positive_sequences[i], 1) for i in train_positive_indices] + \
                 [(negative_sequences[i], 0) for i in train_negative_indices]
    val_data = [(positive_sequences[i], 1) for i in val_positive_indices] + \
               [(negative_sequences[i], 0) for i in val_negative_indices]
    test_data = [(positive_sequences[i], 1) for i in test_positive_indices] + \
                [(negative_sequences[i], 0) for i in test_negative_indices]

    return train_data, val_data, test_data

# def split_dataset(sequences, labels, train_ratio, test_ratio):
#     positive_samples = [(seq, label) for seq, label in zip(sequences, labels) if label == 1]
#     negative_samples = [(seq, label) for seq, label in zip(sequences, labels) if label == 0]
#
#     random.shuffle(positive_samples)
#     random.shuffle(negative_samples)
#
#     train_positive_size = int(train_ratio * len(positive_samples))
#     train_negative_size = int(train_ratio * len(negative_samples))
#
#     train_data = positive_samples[:train_positive_size] + negative_samples[:train_negative_size]
#     test_data = positive_samples[train_positive_size:] + negative_samples[train_negative_size:]
#
#     random.shuffle(train_data)
#     random.shuffle(test_data)
#
#     return train_data, test_data


def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Label'])
        writer.writerows(data)

# 使用示例 - 假设sequences是包含一万多个元素的序列，labels是对应的标签
data_set = pd.read_csv("./all_data/minus_2/balance_output.csv")
sequences = list(data_set['Sequence'])  # 序列数据
labels = list(data_set['Label'])  # 标签数据

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_data, val_data, test_data = split_dataset_nofold(sequences, labels, train_ratio, val_ratio, test_ratio)
# # train_data, test_data = split_dataset(sequences,labels,train_ratio,test_ratio)
#
#
write_to_csv(train_data, './no_fold/train.csv')
write_to_csv(val_data, './no_fold/val.csv')
write_to_csv(test_data, './no_fold/test.csv')