import pandas as pd
import random
neg_data = pd.read_csv("./data/all_data/neg.csv")
pos_data = pd.read_csv("./data/all_data/pos.csv")
fake_data = pd.read_csv("./data/all_data/minus_2/minus_pos.csv")

fake_seq = []
# sum = 0

for _,data in pos_data.iterrows():
    # letters = ['A', 'T', 'C', 'G']
    # random_letter = random.choice(letters)
    fake_seq.append(data['Sequence'][1:414])
    # if data['Sequence'][207] == 'C' and data['Label'] == 0:
    #     sum += 1
        # print(data['Sequence'][207])
# print(sum)

pos_data['Sequence'] = fake_seq
pos_data.to_csv("./data/all_data/minus_2/minus_pos.csv")

# def read_fasta_file(file_path):
#     sequences = []
#     headers = []
#
#     with open(file_path, 'r') as file:
#         sequence = ''
#         header = ''
#
#         for line in file:
#             line = line.strip()
#
#             if line.startswith('>'):
#                 if sequence != '':
#                     sequences.append(sequence)
#                     sequence = ''
#
#                 header = line[1:]
#                 headers.append(header)
#             else:
#                 sequence += line
#
#         sequences.append(sequence)
#
#     return headers, sequences
#
#
# # 使用示例
# fasta_file = './data/ac4c_fasta/neg.fasta'
# headers, sequences = read_fasta_file(fasta_file)
#
# # 输出头部信息和序列内容
# for i in range(len(headers)):
#     print(f'Header: {headers[i]}')
#     print(f'Sequence: {sequences[i][207]}')
