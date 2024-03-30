import pandas as pd


# 定义函数来将 CSV 数据转换为 FASTA 格式
def csv_to_fasta(csv_file, fasta_file):
    # 读取 CSV 文件
    csv_data = pd.read_csv(csv_file)

    # 打开 FASTA 文件用于写入
    with open(fasta_file, 'a') as fasta_file:
        # 遍历 CSV 数据
        num = 0
        for index, row in csv_data.iterrows():
            sequence_id = row['idx']  # 假设 CSV 文件中有一个列名为 'ID'，用于存储序列标识
            sequence_data = row['data']  # 假设 CSV 文件中有一个列名为 'Sequence'，用于存储序列数据
            if len(sequence_data) != 1001 : continue

            # 将数据写入 FASTA 文件
            fasta_file.write(f'>{sequence_id}\n')  # 写入序列标识
            fasta_file.write(f'{sequence_data}\n')  # 写入序列数据
            num += 1

# for i in range(2,24):
name = '15'

# 将 train.csv 转换为 FASTA 格式
csv_to_fasta(f'/mnt/8t/qjb/workspace/cl-rna/store/datasets/reprocess/train/{name}.csv', f'/mnt/sdb/home/lrl/code/ac4c/fasta/{name}.fasta')

# 将 test.csv 转换为 FASTA 格式
csv_to_fasta(f'/mnt/8t/qjb/workspace/cl-rna/store/datasets/reprocess/test/{name}.csv', f'/mnt/sdb/home/lrl/code/ac4c/fasta/{name}.fasta')

# 将 val.csv 转换为 FASTA 格式
csv_to_fasta(f'/mnt/8t/qjb/workspace/cl-rna/store/datasets/reprocess/valid/{name}.csv', f'/mnt/sdb/home/lrl/code/ac4c/fasta/{name}.fasta')
