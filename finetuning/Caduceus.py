import torch
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForMaskedLM
import transformers
from datasets import DatasetDict, concatenate_datasets
import torch.utils.data as Data
import datasets
import random
import pandas as pd
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import time


print("start")
print("build model")

device = torch.device("cuda", 1)

print(colored(f"{transformers.__version__}", "blue"))

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"

checkpoint = '/mnt/sdb/home/lrl/code/git_Caduceus/caduceus-ps_seqlen-131k_d_model-256_n_layer-16'


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0


class Caduceus(nn.Module):
    def __init__(self, hidden_size=128,device = "cuda:1"):
        super(Caduceus, self).__init__()
        self.hidden_dim = 25
        self.emb_dim = 256
        self.device = device


        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)

        self.model = AutoModelForMaskedLM.from_pretrained(checkpoint,trust_remote_code=True)
        # self.GRU = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2, bidirectional=True,dropout=0.2)

        # self.block = nn.Sequential(
        #     nn.Linear(256, 64),  # 1001:128384  51:6784  510：65536
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(64,2)
        # )

        self.block = nn.Sequential(
            nn.Linear(256*1001, 1024),  # 1001:128384  51:6784  510：65536 GRU:50150  lsr:128512
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )


    def forward(self, batch_sentences):  # [batch_size,1]
        batch_sentences = list(batch_sentences)
        batch_sentences_partial = [seq for seq in batch_sentences] # 491 1511

        tokenized = self.tokenizer(batch_sentences_partial,truncation=True,return_tensors="pt")
        input_ids = tokenized['input_ids']
        output = self.model(input_ids=input_ids.to(self.device))
        hidden_state = output['hidden_states'][-1]
        hidden_state = hidden_state.to(self.block[0].weight.dtype)

        # 不使用GRU
        # hidden_state = hidden_state[:,1,:]
        hidden_state = hidden_state.view(hidden_state.shape[0],-1)

        # 用GRU
        # x = hidden_state.permute(1, 0, 2)  # torch.Size([128, 1001, 128])   torch.Size([32, 2003, 256])
        # output, hn = self.GRU(x)  # output: torch.Size([max_len, 128, hidden_dim*2]) hn: torch.Size([4, 128, 25])
        # output = output.permute(1, 0, 2)  # output: torch.Size([128, max_len, hidden_dim*2])
        # hn = hn.permute(1, 0, 2)  # torch.Size([128, 4, hidden_dim])
        #
        # output = output.reshape(output.shape[0], -1)  # torch.Size([128, max_len * hidden_dim *2])
        # hn = hn.reshape(output.shape[0], -1)  # torch.Size([128, 4* hidden_dim])
        # hidden_state = torch.cat([output, hn], 1)  # output:torch.Size([128, hidden_dim *(2*max_len + 4)])


        # 分类器
        output = self.block(hidden_state)   # torch.Size([32, 50150])   torch.Size([32, 100250])


        return output


# show some data in datasets randomly
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    # display(HTML(df.to_html()))
    print(colored(df, "blue"))


class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)

    def __getitem__(self, i):
        return self.data[i], self.label[i]

    def __len__(self):
        return len(self.label)


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


def evaluate(data_iter, net, criterion):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(data_iter, 0):
        labels = labels.to(device)
        output = net(data)
        loss = criterion(output, labels)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, loss


def to_log(log, index):
    with open(f"./results/{index + 1}/result_Mamba_model.log", "a+") as f:
        f.write(log + '\n')


# Note that load_metric has loaded the proper metric associated to your task, which is:
task_to_keys = {
    "cola": ("data", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

num_labels = 2
metric_name = "accuracy"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

if __name__ == '__main__':

    SEED = 42
    set_seed(seed= SEED)


    batchsize = 8

    index = 2

    max_len = 1001

    train_dataset_0 = DatasetDict.from_csv({'train': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/train/{2 * index}.csv'})
    train_dataset_1 = DatasetDict.from_csv({'train': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/train/{2 * index + 1}.csv'})
    train_dataset = concatenate_datasets([train_dataset_0['train'], train_dataset_1['train']])

    valid_dataset_0 = DatasetDict.from_csv({'valid': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/valid/{2 * index}.csv'})
    valid_dataset_1 = DatasetDict.from_csv({'valid': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/valid/{2 * index + 1}.csv'})
    valid_dataset = concatenate_datasets([valid_dataset_0['valid'], valid_dataset_1['valid']])

    test_dataset_0 = DatasetDict.from_csv({'test': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/test/{2 * index}.csv'})
    test_dataset_1 = DatasetDict.from_csv({'test': f'/mnt/sdb/home/lsr/Rm_LR_RNA_modification/data/reprocess/test/{2 * index + 1}.csv'})

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


    epoch_num = 50

    print('training...')

    # 初始化模型
    model = Caduceus().to(device)

    # model = DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 这里是定义损失函数，交叉熵损失函数比较常用解决分类问题
    # 依据你解决什么问题，选择什么样的损失函数
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping()
    # criterion = MarginLoss(0.9, 0.1, 0.5)
    best_acc = 0
    beat_test_acc = 0
    best_epoch = 0
    print("模型数据已经加载完成,现在开始模型训练。")

    for epoch in range(epoch_num):
        loss_ls = []
        t0 = time.time()
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            if len(data) <= 1:
                continue
            # labels = data['label']
            labels = labels.to(device)
            output = model(data)
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, labels)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            loss_ls.append(loss.item())


        print('testing')

        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, _ = evaluate(train_loader, model, criterion)
            valid_performance, valid_roc_data, valid_prc_data, valid_loss = evaluate(valid_loader, model, criterion)

        # results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        print(results)
        # to_log(results, index)
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]

        test_performance, test_roc_data, test_prc_data, _ = evaluate(test_loader, model, criterion)
        test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
            epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tAUC,\tPRE]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[1], test_performance[2], test_performance[3],
            test_performance[4], test_performance[5]) + '\n' + '=' * 60
        test_acc =test_performance[0]
        print(test_results)
        if valid_acc > best_acc:
            best_acc = valid_acc

            # best_performance = valid_performance
            filename = '{}, {}[{:.4f}].pt'.format(f'model_Caduceus_model'+ 'length[{}]'.format(max_len) +', epoch[{}]'.format(epoch + 1), 'ACC',
                                                  test_performance[0])
            save_path_pt = os.path.join(f'Saved_Models/{index + 1}', filename)
            torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

            # to_log(test_results, index)
            test_ROC = valid_roc_data
            test_PRC = valid_prc_data
            save_path_roc = os.path.join(f'ROC/{index + 1}', filename)
            save_path_prc = os.path.join(f'PRC/{index + 1}', filename)
            torch.save(test_roc_data, save_path_roc, _use_new_zipfile_serialization=False)
            torch.save(test_prc_data, save_path_prc, _use_new_zipfile_serialization=False)

        if test_acc > beat_test_acc:
            beat_test_acc = test_acc
            best_epoch = epoch
    print(beat_test_acc,best_epoch)
        # # early_stopping(valid_acc, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break




