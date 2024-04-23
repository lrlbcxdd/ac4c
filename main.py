import os

from sklearn.metrics import roc_curve, auc, roc_auc_score
from termcolor import colored

from models import Transformer_GRU,GRU
import torch
from datasets import DatasetDict, concatenate_datasets
import torch.utils.data as Data
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import time

class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)

    def __getitem__(self, i):
        return self.data[i], self.label[i]

    def __len__(self):
        return len(self.label)

def evaluate(data_iter, net, criterion):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(data_iter, 0):
        labels = labels.to(device)
        output , features = net(data)
        loss = criterion(output, labels)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, loss

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


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    all_logits, all_labels = [], []

    for data in data_iter:
        x = data[0].to(device)
        y = data[1].to(device)

        logits , features = net(data)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        acc_sum += (logits.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_logits[:, 1])  # 假设第二列是正例的概率值

    return acc_sum / n, auc


def to_log(log):
    with open("./results/ExamPle_Log.log", "a+") as f:
        f.write(log + '\n')

device = torch.device("cuda", 3)
criterion_CE = nn.CrossEntropyLoss()
# train_iter , val_iter , test_iter , max_len =  test_loader.load_ac4c_data()


label_enc = {v: k for k, v in enumerate('NATCG')}  # Z:0
index = 5
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


train_seq = [torch.tensor([label_enc[i] for i in seq.split()]) for seq in train_dataset['data']]
val_seq = [torch.tensor([label_enc[i] for i in seq.split()]) for seq in valid_dataset['data']]
test_seq = [torch.tensor([label_enc[i] for i in seq.split()]) for seq in test_dataset['data']]

trainDatas = RNADataset(train_seq, train_dataset['label'])
validDatas = RNADataset(val_seq, valid_dataset['label'])
testDatas = RNADataset(test_seq, test_dataset['label'])

train_loader = Data.DataLoader(trainDatas, batch_size=8, shuffle=True)
valid_loader = Data.DataLoader(validDatas, batch_size=8, shuffle=True)
test_loader = Data.DataLoader(testDatas, batch_size=8, shuffle=True)

if __name__ == '__main__':
    seed = 40

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # net = Another_Transformer.Transformer(1001).to(device)
    # net = Transformer_GRU.model(1001).to(device)
    # net = TextCNN.CnnModel().to(device)
    net = GRU.model(1001).to(device)

    lr =0.001
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay= 5e-4)

    best_acc = 0
    EPOCH = 100


    for epoch in range(EPOCH):
        loss_list = []
        t0 = time.time()
        net.train()

        for batch in train_loader:
            data = batch[0].to(device)
            label = batch[1].to(device)

            logits , features = net(data) # logits:(128,2) features(128,128)

            ce_loss = criterion_CE(logits.view(-1,2),label).mean()

            train_loss = ce_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            loss_list.append(train_loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, _ = evaluate(train_loader, net, criterion_CE)
            valid_performance, valid_roc_data, valid_prc_data, valid_loss = evaluate(valid_loader, net, criterion_CE)

        # results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_list):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        print(results)
        # to_log(results, index)
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        if valid_acc > best_acc:
            best_acc = valid_acc
            test_performance, test_roc_data, test_prc_data, _ = evaluate(test_loader, net, criterion_CE)
            # best_performance = valid_performance
            filename = '{}, {}[{:.4f}].pt'.format(f'model_Transformer_model' + ', epoch[{}]'.format(epoch + 1), 'ACC',
                                                  test_performance[0])
            save_path_pt = os.path.join(f'Saved_Models/{index + 1}', filename)
            torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
            test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tAUC,\tPRE]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                test_performance[4], test_performance[5]) + '\n' + '=' * 60
            print(test_results)
            # to_log(test_results, index)
            test_ROC = valid_roc_data
            test_PRC = valid_prc_data
            save_path_roc = os.path.join(f'ROC/{index + 1}', filename)
            save_path_prc = os.path.join(f'PRC/{index + 1}', filename)
            torch.save(test_roc_data, save_path_roc, _use_new_zipfile_serialization=False)
            torch.save(test_prc_data, save_path_prc, _use_new_zipfile_serialization=False)