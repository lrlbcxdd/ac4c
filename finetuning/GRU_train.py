import torch
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
from models import GRU
from datasets import DatasetDict, concatenate_datasets
import torch.utils.data as Data
from dataloader import test_loader
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import time

print("start")
print("build model")
device = torch.device("cuda", 3)
print(colored(f"{transformers.__version__}", "blue"))
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"



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

    for batch in data_iter:
        data = batch[0].to(device)
        labels = batch[1].to(device)
        output,x = net(data)
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


if __name__ == '__main__':
    index = 5

    train_loader , valid_loader , test_loader , max_len =  test_loader.load_ac4c_data(index)

    epoch_num = 50

    print('training...')

    # 初始化模型
    model = GRU.model(1001).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,weight_decay= 5e-4)

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping()
    # criterion = MarginLoss(0.9, 0.1, 0.5)
    best_acc = 0
    print("模型数据已经加载完成,现在开始模型训练。")

    for epoch in range(epoch_num):
        loss_ls = []
        t0 = time.time()
        model.train()
        for batch in train_loader:
            data = batch[0].to(device)
            labels = batch[1].to(device)

            output,x = model(data)
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, labels)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            loss_ls.append(loss.item())


        print('testing...(约2000秒(CPU))')

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
        if valid_acc > best_acc:
            best_acc = valid_acc
            test_performance, test_roc_data, test_prc_data, _ = evaluate(test_loader, model, criterion)
            # best_performance = valid_performance
            filename = '{}, {}[{:.4f}].pt'.format(f'model_GRU_model' + ', epoch[{}]'.format(epoch + 1), 'ACC',
                                                  test_performance[0])
            save_path_pt = os.path.join(f'Saved_Models/{index + 1}', filename)
            torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
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

        early_stopping(valid_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break




