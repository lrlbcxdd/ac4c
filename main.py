import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, roc_auc_score

from models import Transformer_GRU
from models import Another_Transformer
from models import TextCNN
from dataloader import test_loader,ac4c_loader
import random



def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    all_logits, all_labels = [], []

    for data in data_iter:
        x = data[0].to(device)
        y = data[1].to(device)

        logits, _ = net(x)
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

device = torch.device("cuda", 0)
criterion_CE = nn.CrossEntropyLoss()
train_iter , val_iter , test_iter , max_len =  test_loader.load_ac4c_data()


if __name__ == '__main__':
    seed = 40

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # net = Another_Transformer.Transformer(max_len).to(device)
    net = Transformer_GRU.model(max_len).to(device)
    # net = TextCNN.CnnModel().to(device)

    lr =0.001
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay= 5e-4)

    best_val_acc = 0
    EPOCH = 100


    for epoch in range(EPOCH):
        loss_list = []
        t0 = time.time()
        net.train()

        for batch in train_iter:
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
            train_acc , _ = evaluate_accuracy(train_iter,net,device=device)
            val_acc , auc = evaluate_accuracy(val_iter,net,device=device)

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_list):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, val_acc: {val_acc}, val_AUC: {auc} time: {time.time() - t0:.2f}'
        print(results)
        # to_log(results)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), '/mnt/sdb/home/lrl/code/ac4c/parameters/Transformer_state_dict.pt')
            torch.save({"best_val_acc": best_val_acc}, '/mnt/sdb/home/lrl/code/ac4c/parameters/Transformer_info.pt')
            print(f"best_val_acc: {best_val_acc}")

    net.eval()
    with torch.no_grad():
        pre_trained = torch.load('/mnt/sdb/home/lrl/code/ac4c/parameters/Transformer_state_dict.pt')
        net.load_state_dict(pre_trained)
        test_acc , auc= evaluate_accuracy(test_iter,net,device=device)

    results = f"final test, loss: {np.mean(loss_list):.5f}\n"
    results += f'\ttest_acc: {test_acc}, test_auc: {auc} time: {time.time() - t0:.2f}'
    print(results)

    print(test_acc)








