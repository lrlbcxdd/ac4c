import numpy as np
import torch
import time
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from models import Mamba
from dataloader import ac4c_loader,test_loader
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
train_iter , val_iter , test_iter , max_len=  test_loader.load_ac4c_data(1)


if __name__ == '__main__':
    seed = 40

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net = Mamba.Mamba(seq_len=max_len,d_model=64,state_size=128).to(device)

    optimizer = torch.optim.Adam(net.parameters(),lr=0.001,weight_decay= 5e-6)

    best_val_acc = 0
    EPOCH = 60
    for epoch in range(EPOCH):
        loss_list = []
        t0 = time.time()
        net.train()

        for batch in train_iter:
            optimizer.zero_grad()

            data = batch[0].to(device)
            label = batch[1].to(device)

            logits , _ = net(data)  # torch.Size([b, l])

            ce_loss = criterion_CE(logits.view(-1,2),label).mean()

            train_loss = ce_loss

            train_loss.backward(retain_graph=True)

            for name, param in net.named_parameters():
                if 'out_proj.bias' not in name:
                    # clip weights but not bias for out_proj
                    torch.nn.utils.clip_grad_norm_(param, max_norm=10.0)

            optimizer.step()

            loss_list.append(train_loss.item())

        net.eval()
        total_loss = 0
        with torch.no_grad():
            train_acc, _ = evaluate_accuracy(train_iter, net, device=device)
            val_acc, auc = evaluate_accuracy(val_iter, net, device=device)

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_list):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, val_acc: {val_acc}, val_AUC: {auc} time: {time.time() - t0:.2f}'
        print(results)
        # to_log(results)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), '/mnt/sdb/home/lrl/code/ac4c/parameters/Mamba_state_dict.pt')
            torch.save({"best_val_acc": best_val_acc}, '/mnt/sdb/home/lrl/code/ac4c/parameters/Mamba_info.pt')
            print(f"best_val_acc: {best_val_acc}")

    net.eval()
    with torch.no_grad():
        pre_trained = torch.load( '/mnt/sdb/home/lrl/code/ac4c/parameters/Mamba_state_dict.pt')
        net.load_state_dict(pre_trained)
        test_acc, auc = evaluate_accuracy(test_iter, net, device=device)

    results = f"final test, loss: {np.mean(loss_list):.5f}\n"
    results += f'\ttest_acc: {test_acc}, test_auc: {auc} time: {time.time() - t0:.2f}'
    print(results)

    print(test_acc)








