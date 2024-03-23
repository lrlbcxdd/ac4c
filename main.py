import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from models import Transformer_model
from models import Another_Transformer
from models import TextCNN
from dataloader import ac4c_loader
import random



def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for data in data_iter:
        x = data[0].to(device)
        y = data[1].to(device)

        logits, _ = net(x)
        # print(logits)
        acc_sum += (logits.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def to_log(log):
    with open("./results/ExamPle_Log.log", "a+") as f:
        f.write(log + '\n')

device = torch.device("cuda", 0)
criterion_CE = nn.CrossEntropyLoss()
train_iter , val_iter , test_iter , max_len=  ac4c_loader.load_ac4c_data()


if __name__ == '__main__':
    seed = 40

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net = Transformer_model.model(max_len).to(device)
    # net = TextCNN.CnnModel().to(device)

    lr =0.001
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay= 5e-4)

    best_val_acc = 0
    EPOCH = 50
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
            train_acc = evaluate_accuracy(train_iter,net)
            val_acc = evaluate_accuracy(val_iter,net)

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_list):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, val_acc: {val_acc}, time: {time.time() - t0:.2f}'
        print(results)
        # to_log(results)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save({"best_val_acc": best_val_acc, "model": net.state_dict()},
            #            '/mnt/sdb/home/lrl/code/contrast_pre_test/parameter/GRU.pl')
            print(f"best_val_acc: {best_val_acc}")

    net.eval()
    with torch.no_grad():
        train_acc = evaluate_accuracy(train_iter,net)
        test_acc = evaluate_accuracy(test_iter,net)

    results = f"final test, loss: {np.mean(loss_list):.5f}\n"
    results += f'\ttest_acc: {test_acc}, time: {time.time() - t0:.2f}'
    print(results)

    print(test_acc)








