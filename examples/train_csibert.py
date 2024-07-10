from model import CSIBERT,Linear
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
import torch.nn as nn
import numpy as np
from transformers import BertConfig
import copy
from dataset import load_data

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument('--prediction_len', type=int, default=1)

    parser.add_argument("--data_path",type=str,default="./data/wiloc_linear.pkl")
    parser.add_argument('--train_prop', type=float, default=0.9)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=30)

    parser.add_argument("--path", type=str, default='./csibert_pretrain.pth')
    parser.add_argument("--no_pretrain", action="store_true",default=False)

    args = parser.parse_args()
    return args


def iteration(data_loader,device,model,cls,optim,train=True,prediction_len=1):
    if train:
        model.train()
        cls.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        cls.eval()
        torch.set_grad_enabled(False)

    loss_func=nn.MSELoss()
    loss_list = []
    last_loss_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for magnitude, _, x, y, timestamp in pbar:
        magnitude = magnitude.float().to(device)
        timestamp = timestamp.float().to(device)
        x = x.float().to(device)
        y = y.float().to(device)
        x = x[:,:-prediction_len]
        y = y[:,:-prediction_len]


        input = copy.deepcopy(magnitude)
        non_pad = (input != pad).float().to(device)
        avg = torch.sum(input * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((input - avg) ** 2) * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        input = (input - avg) / (std + 1e-8)
        non_pad=non_pad.bool()
        batch_size, seq_len, carrier_num = input.shape
        rand_word = torch.randn((batch_size, seq_len, carrier_num)).to(device)
        input[~non_pad]=rand_word[~non_pad]


        output=cls(model(input, timestamp))
        x_hat = output[...,:-prediction_len,0]
        y_hat = output[...,:-prediction_len,1]

        loss = loss_func(x_hat, x) + loss_func(y_hat,y)
        last_loss = loss_func(x_hat[:,-1], x[:,-1]) + loss_func(y_hat[:,-1], y[:,-1])

        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()

        loss_list.append(loss.item())
        last_loss_list.append(last_loss.item())

    return np.mean(loss_list), np.mean(last_loss_list)

if __name__ == '__main__':
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=128,num_hidden_layers=6,num_attention_heads=8,intermediate_size=512)
    model=CSIBERT(bertconfig,args.carrier_dim).to(device)
    if not args.no_pretrain:
        model.load_state_dict(torch.load(args.path))

    cls = Linear(input_dims=128, output_dims=2).to(device)
    train_data, test_data = load_data(data_path=args.data_path, train_prop=args.train_prop, train_num=2000, test_num=200, length=args.max_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    optim = torch.optim.Adam(list(model.parameters()) + list(cls.parameters()), lr=args.lr, weight_decay=0.01)

    best_loss = 1e8
    best_last_loss = 1e8
    loss_epoch = 0
    j = 0

    while True:
        j += 1
        loss, last_loss = iteration(train_loader, device, model, cls, optim, train=True, prediction_len=args.prediction_len)
        log = "Epoch {:}, Train Loss {:06f}, Train Last Loss {:06f}".format(j, loss, last_loss)
        print(log)
        with open("CSIBERT.txt", 'a') as file:
            file.write(log)
        loss, last_loss = iteration(test_loader, device, model, cls, optim, train=False, prediction_len=args.prediction_len)
        log = "Test Loss {:06f}, Test Last Loss {:06f}".format(loss, last_loss)
        print(log)
        with open("CSIBERT.txt", 'a') as file:
            file.write(log + "\n")
        if loss < best_loss:
            best_loss = loss
            loss_epoch = 0
            torch.save(cls.state_dict(), "CSIBERT_cls.pth")
            torch.save(model.state_dict(), "CSIBERT_model.pth")
        else:
            loss_epoch += 1
        if last_loss < best_last_loss:
            best_last_loss = last_loss
        if loss_epoch >= args.epoch:
            break
        print("Best Epoch {:}".format(loss_epoch))
    print("Best Loss {:}".format(best_loss))
    print("Best Last Loss {:}".format(best_last_loss))