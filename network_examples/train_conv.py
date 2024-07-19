from model import Resnet,CNN,Linear
import torch
from torch.utils.data import DataLoader
import tqdm
import argparse
import torch.nn as nn
import numpy as np
from dataset import load_data

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--correlation", action="store_true",default=False)
    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--ratio", action="store_true",default=False)

    parser.add_argument("--data_path",type=str,default="./data/wiloc_linear.pkl")
    parser.add_argument('--train_prop', type=float, default=0.9)
    parser.add_argument("--model", type=str, default='resnet')

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=30)
    args = parser.parse_args()
    return args


def iteration(data_loader,device,model,cls,optim,train=True,norm=False,correlation=False,ratio=False):
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
    std_list = []
    mean_list = []
    pbar = tqdm.tqdm(data_loader, disable=False)
    for magnitude, _, x, y, _ in pbar:
        magnitude = magnitude.float().to(device)
        x = x.float().to(device)
        y = y.float().to(device)

        # x = x[:,-1]
        # y = y[:,-1]

        # l = x.shape[-1]
        # x = x[:,l//2]
        # y = y[:,l//2]

        x = torch.mean(x,dim=-1)
        y = torch.mean(y,dim=-1)

        if norm:
            mean = torch.mean(magnitude, dim=-2, keepdim=True)
            std = torch.std(magnitude, dim=-2, keepdim=True)
            magnitude = (magnitude - mean) / (std + 1e-8)
        if correlation:
            magnitude = torch.matmul(magnitude.transpose(-1, -2), magnitude)
        if ratio:
            magnitude1 = magnitude[:,:-1,:]
            magnitude2 = magnitude[:,1:,:]
            magnitude = magnitude2/(magnitude1+1e-8)

        output=cls(model(magnitude))
        x_hat = output[...,0]
        y_hat = output[...,1]

        loss = loss_func(x_hat, x) + loss_func(y_hat,y)


        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()

        loss_list.append(loss.item())
        mean_list.append(torch.mean(torch.sqrt((x_hat-x)**2+(y_hat-y)**2)).item())
        std_list.append(torch.std(torch.sqrt((x_hat-x)**2+(y_hat-y)**2)).item())
    return np.mean(loss_list), np.mean(mean_list), np.mean(std_list)

if __name__ == '__main__':
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    if args.model == "resnet":
        model = Resnet(channel=1).to(device)
    elif args.model == "cnn":
        model = CNN(channel=1).to(device)
    else:
        print("No such model!")
        exit(-1)

    cls = Linear(output_dims=2).to(device)
    train_data, test_data = load_data(data_path=args.data_path, train_prop=args.train_prop, train_num=2000, test_num=200, length=args.max_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    optim = torch.optim.Adam(list(model.parameters()) + list(cls.parameters()), lr=args.lr, weight_decay=0.01)

    best_loss = 1e8
    loss_epoch = 0
    j = 0
    best_mean = 1e8
    best_std = 1e8
    while True:
        j += 1
        loss, mean, std = iteration(train_loader, device, model, cls, optim, train=True, norm=args.norm, correlation=args.correlation, ratio=args.ratio)
        log = "Epoch {:} | Train Loss {:06f}, Train Mean {:06f}, Train Std {:06f} | ".format(j, loss, mean, std)
        print(log)
        with open("log.txt", 'a') as file:
            file.write(log)
        loss, mean, std = iteration(test_loader, device, model, cls, optim, train=False, norm=args.norm, correlation=args.correlation, ratio=args.ratio)
        log = "Test Loss {:06f}, Test Mean {:06f}, Test Std {:06f}".format(loss, mean, std)
        print(log)
        with open("log.txt", 'a') as file:
            file.write(log + "\n")
        if loss < best_loss:
            best_loss = loss
            loss_epoch = 0
            torch.save(cls.state_dict(), "cls.pth")
            torch.save(model.state_dict(), "model.pth")
        else:
            loss_epoch += 1
        if mean < best_mean:
            best_mean = mean
        if std < best_std:
            best_std = std
        if loss_epoch >= args.epoch:
            break
        print("Best Epoch {:}".format(loss_epoch))
    print("Best Loss {:}".format(best_loss))
    print("Best Mean {:}".format(best_mean))
    print("Best Std {:}".format(best_std))

