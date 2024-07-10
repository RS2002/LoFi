from model import CSIBERT,Token_Classifier,Sequence_Classifier
from transformers import BertConfig,AdamW
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
from dataset import load_data

pad=-1000

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0], help="CUDA device ids")
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--max_len", type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--data_path', type=str, default="./data/wiloc.pkl")
    parser.add_argument('--train_prop', type=float, default=0.9)
    parser.add_argument('--GAN', action="store_true", default=False)

    args = parser.parse_args()
    return args



def iteration(data_loader,device,model,discriminator,optim,optim_dis,train=True,gan=False):
    if train:
        model.train()
        discriminator.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        discriminator.eval()
        torch.set_grad_enabled(False)

    loss_list = []
    mse_list = []
    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, _, _, _, timestamp in pbar:
        x = x.float().to(device)
        timestamp = timestamp.float().to(device)
        input = copy.deepcopy(x)

        non_pad = (input != pad).to(device)
        batch_size, seq_len, carrier_num = input.shape
        rand_word = torch.randn((batch_size, seq_len, carrier_num)).to(device)
        loss_mask = torch.zeros([batch_size, seq_len]).to(device)
        chosen_num_min = int(seq_len * 0.1)
        chosen_num_max = int(seq_len * 0.7)
        num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
        row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
        col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
        loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
        loss_mask[~non_pad[..., 0]] = 0
        loss_mask1 = loss_mask.unsqueeze(2).repeat(1, 1, carrier_num)
        loss_mask1 = 1 - loss_mask1
        loss_mask1[~non_pad]=0
        avg = torch.sum(input * loss_mask1, dim=1, keepdim=True) / (torch.sum(loss_mask1, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((input - avg) ** 2) * loss_mask1, dim=1, keepdim=True) / (
                    torch.sum(loss_mask1, dim=1, keepdim=True) + 1e-8))
        input = (input - avg) / (std + 1e-8)
        input[~non_pad.bool()] = rand_word[~non_pad.bool()]
        input_copy = copy.deepcopy(input)
        input[loss_mask.bool()] = rand_word[loss_mask.bool()]


        y = model(input, timestamp)
        y_copy = y.clone()
        y = y * std + avg

        non_pad = non_pad.float()

        avg_hat = torch.mean(y, dim=1, keepdim=True)
        std_hat = torch.sqrt(torch.mean((y - avg_hat) ** 2, dim=1, keepdim=True))

        loss_mse = nn.MSELoss(reduction="none")
        # loss_mse = nn.SmoothL1Loss(reduction="none")
        weight=[3.0,2.0,1.0,1.0,1.0,1.0,0.5,0.5]
        # weight=[1.0]*8
        # loss1: MASK MSE
        loss_mask = loss_mask.unsqueeze(2).repeat(1, 1, carrier_num)
        loss1 = torch.sum(loss_mse(y, x) * loss_mask) / torch.sum(loss_mask)
        mse = (torch.sum(loss_mse(y, x) * loss_mask) / torch.sum(loss_mask)).item()
        # loss2: Total MSE
        loss2 = torch.sum(loss_mse(y, x) * non_pad) / torch.sum(non_pad)
        loss = loss1*weight[0] + loss2*weight[1]
        # loss3,4: Total Avg & Std loss
        loss3 = torch.mean(loss_mse(avg_hat, avg))
        loss4 = torch.mean(loss_mse(std_hat, std))
        loss += loss3*weight[2] + loss4*weight[3]
        # loss5,6: Mask Avg & Std loss
        x_mask = x * loss_mask
        y_mask = y * loss_mask
        x_mask_mean = torch.sum(x_mask, dim=1, keepdim=True) / (torch.sum(loss_mask, dim=1, keepdim=True) + 1e-8)
        y_mask_mean = torch.sum(y_mask, dim=1, keepdim=True) / (torch.sum(loss_mask, dim=1, keepdim=True) + 1e-8)
        x_mask_std = torch.sqrt(
            torch.sum(((x_mask_mean - x_mask) * loss_mask) ** 2, dim=1) / (torch.sum(loss_mask, dim=1) + 1e-8))
        y_mask_std = torch.sqrt(
            torch.sum(((y_mask_mean - y_mask) * loss_mask) ** 2, dim=1) / (torch.sum(loss_mask, dim=1) + 1e-8))
        loss5 = torch.mean(loss_mse(x_mask_mean, y_mask_mean))
        loss6 = torch.mean(loss_mse(x_mask_std, y_mask_std))
        loss += loss5*weight[4] + loss6*weight[5]

        if train:
            input_copy = input_copy * std + avg
            y_copy[~non_pad.bool()] = rand_word[~non_pad.bool()]
            y_copy = y_copy * std + avg

            if gan:
                attn_mask=non_pad[...,0].bool()
                loss_cls = nn.CrossEntropyLoss()
                false = torch.zeros(batch_size, dtype=torch.long).to(device)
                truth = torch.ones(batch_size, dtype=torch.long).to(device)

                truth_hat = discriminator(input_copy,timestamp,attention_mask=attn_mask)
                false_hat = discriminator(y_copy.detach(),timestamp,attention_mask=attn_mask)

                loss_truth = loss_cls(truth_hat, truth)
                loss_false = loss_cls(false_hat, false)
                dis_loss = loss_truth + loss_false

                discriminator.zero_grad()
                dis_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 3.0)
                optim_dis.step()

                gen_loss = loss_cls(discriminator(y_copy, timestamp,attention_mask=attn_mask), truth)
                loss += gen_loss*weight[6]

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        break
            if has_nan:
                print("NAN Gradient->Skip")
                continue
            optim.step()

        loss_list.append(loss.item())
        mse_list.append(mse)

    return np.mean(loss_list), np.mean(mse_list)


def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=128,num_hidden_layers=6,num_attention_heads=8,intermediate_size=512)
    csibert=CSIBERT(bertconfig,args.carrier_dim).to(device)
    csibert_dis=CSIBERT(bertconfig,args.carrier_dim).to(device)
    if len(cuda_devices) > 1 and not args.cpu:
        csibert = nn.DataParallel(csibert, device_ids=cuda_devices)
        csibert_dis = nn.DataParallel(csibert_dis, device_ids=cuda_devices)

    model = Token_Classifier(csibert, args.carrier_dim).to(device)
    discriminator = Sequence_Classifier(csibert_dis, class_num=2).to(device)
    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)
        discriminator = nn.DataParallel(discriminator, device_ids=cuda_devices)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optim_dis = AdamW(discriminator.parameters(), lr=args.lr, weight_decay=0.01)


    train_data, test_data = load_data(data_path=args.data_path, train_prop=args.train_prop, train_num=2000, test_num=200, length=args.max_len)
    train_lodaer = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    best_loss = 1e8
    best_mse = 1e8
    loss_epoch = 0
    mse_epoch = 0
    j = 0

    while True:
        j+=1
        loss,mse=iteration(train_lodaer,device,model,discriminator,optim,optim_dis,train=True,gan=args.GAN)
        log = "Epoch {} | Train Loss {:06f} ,  Train MSE {:06f} | ".format(j, loss, mse)
        print(log)
        with open("Pretrain.txt", 'a') as file:
            file.write(log)
        loss,mse=iteration(test_loader,device,model,discriminator,optim,optim_dis,train=False,gan=args.GAN)
        log = "Test Loss {:06f} , Test MSE {:06f}".format(loss,mse)
        print(log)
        with open("Pretrain.txt", 'a') as file:
            file.write(log + "\n")
        if mse<=best_mse or loss<=best_loss:
            torch.save(csibert.state_dict(), "csibert_pretrain.pth")

        if mse<=best_mse:
            best_mse=mse
            mse_epoch=0
        else:
            mse_epoch+=1
        if loss<=best_loss:
            best_loss=loss
            loss_epoch=0
        else:
            loss_epoch+=1
        if mse_epoch>=args.epoch and loss_epoch>=args.epoch:
            break
        print("MSE Epoch {:}, Loss Epcoh {:}".format(mse_epoch,loss_epoch))

if __name__ == '__main__':
    main()