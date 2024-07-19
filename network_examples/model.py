import torch.nn as nn
import torchvision.models as models
import torch
from transformers import BertModel,BertConfig
import torch.nn.functional as F

class Resnet(nn.Module):
    def __init__(self, output_dims=64, channel=1, pretrained=True):
        super().__init__()
        self.model=models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dims)
        # self.batch_norm=nn.BatchNorm2d(1)
        self.channel = channel

    def forward(self,x):
        if x.shape[1]!=self.channel:
            x = torch.unsqueeze(x,dim=1)
        # x=self.batch_norm(x)
        return self.model(x)

class CNN(nn.Module):
    def __init__(self,channel=1, class_num=64):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.Linear=nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,class_num)
        )
        self.channel = channel

    def forward(self,x):
        if x.shape[1]!=self.channel:
            x = torch.unsqueeze(x,dim=1)
        x=self.model(x)
        x=x.reshape(x.shape[0],-1)
        x=x[:,:64]
        x=self.Linear(x)
        return x

class CSIBERT(nn.Module):
    def __init__(self,bertconfig, input_dim=52):
        super().__init__()
        self.bertconfig=bertconfig
        self.bert=BertModel(bertconfig)
        self.hidden_dim=bertconfig.hidden_size
        self.input_dim=input_dim
        self.len=bertconfig.max_position_embeddings

        self.csi_emb=nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.time_emb=nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.arl = nn.Sequential(
            nn.Linear(self.len, self.len // 2),
            nn.ReLU(),
            nn.Linear(self.len // 2, self.len // 4),
            nn.ReLU(),
            nn.Linear(self.len // 4, 1)
        )

    def forward(self,x,timestamp=None,attention_mask=None):
        x=x.to(torch.float32)
        x=self.attention(x)
        x=self.csi_emb(x)

        if timestamp is not None:
            x_time=self.time_embedding(timestamp)
            x = x + x_time

        y=self.bert(inputs_embeds=x, attention_mask=attention_mask, output_hidden_states=False)
        y=y.last_hidden_state
        return y

    def time_embedding(self,timestamp,t=1):
        device=timestamp.device
        timestamp = (timestamp - timestamp[:, 0:1]) / (timestamp[:,-1:] - timestamp[:, 0:1]) * self.len

        timestamp**=t
        d_model=self.input_dim
        dim=torch.tensor(list(range(d_model))).to(device)
        batch_size,length=timestamp.shape
        timestamp=timestamp.unsqueeze(2).repeat(1, 1, d_model)
        dim=dim.reshape([1,1,-1]).repeat(batch_size,length,1)
        sin_emb = torch.sin(timestamp/10000**(dim//2*2/d_model))
        cos_emb = torch.cos(timestamp/10000**(dim//2*2/d_model))
        mask=torch.zeros(d_model).to(device)
        mask[::2]=1
        emb=sin_emb*mask+cos_emb*(1-mask)
        emb=self.time_emb(emb)
        return emb

    def attention(self, x):
        y = torch.transpose(x, -1, -2)
        attn = self.arl(y)
        y = y * attn
        y = torch.transpose(y, -1, -2)
        return y

class Token_Classifier(nn.Module):
    def __init__(self,bert,class_num=52):
        super().__init__()
        self.bert=bert
        self.classifier=nn.Sequential(
            nn.Linear(bert.hidden_dim, bert.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(bert.hidden_dim//2, class_num)
        )

    def forward(self,x,timestamp,attention_mask=None):
        x=self.bert(x,timestamp,attention_mask=attention_mask)
        x=self.classifier(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class Sequence_Classifier(nn.Module):
    def __init__(self, csibert, class_num, hs=128, da=128, r=4):
        super().__init__()
        self.bert = csibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs * r, hs * r // 2),
            nn.ReLU(),
            nn.Linear(hs * r // 2, class_num)
        )

    def forward(self, x, timestamp,attention_mask=None):
        x = self.bert(x, timestamp,attention_mask=attention_mask)
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res


class LSTM(nn.Module):
    def __init__(self,output_dim=64,input_dim=52):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.fc(x)
        return y

class RNN(nn.Module):
    def __init__(self, output_dims=64,input_dim=52):
        super().__init__()
        self.rnn = nn.RNN(input_dim, 64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, output_dims)

    def forward(self, x):
        x, _ = self.rnn(x)
        y = self.fc(x)
        return y

class GRU(nn.Module):
    def __init__(self, output_dims=64,input_dim=52):
        super(GRU, self).__init__()
        self.input_size = input_dim
        self.hidden_size = 64
        self.output_size = output_dims

        self.gru = nn.GRU(self.input_size, 64, 4, batch_first=True)
        self.fc = nn.Linear(64, self.output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

class Linear(nn.Module):
    def __init__(self, input_dims=64, output_dims=2):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dims,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, output_dims),
            nn.ReLU()
        )

    def forward(self,x):
        return self.model(x)

class Linear_attention(nn.Module):
    def __init__(self, input_dims=64, output_dims=2, da=128, r=4):
        super().__init__()
        self.attention = SelfAttention(input_dims, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(input_dims * r, input_dims * r // 2),
            nn.ReLU(),
            nn.Linear(input_dims * r // 2, output_dims),
            nn.ReLU()
        )


    def forward(self,x):
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res



#test
if __name__ == '__main__':
    x = torch.randn([2,100,52])

    resnet = Resnet()
    cnn = CNN()
    lstm = LSTM()
    rnn = RNN()
    gru = GRU()

    configuration = BertConfig(max_position_embeddings=100, hidden_size=64, num_hidden_layers=6,num_attention_heads=8)
    csibert=CSIBERT(configuration)

    linear = Linear()

    y = linear(cnn(x))
    print(y.shape)
    y = linear(resnet(x))
    print(y.shape)
    y = linear(lstm(x))
    print(y.shape)
    y = linear(rnn(x))
    print(y.shape)
    y = linear(gru(x))
    print(y.shape)
    y = linear(csibert(x))
    print(y.shape)