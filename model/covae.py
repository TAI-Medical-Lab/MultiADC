import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2,kernel_size=k_size, stride=1, padding=k_size//2),
            
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2),
            
        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2


class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, k_size):
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.layer2(x)
        return x


class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 15+1, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),#64*3
            nn.ReLU()
        )

        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.reg3 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.reg4 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.reg5 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )


    def forward(self, A, B, C, D, E, F):
        A = self.reg1(A)
        B = self.reg2(B)
        C = self.reg2(C)
        D = self.reg2(D)
        E = self.reg2(E)
        # print(A.shape,B.shape,C.shape,D.shape,E.shape,F.shape
        # )
        x = torch.cat((A, B,C,D,E,F), 1)
        x = self.reg(x)
        return x


class net(nn.Module):
    def __init__(self, max_smi_len,max_seq_len, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(64, 128)
        self.embedding2 = nn.Embedding(25, 128)
        self.cnn_playload = CNN(NUM_FILTERS, FILTER_LENGTH1)
        self.cnn_linker = CNN(NUM_FILTERS, FILTER_LENGTH1)
        self.cnn_heavy = CNN(NUM_FILTERS, FILTER_LENGTH2)
        self.cnn_light = CNN(NUM_FILTERS, FILTER_LENGTH2)
        self.cnn_antigen = CNN(NUM_FILTERS, FILTER_LENGTH2)
        self.reg = net_reg(NUM_FILTERS)
        self.decoder1 = decoder(max_smi_len, NUM_FILTERS, FILTER_LENGTH1,64)
        self.decoder2 = decoder(max_smi_len, NUM_FILTERS, FILTER_LENGTH1,64)
        self.decoder3 = decoder(max_seq_len, NUM_FILTERS, FILTER_LENGTH2,25)
        self.decoder4 = decoder(max_seq_len, NUM_FILTERS, FILTER_LENGTH2,25)
        self.decoder5 = decoder(max_seq_len, NUM_FILTERS, FILTER_LENGTH2,25)

    def forward(self, heavy,light,antigen,playload, linker, dar, max_smi_len,max_seq_len, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        heavy_init = Variable(heavy.long()).cuda()
        light_init =Variable(light.long()).cuda()
        antigen_init =Variable(antigen.long()).cuda()
        playload_init =Variable(playload.long()).cuda()
        linker_init =Variable(linker.long()).cuda()

        heavy_embedding = self.embedding2(heavy_init).permute(0, 2, 1)
        light_embedding = self.embedding2(light_init).permute(0, 2, 1)
        antigen_embedding = self.embedding2(antigen_init).permute(0, 2, 1)
        playload_embedding = self.embedding1(playload_init).permute(0, 2, 1)
        linker_embedding = self.embedding1(linker_init).permute(0, 2, 1)

        p, mu_p, logvar_p = self.cnn_playload(playload_embedding)
        l, mu_l, logvar_l = self.cnn_linker(linker_embedding)
        h, mu_h, logvar_h = self.cnn_heavy(heavy_embedding)
        lt, mu_lt, logvar_lt = self.cnn_light(light_embedding)
        a, mu_a, logvar_a = self.cnn_antigen(antigen_embedding)

        dar=dar.view(-1,1)
        out = self.reg(p,l,h,lt,a,dar).squeeze()
        p = self.decoder1(p, max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        l = self.decoder2(l, max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        h = self.decoder3(h, max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        lt = self.decoder4(lt, max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        a = self.decoder5(a, max_seq_len, NUM_FILTERS, FILTER_LENGTH2)

        return out, p,l,h,lt,a,heavy_init,light_init,antigen_init,playload_init,linker_init,mu_p, logvar_p, mu_l, logvar_l, mu_h, logvar_h, mu_lt, logvar_lt, mu_a, logvar_a

