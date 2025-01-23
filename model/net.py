import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from model import gt_net_compound
from model import gin
from model import covae

from torch_geometric.nn.conv import GATConv,GATv2Conv,TransformerConv


if torch.cuda.is_available():
    device = torch.device('cuda')

class VAE(nn.Module):

    # def __init__(self, input_dim=641, h_dim=256, z_dim=128):
    def __init__(self, input_dim=5889, h_dim=1024, z_dim=128):
        
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        #  [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        """
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        x = x.view(batch_size, self.input_dim)  

        # encoder
        mu, log_var = self.encode(x)
        return mu, log_var
        # # reparameterization trick
        # sampled_z = self.reparameterization(mu, log_var)
        # # decoder
        # x_hat = self.decode(sampled_z)
        # # reshape
        # x_hat = x_hat.view(batch_size, 1, 28, 28)
        # return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  
        return x_hat


class GateLinearUnit(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.Tanh()):
        super(GateLinearUnit, self).__init__()
        # self.batch_norm = batch_norm
        self.activation = activation
        # self.conv_layer1 = nn.Conv2d(1, num_filers, (kernel_size, input_size), bias=bias)
        # self.conv_layer2 = nn.Conv2d(1, num_filers, (kernel_size, input_size), bias=bias)
        self.layer1=nn.Linear(in_features=input_size,out_features=output_size,bias=False)
        # self.batch_norm = nn.BatchNorm2d(num_filers)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.layer1.weight)
        # nn.init.kaiming_uniform_(self.conv_layer2.weight)

    def gate(self, inputs):
        return self.sigmoid(inputs)

    def forward(self, inputs):
        # inputs = inputs
        output = self.layer1(inputs)
        # gate_output = self.conv_layer2(inputs)
        # Gate Operation

        output = inputs * self.gate(output)

        return output


class DTF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(DTF, self).__init__()
        inter_channels = int(channels // r)

        self.att1 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.att2 = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        w1 = self.sigmoid(self.att1(fd + fp))
        # print('w1:', w1.shape)
        fout1 = fd * w1 + fp * (1 - w1)

        w2 = self.sigmoid(self.att2(fout1))
        # print('w2', w2.shape)
        # fd = fd * w2
        # fp = fp * (1 - w2)
        fout2 = fd * w2 + fp * (1 - w2)
        
        w3 = self.sigmoid(fout2)
        fout = w3 * fout2 + (1 - w3) * (fd + fp)

        # fout = torch.cat([fout1, fout2], dim=1)
        return fout
      

class MultiADC(nn.Module):
    def __init__(self, device='cpu',compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=2):
        super(MultiADC, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads
        
        self.Compound_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)
  
        # self.adc_encoder = GATv2Conv(in_channels=128, out_channels=16, heads=8, concat=True, negative_slope=0.2, dropout=0.1,edge_dim = 5)
        self.Adc_encoder = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=128, edge_dim=5, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=8)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.protein_dim, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.protein_dim, out_channels=self.protein_dim * 2, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.protein_dim * 2, out_channels=self.protein_dim, kernel_size=12),
            nn.ReLU(),
        )


        self.Fuse_linear = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(128, 128)
        )
        # self.glu = GateLinearUnit(input_size=256, output_size=256)

        self.Classifier = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.Linear(128, 1024),#ablation 1/2
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )
        # self.fuse_encoder = VAE()

        self.Protein_linear = nn.Linear(1280, 128)
        self.dtf = DTF()


    def get_vn_feature(self, bg, feats):
        num_nodes = bg.batch_num_nodes()
        
        # num_nodes_list = num_nodes.numpy().tolist()
        # out=torch.tensor([])
        out=[]
        s=num_nodes[0]
        # print(num_nodes.sum())
        # out=feats[s-1]
        s=0
        for i in range(len(num_nodes)):
            s=s+num_nodes[i].item()
            out.append(feats[s-1])
            
        out = torch.stack(out)        
        return out

    def get_graph_feature(self, bg, feats):
        num_nodes = bg.batch_num_nodes()
    
        out = []
        start_idx = 0
        for n_nodes in num_nodes:
            subgraph_feats = feats[start_idx:start_idx + n_nodes]
            # subgraph = subgraph_feats.sum(dim=0)
            subgraph, _ = torch.max(subgraph_feats, dim=0)
            # subgraph= torch.mean(subgraph_feats,dim=0)
            out.append(subgraph)
            start_idx += n_nodes
        
        # 将所有子图的表示堆叠成一个张量
        out = torch.stack(out)
        return out

    def allocate_node_feats(self,batch_adc_graph,batch_playload,batch_linker,batch_light,batch_heavy,batch_antigen):
        #one batch smaples
        nodenum=8
        for i in range(batch_playload.size(0)):
            
            batch_adc_graph.nodes[i * nodenum].data['atom'] = batch_playload[i].unsqueeze(0)
            batch_adc_graph.nodes[i * nodenum + 1].data['atom'] = batch_linker[i].unsqueeze(0)
            batch_adc_graph.nodes[i * nodenum + 2].data['atom'] = batch_light[i].unsqueeze(0)
            batch_adc_graph.nodes[i * nodenum + 3].data['atom'] = batch_light[i].unsqueeze(0)
            batch_adc_graph.nodes[i * nodenum + 4].data['atom'] = batch_heavy[i].unsqueeze(0)
            batch_adc_graph.nodes[i * nodenum + 5].data['atom'] = batch_heavy[i].unsqueeze(0)
            batch_adc_graph.nodes[i * nodenum + 6].data['atom'] = batch_antigen[i].unsqueeze(0)
        return batch_adc_graph

    def forward(self, heavy, light, antigen, playload_graph, linker_graph, dar,adc_graph,components):

        playload_node_feat = self.Compound_encoder(playload_graph)  
        linker_node_feat = self.Compound_encoder(linker_graph)
        playload = self.get_vn_feature(playload_graph, playload_node_feat)# Virtual node feature
        linker = self.get_vn_feature(linker_graph, linker_node_feat)
        # playload = self.get_graph_feature(playload_graph,playload_feat)#Use other READOUT methods
        # linker = self.get_graph_feature(linker_graph,linker_feat)
        
        heavy=self.Protein_linear(heavy)
        light=self.Protein_linear(light)
        antigen=self.Protein_linear(antigen)
        # dar = dar.reshape(-1, 1)
        adc_graph=self.allocate_node_feats(adc_graph,playload,linker,light,heavy,antigen)

        components_feat= components
        # print(fused_vector.shape)
        components_feat= self.Fuse_linear(components_feat)
        components_feat+=components

        # h = adc.ndata['atom'].float()
        # adc_feat=self.adc_encoder(adc,h)#GIN
        # src, dst = adc.edges()
        # edge_index = torch.stack([src, dst], dim=0)
        # e=adc.edata['bond']
        # adc_feat=self.adc_encoder(h,edge_index,e)#GAT/TransformerCon
        
        adc_feat=self.Adc_encoder(adc_graph)
        # adc_feat = self.get_graph_feature(adc,adc_feat)#sum/mean
        adc_feat = self.get_vn_feature(adc_graph,adc_feat)#vn

        all=torch.cat([components_feat,adc_feat], dim=1)
        x = self.Classifier(all)
        return x
    