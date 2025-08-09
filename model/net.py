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
    def __init__(self, device='cpu', compound_dim=128, protein_dim=128, 
                 gt_layers=3, gt_heads=4, out_dim=1, dropout_rate=0.2):
        super(MultiADC, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads
        
        # Compound encoders
        self.Compound_encoder = gt_net_compound.GraphTransformer(
            device, n_layers=gt_layers, node_dim=44, edge_dim=10, 
            hidden_dim=compound_dim, out_dim=compound_dim, 
            n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, 
            pos_enc_dim=8
        )
        
        self.Adc_encoder = gt_net_compound.GraphTransformer(
            device, n_layers=gt_layers, node_dim=128, edge_dim=5, 
            hidden_dim=compound_dim, out_dim=compound_dim, 
            n_heads=gt_heads, in_feat_dropout=0.1, dropout=0.1, 
            pos_enc_dim=8
        )

        # Protein feature extractors
        self.Protein_linear = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, protein_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Projection layers for ADC graph nodes
        self.proj_payload = nn.Linear(compound_dim, 128)
        self.proj_linker = nn.Linear(compound_dim, 128)
        self.proj_light = nn.Linear(protein_dim, 128)
        self.proj_heavy = nn.Linear(protein_dim, 128)
        self.proj_antigen = nn.Linear(protein_dim, 128)
        
        # Component fusion
        self.downdim = nn.Linear(5889, 256)
        self.Fuse_linear = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128)
        )   
        # Enhanced classifier
        self.Classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(256, out_dim)
        )
        # Auxiliary output for DAR prediction
        self.aux_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def get_graph_feature_mean_max(self, bg, feats, method):

        num_nodes = bg.batch_num_nodes()
        out = []
        start_idx = 0
        
        for n_nodes in num_nodes:

            subgraph_feats = feats[start_idx:start_idx + n_nodes]
            
            if method == 'mean':
                subgraph_summary = subgraph_feats.mean(dim=0)
            elif method == 'max':
                subgraph_summary = subgraph_feats.max(dim=0)[0]  
            elif method == 'sum':
                subgraph_summary = subgraph_feats.sum(dim=0)  
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            out.append(subgraph_summary)
            start_idx += n_nodes
        
        out = torch.stack(out)  # shape: (batch_size, feat_dim)
        return out
    def get_graph_feature(self, bg, feats, method='mean'):
        """Flexible graph feature pooling"""
        if method == 'mean':
            return self.get_graph_feature_mean_max(bg, feats,'mean')
        elif method == 'max':
            return self.get_graph_feature_mean_max(bg, feats,'max')
        elif method == 'sum':
            return self.get_graph_feature_mean_max(bg, feats,'sum')
        elif method == 'mean_max':
            mean_pool = self.get_graph_feature_mean_max(bg, feats,'mean')
            max_pool = self.get_graph_feature_mean_max(bg, feats,'max')
            return torch.cat([mean_pool, max_pool], dim=1)
        else:  # Virtual node
            return self.get_vn_feature(bg, feats)

    def get_vn_feature(self, bg, feats):
        """Virtual node feature extraction"""
        num_nodes = bg.batch_num_nodes()
        out = []
        start_idx = 0
        for n_nodes in num_nodes:
            vn_feature = feats[start_idx + n_nodes - 1]  # Last node is virtual
            out.append(vn_feature)
            start_idx += n_nodes
        return torch.stack(out)

    def allocate_node_feats(self, batch_adc_graph, batch_payload, batch_linker, 
                           batch_light, batch_heavy, batch_antigen):
        """Assign features to ADC graph nodes"""
        nodenum = 8
        for i in range(batch_payload.size(0)):
            batch_adc_graph.nodes[i * nodenum].data['atom'] = self.proj_payload(batch_payload[i].unsqueeze(0))
            batch_adc_graph.nodes[i * nodenum + 1].data['atom'] = self.proj_linker(batch_linker[i].unsqueeze(0))
            batch_adc_graph.nodes[i * nodenum + 2].data['atom'] = self.proj_light(batch_light[i].unsqueeze(0))
            batch_adc_graph.nodes[i * nodenum + 3].data['atom'] = self.proj_light(batch_light[i].unsqueeze(0))
            batch_adc_graph.nodes[i * nodenum + 4].data['atom'] = self.proj_heavy(batch_heavy[i].unsqueeze(0))
            batch_adc_graph.nodes[i * nodenum + 5].data['atom'] = self.proj_heavy(batch_heavy[i].unsqueeze(0))
            batch_adc_graph.nodes[i * nodenum + 6].data['atom'] = self.proj_antigen(batch_antigen[i].unsqueeze(0))
        return batch_adc_graph

    def forward(self, heavy, light, antigen, payload_graph, linker_graph, dar, adc_graph, components):
        # Process proteins
        # print(heavy.shape)
        heavy = self.Protein_linear(heavy)
        light = self.Protein_linear(light)
        antigen = self.Protein_linear(antigen)
        
        # Process compounds
        payload_feat = self.Compound_encoder(payload_graph)
        linker_feat = self.Compound_encoder(linker_graph)
        payload = self.get_graph_feature(payload_graph, payload_feat, 'vn')
        linker = self.get_graph_feature(linker_graph, linker_feat, 'vn')
        
        # Process components
        components = self.downdim(components)
        components_feat = self.Fuse_linear(components)

        # Build ADC graph
        adc_graph = self.allocate_node_feats(
            adc_graph, payload, linker, light, heavy, antigen
        )
        # Process ADC graph
        adc_feat = self.Adc_encoder(adc_graph)
        adc_feat = self.get_graph_feature(adc_graph, adc_feat, 'vn')
        
        # Concatenate features
        fused_features = torch.cat([components_feat, adc_feat], dim=1)
        # fused_features = components_feat
        
        # Predictions
        main_output = self.Classifier(fused_features)
        # aux_output = self.aux_classifier(fused_features)
        
        return main_output
