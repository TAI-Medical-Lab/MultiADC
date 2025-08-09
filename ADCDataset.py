import dgl
import torch
import numpy as np
import pickle
import h5py
# from adcutils import smiles2adjoin, molecular_fg
from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from smile_process import laplacian_positional_encoding


def compound_fingerprint_get( smiles_list):
    morgan_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES string")
        else:
            morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        morgan_list.append(morgan)
    return np.array(morgan_list)

def antigen_get( adcid):
    antigen_list = []
    #ESM-2 Output
    with open('dataset/protein_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    for id in adcid:
        proteinemb = proteinembs.get(id.upper(), None)
        # proteinemb = proteinembs[id]
        if proteinemb is None:
            proteinemb=torch.tensor([0.0]*1280)
        proteinemb=proteinemb.cpu().numpy()
        antigen_list.append(proteinemb)
    return  np.array(antigen_list)

def heavy_get( adcid):
    heavy_list = []
    with open('dataset/protein_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    for id in adcid:
        proteinemb = proteinembs.get(id.upper(), None)
        if proteinemb is None:
            proteinemb=torch.tensor([0.0]*1280)
        proteinemb=  proteinemb.cpu().numpy()

        # proteinemb = proteinembs[id]
        heavy_list.append(proteinemb)
    return np.array(heavy_list)

def light_get( adcid):
    light_list = []
    with open('dataset/protein_1280.pkl', 'rb') as f:
        proteinembs = pickle.load(f)
    for id in adcid:
        proteinemb = proteinembs.get(id.upper(), None)
        # proteinemb = proteinembs[id]
        if proteinemb is None:
            proteinemb=torch.tensor([0.0]*1280)
        proteinemb=proteinemb.cpu().numpy()
        light_list.append(proteinemb)
    return  np.array(light_list)

def compound_graph_get( smiles):
    # smiles_TVdataset = self.data[:, 0]
    compounds_graph = []
    # N = len(id_TVdataset)
    with open('dataset/processed_aug/compound_graphs_vn.pkl', 'rb') as f:  
        smiles2graph = pickle.load(f)
    for no, smile in enumerate(smiles):
        # print('/'.join(map(str, [no + 1, N])))
        # compound_graph_TVdataset, _ = load_graphs('dataset/' + self.dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        compound_graph = smiles2graph[smile]
        compounds_graph.append(compound_graph[0])
    return compounds_graph

class Dim_Reduct_Data:
    """
    Perform dimensionality reduction on the component features, and append the reduced-dimensional features to the end of the original dataset
    """
    
    def __init__(self, n_components=128):
        self.n_components = n_components
        self.scaler = None
        self.pca = None

    def fit_transform(self, train_data):
        """
        Perform dimensionality reduction on the training set.
        """
        heavy = heavy_get(train_data['heavy'])
        light = light_get(train_data['light'])
        antigen = antigen_get(train_data['antigen'])
        payload = compound_fingerprint_get(train_data['payload'])
        linker = compound_fingerprint_get(train_data['linker'])
        # print(type(heavy), heavy.shape if hasattr(heavy, 'shape') else len(heavy))
        # print(type(light), light.shape if hasattr(light, 'shape') else len(light))
        # print(type(antigen), antigen.shape if hasattr(antigen, 'shape') else len(antigen))
        # print(type(payload), payload.shape if hasattr(payload, 'shape') else len(payload))
        # print(type(linker), linker.shape if hasattr(linker, 'shape') else len(linker))

        
        # print(type(dar), dar.shape if hasattr(dar, 'shape') else len(dar))
        dar = np.expand_dims(train_data['dar'], axis=1)
        fused_vector = np.concatenate([heavy, light, antigen, payload, linker, dar], axis=1)

        # self.scaler = StandardScaler()
        # fused_vector_scaled = self.scaler.fit_transform(fused_vector)

        # self.pca = PCA(n_components=128)
        # fused_vector_reduced = self.pca.fit_transform(fused_vector_scaled)

        # return np.hstack((train_data, fused_vector_reduced))
        return np.hstack((train_data, fused_vector))

    def transform(self, test_data):
        """
        Apply the dimensionality reduction weights from the training set to the test set.
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA model and scaler must be trained on the training data first!")

        heavy = heavy_get(test_data['heavy'])
        light = light_get(test_data['light'])
        antigen = antigen_get(test_data['antigen'])
        payload = compound_fingerprint_get(test_data['payload'])
        linker = compound_fingerprint_get(test_data['linker'])
        dar = np.expand_dims(test_data['dar'], axis=1)


        fused_vector = np.concatenate([heavy, light, antigen, payload, linker, dar], axis=1)

        # fused_vector_scaled = self.scaler.transform(fused_vector)
        # fused_vector_reduced = self.pca.transform(fused_vector_scaled)

        # return np.hstack((test_data, fused_vector_reduced))
        return np.hstack((test_data, fused_vector))



class MultiADC_Dataset(Dataset):

    def __init__(self, dataset_fold=None):
        self.data = dataset_fold
        # print(np.array(self.data[0]).shape)
        self.adcid = self.data[:, 0]
        self.payload = compound_graph_get(self.data[:, 4])
        self.linker = compound_graph_get(self.data[:, 5])
        self.heavy = heavy_get(self.data[:, 1])
        self.light = light_get(self.data[:, 2])
        self.antigen = antigen_get(self.data[:, 3])
        self.dar = self.data[:, 6]
        self.label = self.data[:,7]
        self.adc = self.Vitrual_ADC_graph(self.adcid, self.dar)
        self.dar = np.expand_dims(self.dar, axis=1)

        self.fused_vector_reduced =self.data[:, 8:]#Component features after dimensionality reduction.

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.heavy[idx],
            self.light[idx],
            self.antigen[idx],
            self.payload[idx],
            self.linker[idx],
            self.dar[idx],
            self.label[idx],
            self.adc[idx],
            self.fused_vector_reduced[idx]
        )


    def Vitrual_ADC_graph(self, adcid, dars):
        """
        Bulid virtrual ADC graph
        """
        adc_graph = []
        for dar in dars:
            g = dgl.DGLGraph()
            g.add_nodes(7)
            x = torch.randn(7, 128)
            src_list = [0,1,1,1,1,2,3,4,5,2,4,3]
            dst_list = [1,2,3,4,5,6,6,6,6,4,5,5]
            g.add_edges(src_list, dst_list)
            g = dgl.to_bidirected(g)
            g.ndata['atom'] = x
            g.edata['bond'] = torch.randn(24, 5)
            g.edata['bond'][g.edge_ids([0, 1], [1, 0])] = torch.Tensor([1,0,0,0,0]).repeat(2, 1)
            g.edata['bond'][g.edge_ids([1,1,1,1,2,3,4,5], [2,3,4,5,1,1,1,1])] =  torch.Tensor([0,1,0,0,dar]).repeat(8, 1)
            g.edata['bond'][g.edge_ids([2,3,4,4,5,5], [4,5,2,5,4,3])] =  torch.Tensor([0,0,1,0,0]).repeat(6, 1)
            g.edata['bond'][g.edge_ids([2,3,4,5,6,6,6,6], [6,6,6,6,2,3,4,5])] = torch.Tensor([0,0,0,1,0]).repeat(8, 1)
            g = dgl.add_nodes(g, 1)
            for i in range(g.num_nodes()-1):
                g = dgl.add_edges(g, i, g.num_nodes()-1)
                g = dgl.add_edges(g, g.num_nodes()-1, i)
            g = laplacian_positional_encoding(g, pos_enc_dim=8)
            adc_graph.append(g)
        return adc_graph

    def collate(self, sample):
        heavy, light, antigen, payload, linker, dar, label, adc, fused_vector_reduced = map(list, zip(*sample))
        adc_graphs = dgl.batch(adc)
        payload_graphs= dgl.batch(payload)
        linker_graphs= dgl.batch(linker)
        # dar = np.array(dar)
        dar = torch.FloatTensor(dar)
        # label = torch.LongTensor(label)
        label = torch.FloatTensor(label)#bce
        fused_vector_reduced = torch.FloatTensor(fused_vector_reduced)
        return heavy, light, antigen, payload_graphs, linker_graphs, dar, label, adc_graphs, fused_vector_reduced



