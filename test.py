import pandas as pd
import time
import os
import random
import numpy as np
import math
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_auc_score,confusion_matrix,precision_recall_curve,auc
# from metrics import *
from sklearn.utils import resample
import esm
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from loss import FocalLoss
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import KFold
# from ADCDataset2 import compound_graph_get_graph
from drug_process import smiles_to_graph
import joblib
import dgl
from scipy import sparse as sp
from model.net import MultiADC
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
def score(y_test, y_pred):
    if np.isnan(y_pred).any():
        return 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0,0,0
    auc_roc_score = roc_auc_score(y_test, y_pred)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    prauc = auc(recall, prec)
    y_pred_print = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    PPV = tp / (tp + fp)
    NPV = tn / (fn + tn)
    return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g
def vitrual_ADC_graph(dar):
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
    # adc_graph.append(g)
    return g


def get_esm(sequence, id='sequence'):
    

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data
    data = [('protein', sequence)]  # 构造一个包含ID和序列的元组列表

    # Convert the data into a format that can be fed into the model
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representation = token_representations[0, 1 : batch_lens[0] - 1].mean(0)

    return sequence_representation

def finger_get( smi):
    # morgan_list = []
    # for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print("Invalid SMILES string")
    else:
        morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        # morgan_list.append(morgan)
    return torch.tensor(np.array(morgan))


def pca_transform(input,filepath_scaler, filepath_pca):
    scaler = joblib.load(filepath_scaler)
    pca = joblib.load(filepath_pca)
    fused_vector_scaled = scaler.transform(input)
    fused_vector_reduced = pca.transform(fused_vector_scaled)
    return fused_vector_reduced


if __name__ == '__main__':
    """select seed"""
    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 
    heavy_seq = 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK'
    light_seq = 'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
    antigen_seq = 'MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRILKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGRLGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTFGAKPYDGIPAREIPDLLEKGERLPQPPICTIDVYMIMVKCWMIDSECRPRFRELVSEFSRMARDPQRFVVIQNEDLGPASPLDSTFYRSLLEDDDMGDLVDAEEYLVPQQGFFCPDPAPGAGGMVHHRHRSSSTRSGGGDLTLGLEPSEEEAPRSPLAPSEGAGSDVFDGDLGMGAAKGLQSLPTHDPSPLQRYSEDPTVPLPSETDGYVAPLTCSPQPEYVNQPDVRPQPPSPREGPLPAARPAGATLERPKTLSPGKNGVVKDVFAFGGAVENPEYLTPQGGAAPQPHPPPAFSPAFDNLYYWDQDPPERGAPPSTFKGTPTAENPEYLGLDVPV'
    playload_s = 'CCN(C(=O)CN)C1COC(OC2C(OC3C#C/C=C\C#CC4(O)CC(=O)C(NC(=O)OC)=C3/C4=C\CSSC(C)(C)CC(=O)NCCOCCOC)OC(C)C(NOC3CC(O)C(SC(=O)c4c(C)c(I)c(OC5OC(C)C(O)C(OC)C5O)c(OC)c4OC)C(C)O3)C2O)CC1OC'
    linker_s = 'CC(C)C(NC(=O)OCCN(CCOC(=O)NC(C(=O)NC(CCCNC(N)=O)C(=O)Nc1ccc(CO)cc1)C(C)C)S(=O)(=O)NC(=O)OCCOCCNS(=O)(=O)NC(=O)OCC1C2CCC#CCCC21)C(=O)NC(CCCNC(N)=O)C(=O)Nc1ccc(CO)cc1'
    dar_str = '1.86'  

    dar_value = torch.tensor([float(dar_str)], dtype=torch.float32)

    heavy_emb = get_esm(heavy_seq)
    light_emb = get_esm(light_seq)
    antigen_emb = get_esm(antigen_seq)
    playload = finger_get(playload_s)
    linker = finger_get(linker_s)
    device='cpu'
    input_data = torch.cat([
        heavy_emb.unsqueeze(0), 
        light_emb.unsqueeze(0), 
        antigen_emb.unsqueeze(0), 
        playload.unsqueeze(0), 
        linker.unsqueeze(0), 
        dar_value.unsqueeze(0)
    ], dim=1)  
    # pca_processor = ADCPCAn(n_components=128)
    # pca_processor.pca_transform('scaler.joblib', 'pca.joblib')
    pca_f = pca_transform(input_data,'scaler.joblib', 'pca.joblib') 

    pca_f=torch.FloatTensor(pca_f)
    playload_g = smiles_to_graph(playload_s)
    linker_g = smiles_to_graph(linker_s)
    adc_g = vitrual_ADC_graph(dar_value)

    model = MultiADC_test(device=device,compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=1).to(device)

    state_dict = torch.load('MultiADC_Model.pth', map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()  
    with torch.no_grad():
        output = model(
            heavy=heavy_emb,
            light=light_emb,
            antigen=antigen_emb,
            playload_graph=playload_g,
            linker_graph=linker_g,
            dar=dar_value,
            adc=adc_g,
            pca=pca_f
        )
        preds_class = torch.where(torch.sigmoid(output) >= 0.5, 1, 0).cpu().numpy()
        preds_class=preds_class.squeeze()
        preds_class=str(preds_class)
        print('This adc activity is：'+preds_class)