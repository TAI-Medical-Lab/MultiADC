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
from ADCDataset import MultiADC_Dataset,Dim_Reduct_Data
from model.net import MultiADC
from loss import DiceLoss,BCEFocalLoss
from sklearn.utils import resample

# from model.covae import net
from loss import FocalLoss
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
 
from sklearn.model_selection import KFold



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

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    sample_num = 0
    total_preds = []
    total_labels = []
    for batch_idx, data in enumerate(train_loader):
        heavy, light, antigen, playload_graph, linker_graph, dar, label ,adc_graph,components= data[:]
        heavy = torch.tensor(np.array(heavy)).to(device)
        light = torch.tensor(np.array(light)).to(device)
        antigen = torch.tensor(np.array(antigen)).to(device)
        playload_graph = playload_graph.to(device)
        linker_graph = linker_graph.to(device)
        dar = dar.to(device)
        label = label.to(device)
        label = label.unsqueeze(1)
        components = components.to(device)
        adc_graph=adc_graph.to(device)

        output = model(heavy=heavy, light=light, antigen=antigen,
                       playload_graph=playload_graph, linker_graph=linker_graph, dar=dar,adc_graph=adc_graph,components=components)
        
        batch_loss = criterion(output, label)

        total_loss += batch_loss.item() * label.size(0)
        sample_num += label.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        preds_score = torch.sigmoid(output).to('cpu').data.numpy() #BCE
        # print(preds_score)
        preds_score = preds_score.flatten()  
        total_preds.extend(preds_score)
        total_labels.extend(label.cpu().numpy())

    train_loss = total_loss / sample_num
    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV=score(total_labels, total_preds)
    # print('traindataset {},{},{},{}'.format(tp, tn, fn, fp))
    return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV,train_loss

def test(model, device, test_loader):
    model.eval()
    total_preds = []
    total_labels = []
    sample_num = 0
    total_loss=0
    with torch.no_grad():
        for data in test_loader:
            heavy, light, antigen, playload_graph, linker_graph, dar, label ,adc_graph,components= data[:]
            heavy = torch.tensor(np.array(heavy)).to(device)
            light = torch.tensor(np.array(light)).to(device)
            antigen = torch.tensor(np.array(antigen)).to(device)
            playload_graph = playload_graph.to(device)
            linker_graph = linker_graph.to(device)
            components = components.to(device)
            dar = dar.to(device)
            label = label.to(device)
            label = label.unsqueeze(1)#BCE
            adc_graph=adc_graph.to(device)

            output = model(heavy=heavy, light=light, antigen=antigen,
                       playload_graph=playload_graph, linker_graph=linker_graph, dar=dar,adc_graph=adc_graph,components=components)
            batch_loss = criterion(output, label)

            total_loss += batch_loss.item() * output.size(0)
            sample_num += output.size(0)

            preds_score = torch.sigmoid(output).to('cpu').data.numpy() 
            preds_score = preds_score.flatten()  
            total_preds.extend(preds_score)
            total_labels.extend(label.cpu().numpy())


    tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV = score(total_labels, total_preds)
    test_loss = total_loss / sample_num
    return  test_loss,tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV



if __name__ == '__main__':
    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    file_path = 'dataset/processed/'
    log_file = 'logs/' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    results_file = 'results/' + str(time.strftime("%m%d-%H%M", time.localtime())) + '.txt'
    os.makedirs('results/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)


    batch = 64
    lr=0.00005
    k_fold = 5
    Patience = 30
    epochs = 300
    se_list = []
    sp_list = []
    mcc_list = []
    acc_list = []
    auc_list = []
    F1_list = []
    BA_list = []
    prauc_list = []
    PPV_list = []
    NPV_list = []

    df = pd.read_csv('dataset/dataset.csv')
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=SEED)
    pca_processor = Dim_Reduct_Data(n_components=128)

    #Standardize and normalize the DAR column
    dar_data = df.iloc[:, -2].values.reshape(-1, 1)
    scaler = StandardScaler()
    dar_data_standardized = scaler.fit_transform(dar_data)
    dar_data_normalized = normalize(dar_data_standardized, axis=0).flatten()
    df.iloc[:, -2] = dar_data_normalized

    print('Length of dataset:', len(df))
    i_fold=0
    for train_index, val_index in kf.split(df):
        
        i_fold += 1

        train_fold = df.iloc[train_index]
        test_fold = df.iloc[val_index]
        train_fold, val_fold = train_test_split(train_fold, test_size=30,random_state=SEED)
        #You can split a validation set to implement an early stopping strategy during training. 
        #However, due to the limited number of samples, we do not recommend creating a separate validation set. 
        #If you still choose to split one, we suggest setting the size of the validation set to a relatively small value.
        
        label_counts = train_fold['label'].value_counts()
        if label_counts[0] < label_counts[1]:
            minority = train_fold[train_fold['label'] == 0]
            majority = train_fold[train_fold['label'] == 1]
        else:
            minority = train_fold[train_fold['label'] == 1]
            majority = train_fold[train_fold['label'] == 0]

        # Upsample the minority class.
        minority_upsampled = resample(minority,
                                    replace=True,     
                                    n_samples=len(majority),  
                                    random_state=SEED)  
        train_fold_balanced = pd.concat([majority, minority_upsampled])
        train_fold_balanced = train_fold_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

        train_fold_processed = pca_processor.fit_transform(train_fold_balanced)
        test_fold_processed = pca_processor.transform(test_fold)
        val_fold_processed = pca_processor.transform(val_fold)

        train_set = MultiADC_Dataset(train_fold_processed)
        test_set = MultiADC_Dataset(test_fold_processed)
        val_set = MultiADC_Dataset(val_fold_processed)

        train_loader = DataLoader(train_set, batch_size=batch, shuffle=False, collate_fn=train_set.collate, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=False)

        model = MultiADC(device=device,compound_dim=128, protein_dim=128, gt_layers=3, gt_heads=4, out_dim=1)
        model.to(device)

        best_ci =0
        best_auc = 0
        best_rm2 = 0
        best_mcc = 0
        best_epoch = -1
        patience = 0
        
        metric_dict = {'se':0, 'sp':0, 'mcc':0, 'acc':0, "auc":0, 'F1':0, 'BA':0, 'prauc':0, 'PPV':0, 'NPV':0}

        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.01)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.9, patience=30, verbose=True, min_lr=1e-5)
        scheduler=optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10,20,30], gamma=0.8, last_epoch=-1)
        
        # criterion = BCEFocalLoss(gamma=2, alpha=0.4)
        # criterion = BCELoss(gamma=2, alpha=0.4)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()

        print('Start Training.')
        for epoch in range(epochs):
            train_tp, train_tn, train_fn, train_fp, train_se, train_sp, train_mcc, train_acc, train_auc_roc, train_F1, train_BA, train_prauc, train_PPV, train_NPV,train_train_loss=train(model, device, train_loader, optimizer,criterion)
            val_loss,val_tp, val_tn, val_fn, val_fp, val_se, val_sp, val_mcc, val_acc, val_auc_roc, val_F1, val_BA, val_prauc, val_PPV, val_NPV = test(model, device, val_loader)
            test_loss,test_tp, test_tn, test_fn, test_fp, test_se, test_sp, test_mcc, test_acc, test_auc_roc, test_F1, test_BA, test_prauc, test_PPV, test_NPV = test(model, device, test_loader)
            
            # scheduler.step(train_auc_roc)
            scheduler.step()

            with open(log_file, 'a') as f:
                f.write(str(time.strftime("%m-%d %H:%M:%S", time.localtime())) + ' epoch:' + str(epoch+1) + ' test_loss' + str(test_loss) + ' se:' + str(round(test_se,4)) +' '+ 'sp:' + str(
                    round(test_sp,4)) + ' ' + 'mcc:' + str(round(test_mcc,4)) +' acc:'+str(round(test_acc,4)) +' auc:'+str(round(test_auc_roc,4)) + ' F1:'+str(round(test_F1,4)) + ' BA:'+str(round(test_BA,4))+ ' prauc:'+str(round(test_prauc,4))+
                    ' PPV:'+str(round(test_PPV,4))+ ' NPV:'+str(round(test_NPV,4))+ '\n')

                print('epoch:' + str(epoch+1) + ' test_loss' + str(test_loss) + ' se:' + str(round(test_se,4)) +' '+ 'sp:' + str(
                    round(test_sp,4)) + ' ' + 'mcc:' + str(round(test_mcc,4)) +' acc:'+str(round(test_acc,4)) +' auc:'+str(round(test_auc_roc,4)) + ' F1:'+str(round(test_F1,4)) + ' BA:'+str(round(test_BA,4))+ ' prauc:'+str(round(test_prauc,4))+
                    ' PPV:'+str(round(test_PPV,4))+ ' NPV:'+str(round(test_NPV,4)))

            if val_mcc > best_mcc:

                best_epoch = epoch + 1
                # best_auc = auc_roc_score
                best_mcc=val_mcc
                patience = 0
                with open(log_file, 'a') as f:
                    f.write('MCC improved at epoch ' + str(best_epoch) + ' best_mcc:' + str(round(best_mcc,4)) + '\n')
                print('MCC improved at epoch ' + str(best_epoch) + ' best_mcc:' + str(round(best_mcc,4)) )
            else:
                patience += 1

            if patience == Patience:
                # Early stop
                metric_dict['se'] = test_se
                metric_dict['sp'] = test_sp
                metric_dict['mcc'] = test_mcc
                metric_dict['acc'] = test_acc
                metric_dict['auc'] = test_auc_roc
                metric_dict['F1'] = test_F1
                metric_dict['BA'] = test_BA
                metric_dict['prauc'] = test_prauc
                metric_dict['PPV'] = test_PPV
                metric_dict['NPV'] = test_NPV
                break

        #Save the results of the test set for each fold. 
        se_list.append(metric_dict['se'])
        sp_list.append(metric_dict['sp'])
        mcc_list.append(metric_dict['mcc'])
        acc_list.append(metric_dict['acc'])
        auc_list.append(metric_dict['auc'])
        F1_list.append(metric_dict['F1'])
        BA_list.append(metric_dict['BA'])
        prauc_list.append(metric_dict['prauc'])
        PPV_list.append(metric_dict['PPV'])
        NPV_list.append(metric_dict['NPV'])

        with open(log_file, 'a') as f:
            f.write('Fold ' + str(i_fold ) + '---' + 'se:' + str(round(metric_dict['se'],4)) +' '+ 'sp:' + str(
                        round(metric_dict['sp'],4)) + ' ' + 'mcc:' + str(round(metric_dict['mcc'],4)) +' acc:'+str(round(metric_dict['acc'],4))+
                        ' auc:'+str(round(metric_dict['auc'],4)) + ' F1:'+str(round(metric_dict['F1'],4)) +' BA:'+str(round(metric_dict['BA'],4)) +
                        ' prauc:'+str(round(metric_dict['prauc'],4)) +' PPV:'+str(round(metric_dict['PPV'],4)) +' NPV:'+str(round(metric_dict['NPV'],4)) +'\n')
        with open(results_file, 'a') as f:
            f.write('Fold ' + str(i_fold ) + '---' + 'se:' + str(round(metric_dict['se'],4)) +' '+ 'sp:' + str(
                        round(metric_dict['sp'],4)) + ' ' + 'mcc:' + str(round(metric_dict['mcc'],4)) +' acc:'+str(round(metric_dict['acc'],4))+
                        ' auc:'+str(round(metric_dict['auc'],4)) + ' F1:'+str(round(metric_dict['F1'],4)) +' BA:'+str(round(metric_dict['BA'],4)) +
                        ' prauc:'+str(round(metric_dict['prauc'],4)) +' PPV:'+str(round(metric_dict['PPV'],4)) +' NPV:'+str(round(metric_dict['NPV'],4)) +'\n')

    #Average 
    se_mean, se_var = np.mean(se_list), np.sqrt(np.var(se_list))
    sp_mean, sp_var = np.mean(sp_list), np.sqrt(np.var(sp_list))
    mcc_mean, mcc_var = np.mean(mcc_list), np.sqrt(np.var(mcc_list))
    acc_mean, acc_var = np.mean(acc_list), np.sqrt(np.var(acc_list))
    auc_mean, auc_var = np.mean(auc_list), np.sqrt(np.var(auc_list))
    F1_mean, F1_var = np.mean(F1_list), np.sqrt(np.var(F1_list))
    BA_mean, BA_var = np.mean(BA_list), np.sqrt(np.var(BA_list))
    prauc_mean, prauc_var = np.mean(prauc_list), np.sqrt(np.var(prauc_list))
    PPV_mean, PPV_var = np.mean(PPV_list), np.sqrt(np.var(PPV_list))
    NPV_mean, NPV_var = np.mean(NPV_list), np.sqrt(np.var(NPV_list))
    
    with open(results_file, 'a') as f:
        #Save the final results.
        f.write(f'Mean results: se:{se_mean:.4f}({se_var:.4f}) sp:{sp_mean:.4f}({sp_var:.4f}) mcc:{mcc_mean:.4f}({mcc_var:.4f}) acc:{acc_mean:.4f}({acc_var:.4f}) auc:{auc_mean:.4f}({auc_var:.4f}) F1:{F1_mean:.4f}({F1_var:.4f}) BA:{BA_mean:.4f}({BA_var:.4f}) prauc:{prauc_mean:.4f}({prauc_var:.4f}) PPV:{PPV_mean:.4f}({PPV_var:.4f}) NPV:{NPV_mean:.4f}({NPV_var:.4f})')