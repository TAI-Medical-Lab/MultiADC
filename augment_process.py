import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm  # for progress bar (optional)
import random
from typing import List, Tuple, Dict, Union
import csv
import os
import re
REPLACEMENT_DICT = {
    'A': ['V'],  # 丙氨酸→缬氨酸
    'S': ['T'],  # 丝氨酸→苏氨酸
    'F': ['Y'],  # 苯丙氨酸→酪氨酸
    'K': ['R'],  # 赖氨酸→精氨酸
    'C': ['M'],  # 半胱氨酸→蛋氨酸
    'D': ['E'],  # 天冬氨酸→谷氨酸
    'N': ['Q'],  # 天冬酰胺→谷氨酰胺
    'V': ['I']   # 缬氨酸→异亮氨酸
}
PROTEIN_AUG_METHODS = ['RD', 'RA', 'GRS', 'LRS', 'SR', 'SS', 'RD&LRS', 'RA&SS']
PROTEIN_AUG_PROBS = [0.2, 0.08, 0.08, 0.08, 0.08, 0.08, 0.2, 0.2]

def randomize_smiles(sml, seed=None):
    """Generate randomized SMILES for a given molecule"""
    try:
        if seed is not None:
            np.random.seed(seed)
        m = Chem.MolFromSmiles(sml)
        if m is None:
            return np.nan
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        randomized_mol = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(randomized_mol, canonical=False)
    except:
        return np.nan


def get_aug_smiles(input_csv,output_csv):

    df = pd.read_csv(input_csv)
    # Prepare output data
    output_data = []
    # Process each molecule
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_id = row['id']
        original_smiles = row['sequence']
        

        # output_data.append({
        #     'id': original_id,  
        #     'sequence': original_smiles,
        #     'variant_id': 'origin'
        # })
        
        for i in range(1, 11):  # variant_num = 1~9
            randomized_smiles = randomize_smiles(original_smiles, seed=i)  
            
            output_data.append({
                'id': original_id,  
                'sequence': randomized_smiles,
                'variant_id': f'rand_{i}'
            })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)

    print(f"Generated {len(output_df)} randomized SMILES saved to {output_csv}")

def replacement_augmentation(
    sequence: str,
    p: float = 0.1,
    mode: str = 'RD'
) -> str:
    augmented = []
    for aa in sequence:
        if random.random() < p:
            if mode == 'RD' and aa in REPLACEMENT_DICT:
                new_aa = random.choice(REPLACEMENT_DICT[aa])
            elif mode == 'RA':
                new_aa = 'A'
            else:
                new_aa = aa
            augmented.append(new_aa)
        else:
            augmented.append(aa)
    return ''.join(augmented)

def shuffling_augmentation(
    sequence: str,
    mode: str = 'global',
    local_window: int = 50
) -> str:
    seq_list = list(sequence)
    n = len(seq_list)
    if mode == 'global':
        shuffled = random.sample(seq_list, n)
    elif mode == 'local' and n >= 2:
        start = random.randint(0, max(0, n - local_window - 1))
        end = min(start + local_window, n)
        window = seq_list[start:end]
        random.shuffle(window)
        shuffled = seq_list[:start] + window + seq_list[end:]
    else:
        shuffled = seq_list.copy()
    return ''.join(shuffled)

def sequence_reversion(sequence: str) -> str:
    return sequence[::-1]

def subsampling_augmentation(
    sequence: str,
    window_length: int = 50
) -> str:
    n = len(sequence)
    if n <= window_length:
        return sequence
    start = random.randint(0, n // 2)
    end = min(start + window_length, n)
    return sequence[start:end]


def apply_combined_augmentation(sequence: str, aug1: str, aug2: str, p: float = 0.1, window: int = 50):
    if aug1 == 'RD':
        seq = replacement_augmentation(sequence, p=p, mode='RD')
    elif aug1 == 'RA':
        seq = replacement_augmentation(sequence, p=p, mode='RA')
    else:
        seq = sequence

    if aug2 == 'LRS':
        seq = shuffling_augmentation(seq, mode='local', local_window=window)
    elif aug2 == 'SS':
        seq = subsampling_augmentation(seq, window_length=window)
    elif aug2 == 'GRS':
        seq = shuffling_augmentation(seq, mode='global')
    elif aug2 == 'SR':
        seq = sequence_reversion(seq)

    return seq

def get_aug_proteins(input_file: str, output_file: str, p: float = 0.01, window: int = 50):
    augmented_data = []

    with open(input_file, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            original_id = row['id']
            original_seq = row['sequence']


            # augmented_data.append({
            #     'id': original_id,
            #     'newsequence': original_seq,
            #     'method': 'original'
            # })

            methods = {
                'RD': lambda: replacement_augmentation(original_seq, p=p, mode='RD'),
                'RA': lambda: replacement_augmentation(original_seq, p=p, mode='RA'),
                'GRS': lambda: shuffling_augmentation(original_seq, mode='global'),
                'LRS': lambda: shuffling_augmentation(original_seq, mode='local', local_window=window),
                'SR': lambda: sequence_reversion(original_seq),
                'SS': lambda: subsampling_augmentation(original_seq, window_length=window),
                'RD&LRS': lambda: apply_combined_augmentation(original_seq, 'RD', 'LRS', p=p, window=window),
                'RA&SS': lambda: apply_combined_augmentation(original_seq, 'RA', 'SS', p=p, window=window),
            }

            for method_name, func in methods.items():
                new_seq = func()
                augmented_data.append({
                    'id': original_id,
                    'sequence': new_seq,
                    'method': method_name
                })

    with open(output_file, 'w', newline='') as outfile:
        fieldnames = ['id', 'sequence', 'method']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(augmented_data)

SEED=999
random.seed(SEED)
np.random.seed(SEED)

def generate_new_id_init(original_id, heavy_aug, light_aug, antigen_aug, payload_aug, linker_aug):
    return f"{original_id}_h_{heavy_aug}_lg_{light_aug}_a_{antigen_aug}_p_{payload_aug}_lk_{linker_aug}"

def generate_new_id(original_full_id, heavy_aug=None, light_aug=None, antigen_aug=None, payload_aug=None, linker_aug=None):
    """
    Generate New ID: Parse existing enhancement tags from the `original_full_id`, and only replace fields that are not None.

    Parameters:
        original_full_id (str): e.g., 'DRG0FBKIO_h_ori_lg_SS_a_GR_p_rand_1_lk_ori'
        *_aug: If None, keep the original value; otherwise, replace with the new value
    """
    base_id = original_full_id[:9]
    
    def extract_field(pattern):
        match = re.search(pattern, original_full_id)
        return match.group(1) if match else "ori"
    
    current_h = extract_field(r'_h_([^_]+)')
    current_lg = extract_field(r'_lg_([^_]+)')
    current_a = extract_field(r'_a_([^_]+)')
    current_p = extract_field(r'_p_([^_]+)')
    current_lk = extract_field(r'_lk_([^_]+)')

    h = heavy_aug if heavy_aug is not None else current_h
    lg = light_aug if light_aug is not None else current_lg
    a = antigen_aug if antigen_aug is not None else current_a
    p = payload_aug if payload_aug is not None else current_p
    lk = linker_aug if linker_aug is not None else current_lk

    return f"{base_id}_h_{h}_lg_{lg}_a_{a}_p_{p}_lk_{lk}"


def extract_number(filename):
    match = re.search(r'part_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Can not extract number from {filename}")

def get_augonly_adc(ratio,pro_file,smi_file,aug_pro_file,aug_smi_file,adc_spilt_folder,only_smi=False,only_pro=False):

    protein_df = pd.read_csv(pro_file)
    smile_df = pd.read_csv(smi_file)
    aug_protein_df = pd.read_csv(aug_pro_file)
    aug_smile_df = pd.read_csv(aug_smi_file)

    selected_protein_ids = set(random.sample(protein_df['id'].tolist(), k=int(len(protein_df)*ratio)))
    selected_smile_ids = set(random.sample(smile_df['id'].tolist(), k=int(len(smile_df)*ratio)))
    protein_aug_map = {}
    for _, row in aug_protein_df.iterrows():
        key = (row['id'], row['method'])
        if key not in protein_aug_map:
            protein_aug_map[key] = []
        protein_aug_map[key].append(row['sequence'])

    smile_aug_methods = ['rand_1', 'rand_3', 'rand_5','rand_8']
    smile_aug_probs = [0.25,0.25,0.25,0.25]


    smile_aug_map = {}
    for _, row in aug_smile_df.iterrows():
        key = (row['id'], row['method'])
        if key not in smile_aug_map:
            smile_aug_map[key] = []
        smile_aug_map[key].append(row['sequence'])

    protein_id_to_seq = dict(zip(protein_df['id'], protein_df['sequence']))
    smile_id_to_seq = dict(zip(smile_df['id'], smile_df['sequence']))
    items = os.listdir(adc_spilt_folder)
    print(items)

    items.sort(key=extract_number)

    for item in items:
        number = extract_number(item)
        augmented_data = []
        data_df = pd.read_csv(os.path.join(adc_spilt_folder, item))

        
        for _, original_row in data_df.iterrows():

            new_id = generate_new_id(original_row['id'], "ori", "ori", "ori", "ori", "ori")
            new_row = original_row.copy()
            new_row['id'] = new_id
            
            for field in ['heavy', 'light', 'antigen']:
                if new_row[field] in protein_id_to_seq:
                    new_row[field] = protein_id_to_seq[new_row[field]]
            
            for field in ['payload', 'linker']:
                if new_row[field] in smile_id_to_seq:
                    new_row[field] = smile_id_to_seq[new_row[field]]
            
            augmented_data.append(new_row)

        if not only_smi: 

            for _, original_row in data_df.iterrows():


                if original_row['heavy'] in selected_protein_ids:
                    method = np.random.choice(PROTEIN_AUG_METHODS, p=PROTEIN_AUG_PROBS)
                    key = (original_row['heavy'], method)
                    if key in protein_aug_map:
                        for seq in protein_aug_map[key]:
                            new_row = original_row.copy()
                            new_id = generate_new_id(original_row['id'], method, "ori", "ori", "ori", "ori")
                            new_row['id'] = new_id
                            new_row['heavy'] = seq
                            augmented_data.append(new_row)
                
                if original_row['light'] in selected_protein_ids:
                    method = np.random.choice(PROTEIN_AUG_METHODS, p=PROTEIN_AUG_PROBS)
                    key = (original_row['light'], method)
                    if key in protein_aug_map:
                        for seq in protein_aug_map[key]:
                            new_row = original_row.copy()
                            new_id = generate_new_id(original_row['id'], "ori", method, "ori", "ori", "ori")
                            new_row['id'] = new_id
                            new_row['light'] = seq
                            augmented_data.append(new_row)
                
                if original_row['antigen'] in selected_protein_ids:
                    method = np.random.choice(PROTEIN_AUG_METHODS, p=PROTEIN_AUG_PROBS)
                    key = (original_row['antigen'], method)
                    if key in protein_aug_map:
                        for seq in protein_aug_map[key]:
                            new_row = original_row.copy()
                            new_id = generate_new_id(original_row['id'], "ori", "ori", method, "ori", "ori")
                            new_row['id'] = new_id
                            new_row['antigen'] = seq
                            augmented_data.append(new_row)
            # print(len(augmented_data))
        if not only_pro:

            for _, original_row in data_df.iterrows():
                if original_row['payload'] in selected_smile_ids:
                    method = np.random.choice(smile_aug_methods, p=smile_aug_probs)
                    key = (original_row['payload'], method)
                    if key in smile_aug_map:
                        for seq in smile_aug_map[key]:
                            new_row = original_row.copy()
                            new_id = generate_new_id(original_row['id'], "ori", "ori", "ori", method, "ori")
                            new_row['id'] = new_id
                            new_row['payload'] = seq
                            augmented_data.append(new_row)
                
                if original_row['linker'] in selected_smile_ids:
                    method = np.random.choice(smile_aug_methods, p=smile_aug_probs)
                    key = (original_row['linker'], method)
                    if key in smile_aug_map:
                        for seq in smile_aug_map[key]:
                            new_row = original_row.copy()
                            new_id = generate_new_id(original_row['id'], "ori", "ori", "ori", "ori", method)
                            new_row['id'] = new_id
                            new_row['linker'] = seq
                            augmented_data.append(new_row)


        print(len(augmented_data))

        for row in augmented_data:
            if "_h_ori_lg_ori_a_ori_p_ori_lk_ori" in row['id']:
                continue
            
            for field in ['heavy', 'light', 'antigen']:
                if row[field] in protein_id_to_seq:
                    row[field] = protein_id_to_seq[row[field]]

            for field in ['payload', 'linker']:
                if row[field] in smile_id_to_seq:
                    row[field] = smile_id_to_seq[row[field]]

        final_df = pd.DataFrame(augmented_data)



        if only_smi and only_pro:
            final_df.to_csv(f'dataset/aug/aug{ratio}/all_{number}.csv', index=False)
        elif not only_smi: 
            final_df.to_csv(f'dataset/aug/aug{ratio}/only_pro_{number}.csv', index=False)
        else : 
            final_df.to_csv(f'dataset/aug/aug{ratio}/only_smi_{number}.csv', index=False)

        print(f"Data augmentation completed! Original number of samples: {len(data_df)}, number of samples after augmentation: {len(final_df)}")

def get_augmix_adc(ratio,pro_file,smi_file,aug_pro_file,aug_smi_file,adc_spilt_folder):
    protein_df = pd.read_csv(pro_file)
    smile_df = pd.read_csv(smi_file)
    aug_protein_df = pd.read_csv(aug_pro_file)
    aug_smile_df = pd.read_csv(aug_smi_file)

    selected_protein_ids = set(random.sample(protein_df['id'].tolist(), k=int(len(protein_df)*ratio)))
    selected_smile_ids = set(random.sample(smile_df['id'].tolist(), k=int(len(smile_df)*ratio)))
    protein_aug_map = {}
    for _, row in aug_protein_df.iterrows():
        key = (row['id'], row['method'])
        if key not in protein_aug_map:
            protein_aug_map[key] = []
        protein_aug_map[key].append(row['sequence'])

    smile_aug_methods = ['rand_1', 'rand_3', 'rand_5','rand_8']
    smile_aug_probs = [1/4, 1/4, 1/4,1/4]


    smile_aug_map = {}
    for _, row in aug_smile_df.iterrows():
        key = (row['id'], row['method'])
        if key not in smile_aug_map:
            smile_aug_map[key] = []
        smile_aug_map[key].append(row['sequence'])

    protein_id_to_seq = dict(zip(protein_df['id'], protein_df['sequence']))
    smile_id_to_seq = dict(zip(smile_df['id'], smile_df['sequence']))
    items = os.listdir(adc_spilt_folder)
    print(items)#['part_4.csv', 'part_5.csv', 'part_1.csv', 'part_2.csv', 'part_3.csv']

    items.sort(key=extract_number)

    # print(selected_protein_ids)
    for item in items:
        number = extract_number(item)
        augmented_data = []
        data_df = pd.read_csv(os.path.join(adc_spilt_folder, item))
        for _, original_row in data_df.iterrows():
            new_id = generate_new_id_init(original_row['id'], "ori", "ori", "ori", "ori", "ori")
            new_row = original_row.copy()
            new_row['id'] = new_id
            
            # for field in ['heavy', 'light', 'antigen']:
            #     if new_row[field] in protein_id_to_seq:
            #         new_row[field] = protein_id_to_seq[new_row[field]]
            
            # for field in ['payload', 'linker']:
            #     if new_row[field] in smile_id_to_seq:
            #         new_row[field] = smile_id_to_seq[new_row[field]]
            
            augmented_data.append(new_row)
        # print(len(augmented_data))

        data_df1 = pd.DataFrame(augmented_data)  
        # print(data_df1)
        for _, original_row in data_df1.iterrows():
            if original_row['heavy'] in selected_protein_ids:
                method = np.random.choice(PROTEIN_AUG_METHODS, p=PROTEIN_AUG_PROBS)
                key = (original_row['heavy'], method)
                if key in protein_aug_map:
                    for seq in protein_aug_map[key]:
                        new_row = original_row.copy()
                        new_id = generate_new_id(original_row['id'], heavy_aug=method)
                        new_row['id'] = new_id
                        new_row['heavy'] = seq
                        augmented_data.append(new_row)
            
            if original_row['light'] in selected_protein_ids:
                method = np.random.choice(PROTEIN_AUG_METHODS, p=PROTEIN_AUG_PROBS)
                key = (original_row['light'], method)
                if key in protein_aug_map:
                    for seq in protein_aug_map[key]:
                        new_row = original_row.copy()
                        new_id = generate_new_id(original_row['id'], light_aug=method)
                        new_row['id'] = new_id
                        new_row['light'] = seq
                        augmented_data.append(new_row)
            
            if original_row['antigen'] in selected_protein_ids:
                method = np.random.choice(PROTEIN_AUG_METHODS, p=PROTEIN_AUG_PROBS)
                key = (original_row['antigen'], method)
                if key in protein_aug_map:
                    for seq in protein_aug_map[key]:
                        new_row = original_row.copy()
                        new_id = generate_new_id(original_row['id'], antigen_aug=method)
                        new_row['id'] = new_id
                        new_row['antigen'] = seq
                        augmented_data.append(new_row)
        # print(len(augmented_data))
        data_df2 = pd.DataFrame(augmented_data)
        for _, original_row in data_df2.iterrows():

            if original_row['payload'] in selected_smile_ids:
                method = np.random.choice(smile_aug_methods, p=smile_aug_probs)
                key = (original_row['payload'], method)
                if key in smile_aug_map:
                    for seq in smile_aug_map[key]:
                        new_row = original_row.copy()
                        new_id = generate_new_id(original_row['id'], payload_aug= method)
                        new_row['id'] = new_id
                        new_row['payload'] = seq
                        augmented_data.append(new_row)
            
            if original_row['linker'] in selected_smile_ids:
                method = np.random.choice(smile_aug_methods, p=smile_aug_probs)
                key = (original_row['linker'], method)
                if key in smile_aug_map:
                    for seq in smile_aug_map[key]:
                        new_row = original_row.copy()
                        new_id = generate_new_id(original_row['id'], linker_aug= method)
                        new_row['id'] = new_id
                        new_row['linker'] = seq
                        augmented_data.append(new_row)


        print(len(augmented_data))

        for row in augmented_data:

            # if "_h_ori_lg_ori_a_ori_p_ori_lk_ori" in row['id']:
            #     continue
            
            for field in ['heavy', 'light', 'antigen']:
                if row[field] in protein_id_to_seq:
                    row[field] = protein_id_to_seq[row[field]]

            for field in ['payload', 'linker']:
                if row[field] in smile_id_to_seq:
                    row[field] = smile_id_to_seq[row[field]]

        final_df = pd.DataFrame(augmented_data)
        final_df.to_csv(f'dataset/aug/aug{ratio}/mix_{number}.csv', index=False)

        print(f"Data augmentation completed! Original number of samples: {len(data_df)}, number of samples after augmentation: {len(final_df)}")


if __name__ == "__main__":

    get_aug_smiles("dataset/molecules.csv","dataset/rand10_smiles.csv")
    get_aug_proteins('dataset/proteins.csv', 'dataset/aug_proteins.csv', p=0.1, window=50)
    # protein_seq = "MAGIHKSLEFTPQYVCNRDW"
    
    # #
    # print(protein_seq)
    # print("RD:", replacement_augmentation(protein_seq, p=0.3, mode='dictionary'))
    # print("RA:", replacement_augmentation(protein_seq, p=0.3, mode='alanine'))
    # print("GRS:", shuffling_augmentation(protein_seq, mode='global'))
    # print("LRS:", shuffling_augmentation(protein_seq, mode='local', local_window=5))
    # print("SR:", sequence_reversion(protein_seq))
    # print("SS:", subsampling_augmentation(protein_seq, min_length=10))
    
    #
    # config = {
    #     'replacement': {'mode': 'dictionary', 'p': 0.1},
    #     'shuffling': {'mode': 'local', 'window': 5},
    #     'subsample': {'min_length': 8},
    #     'reverse_prob': 0.3
    # }
    # print(apply_augmentations(protein_seq, config))
    pro_file='dataset/proteins.csv'
    smi_file='dataset/molecules.csv'
    aug_pro_file='dataset/aug_proteins.csv'
    aug_smi_file='dataset/rand10_smiles.csv'
    adc_spilt_folder='dataset/part'
    get_augonly_adc(0.1,pro_file,smi_file,aug_pro_file,aug_smi_file,adc_spilt_folder,only_smi=True,only_pro=False)
    get_augmix_adc(0.1,pro_file,smi_file,aug_pro_file,aug_smi_file,adc_spilt_folder)
