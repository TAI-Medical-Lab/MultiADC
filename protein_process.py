import torch
import esm

import numpy as np
import matplotlib.pyplot as plt
import json, pickle
from collections import OrderedDict
import os

import pandas as pd
from tqdm import tqdm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

csv_paths = ["dataset/proteins.csv", "dataset/aug_proteins.csv"]
unique_sequences = set()
for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        unique_sequences.add(row["sequence"])


print(len(unique_sequences))
def get_esm_embeddings(seqs, batch_size=16):
    embeddings = {}
    for i in tqdm(range(0, len(seqs), batch_size), total=(len(seqs)+batch_size-1)//batch_size, desc="ESM Encoding"):
        # batch = seqs[i:i+batch_size]
        batch = [seq.upper() for seq in seqs[i:i+batch_size]]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch)]


        labels, strs, tokens = batch_converter(data)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33]
        batch_lens = (tokens != alphabet.padding_idx).sum(1)

        for j, tokens_len in enumerate(batch_lens):
            emb = token_representations[j, 1 : tokens_len - 1].mean(0).cpu()
            embeddings[batch[j]] = emb

    return embeddings


seq_list = list(unique_sequences)
for seq in seq_list:
    if any(c not in "ARNDCEQGHILKMFPSTWYV" for c in seq):
        print(f"Invalid sequence: {seq}")

embeddings = get_esm_embeddings(seq_list, batch_size=1)


output_path = "protein_1280.pkl"
with open(output_path, "wb") as f:
    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

file_path = 'protein_1280.pkl'

with open(file_path, 'rb') as file:
    seq2tensor = pickle.load(file)

print(seq2tensor['DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'])