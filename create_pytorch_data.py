import pandas as pd
import pickle
from aev_plig.create_pytorch_data import GraphDataset

if __name__ == "__main__":
    print("loading graph from pickle file for pdbbind2020")
    with open("data/pdbbind.pickle", 'rb') as handle:
        pdbbind_graphs = pickle.load(handle)

    print("loading graph from pickle file for BindingNet")
    with open("data/bindingnet.pickle", 'rb') as handle:
        bindingnet_graphs = pickle.load(handle)

    print("loading graph from pickle file for BindingDB")
    with open("data/bindingdb.pickle", 'rb') as handle:
        bindingdb_graphs = pickle.load(handle)

    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}

    pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
    pdbbind = pdbbind[['PDB_code', '-logKd/Ki', 'split_core', 'max_tanimoto_fep_benchmark']]
    pdbbind = pdbbind.rename(columns={'PDB_code': 'unique_id', 'split_core': 'split', '-logKd/Ki': 'pK'})
    pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]
    pdbbind = pdbbind[['unique_id', 'pK', 'split']]

    bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
    bindingnet = bindingnet.rename(columns={'-logAffi': 'pK', 'unique_identify': 'unique_id'})[['unique_id', 'pK', 'max_tanimoto_fep_benchmark']]
    bindingnet['split'] = 'train'
    bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.9]
    bindingnet = bindingnet[['unique_id', 'pK', 'split']]

    bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
    bindingdb = bindingdb[['unique_id', 'pK', 'max_tanimoto_fep_benchmark']]
    bindingdb['split'] = 'train'
    bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9]
    bindingdb = bindingdb[['unique_id', 'pK', 'split']]

    data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
    print(data[['split']].value_counts())

    dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'

    for split in ['train', 'valid', 'test']:
        df = data[data['split'] == split]
        ids, y = list(df['unique_id']), list(df['pK'])
        print(f'preparing {dataset}_{split}.pt in pytorch format!')
        GraphDataset(root='data', dataset=f'{dataset}_{split}', ids=ids, y=y, graphs_dict=graphs_dict)
