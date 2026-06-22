import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import Chem
from aev_plig import graph_generator

"""
Load data
"""
df = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
folder = "data/bindingnet/from_chembl_client/"

"""
Generate for all complexes: ANI-2x with 22 atom types. Only 2-atom interactions.
"""
atom_keys, atom_map, radial_coefs = graph_generator._common_setup()

mol_graphs = {}
for index, row in tqdm(df.iterrows()):
    unique_identify = row['unique_identify']
    target = row['target']
    pdb = row['pdb']
    compnd = row['compnd']

    sdf_file = folder + f"{pdb}/target_{target}/{compnd}/{pdb}_{target}_{compnd}.sdf"
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    lig = suppl[0]

    protein_path = folder + f"{pdb}/rec_h_opt.pdb"

    mol_df, aevs = graph_generator.GetMolAEVs_extended(protein_path, lig, atom_keys, radial_coefs, atom_map)
    graph = graph_generator.mol_to_graph(lig, mol_df, aevs)
    mol_graphs[unique_identify] = graph

output_file_graphs = "data/bindingnet.pickle"
with open(output_file_graphs, 'wb') as handle:
    pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
