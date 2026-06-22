import pandas as pd
import pickle
import os
from tqdm import tqdm
from rdkit import Chem
from aev_plig import graph_generator

'''
Load data
'''
data = pd.read_csv("data/pdbbind_processed.csv", index_col=0)

'''
Generate for all complexes: ANI-2x with 22 atom types. Only 2-atom interactions
'''
print("The number of data points is ", len(data))

atom_keys, atom_map, radial_coefs = graph_generator._common_setup()

mol_graphs = {}
failed_list = []
failed_after_reading = []

for i, pdb in tqdm(enumerate(data["PDB_code"])):
    if data["refined"][i]:
        folder = "data/pdbbind/refined-set/"
    else:
        folder = "data/pdbbind/general-set/"

    mol_path = os.path.join(folder, pdb, f'{pdb}_ligand.mol2')
    mol = Chem.MolFromMol2File(mol_path)

    if mol is None:
        print("can't read molecule structure:", pdb)
        failed_list.append(pdb)
        continue
    else:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        protein_path = os.path.join(folder, pdb, f'{pdb}_protein.pdb')

        mol_df, aevs = graph_generator.GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map)
        graph = graph_generator.mol_to_graph(mol, mol_df, aevs)
        mol_graphs[pdb] = graph

    except ValueError as e:
        print(e)
        failed_after_reading.append(pdb)
        continue

print(len(failed_list), len(failed_after_reading))

output_file_graphs = "data/pdbbind.pickle"
with open(output_file_graphs, 'wb') as handle:
    pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
