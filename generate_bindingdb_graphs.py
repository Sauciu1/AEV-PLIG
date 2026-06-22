import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import Chem
from aev_plig import graph_generator

"""
Load data
"""
df = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
folder = "data/bindingdb/surflex/"

"""
Generate for all complexes: ANI-2x with 22 atom types. Only 2-atom interactions.
"""
atom_keys, atom_map, radial_coefs = graph_generator._common_setup()

mol_graphs = {}
for index, row in tqdm(df.iterrows()):
    mol2_file = folder + row["folder"] + "/" + row["mol2_file"]
    lig = Chem.MolFromMol2File(mol2_file)
    lig = Chem.AddHs(lig, addCoords=True)

    protein_path = folder + row["folder"] + "/" + row["pdb_file"]

    mol_df, aevs = graph_generator.GetMolAEVs_extended(protein_path, lig, atom_keys, radial_coefs, atom_map)
    graph = graph_generator.mol_to_graph(lig, mol_df, aevs)
    mol_graphs[row["unique_id"]] = graph

output_file_graphs = "data/bindingdb.pickle"
with open(output_file_graphs, 'wb') as handle:
    pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
