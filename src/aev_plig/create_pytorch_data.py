import pandas as pd
import pickle
import os
import argparse

from torch_geometric.data import InMemoryDataset, Data
from torchani_mod.utils import GraphDataset as BaseGraphDataset
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_graphs_from_folder(folder: str = "data/aev_plig/graphs"):
    """Load pickled graphs from a folder"""
    files = [
        file
        for file in os.listdir(folder)
        if file.endswith(".pickle") or file.endswith(".pkl")
    ]
    if not files:
        raise ValueError(f"No pickle files found in folder {folder}")
    graphs = {}

    for file in files:
        with open(os.path.join(folder, file), "rb") as handle:
            graphs.update(pickle.load(handle))
    if not graphs:
        raise ValueError(f"Pickling failed to load any graphs from folder {folder}")
    return graphs


class GraphDataset(BaseGraphDataset):
    def __init__(
        self,
        root="data",
        dataset=None,
        ids=None,
        y=None,
        graphs_dict=None,
        y_scaler=None,
        regenerate=True,
        **kwargs,
    ):
        super(GraphDataset, self).__init__(root, dataset, ids, y, graphs_dict, y_scaler)

        torch.serialization.add_safe_globals([Data])
        if not regenerate and os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])
        else:
            self.process(ids, y, graphs_dict)
            self.load(self.processed_paths[0])

        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self._data.y, (self.__len__(), 1)))
        self.y_scaler = y_scaler
        self._data.y = [torch.tensor(element[0]).float() for element in self.y_scaler.transform(np.reshape(self._data.y, (self.__len__(), 1)))]

    def process(self, ids, y, graphs_dict):
        assert len(ids) == len(y), "Number of datapoints and labels must be the same"
        data_list = []
        for i in range(len(ids)):
            pdbcode = ids[i]
            label = y[i]
            _, features, edge_index, edge_features = graphs_dict[pdbcode]

            data_point = Data(
                x=torch.Tensor(np.array(features)),
                edge_index=torch.LongTensor(np.array(edge_index)).T,
                edge_attr=torch.Tensor(np.array(edge_features)),
                y=torch.FloatTensor(np.array([label])),
            )
            data_list.append(data_point)

        print("Graph construction done. Saving to file.")
        self.save(data_list, self.processed_paths[0])


def save_splits(
    df: pd.DataFrame,
    name: str,
    output_folder: str = "data",
    regenerate: bool = False,
) -> None:
    """Save train/valid/test splits in PyTorch format"""
    graphs_dict = load_graphs_from_folder()

    for split in ["train", "valid", "test"]:
        if split not in df["split"].values:
            print(f"Warning: split '{split}' not found in DataFrame. Skipping.")
            continue

        graph_name = f"{name}_{split}"
        print(f"preparing {graph_name} in pytorch format!")
        split_df = df[df["split"] == split]
        missing_samples = split_df[~split_df["unique_id"].isin(graphs_dict.keys())]
        split_df = split_df[split_df["unique_id"].isin(graphs_dict.keys())]

        ids, y = list(split_df["unique_id"]), list(split_df["pK"])
        GraphDataset(
            root=output_folder,
            dataset=graph_name,
            ids=ids,
            y=y,
            graphs_dict=graphs_dict,
            regenerate=regenerate,
        )

        print(f"Saved PyTorch datasets to {graph_name}")
        print(f"Finished processing {graph_name} of length {len(split_df)}")
        if not missing_samples.empty:
            print(
                f"Warning: {len(missing_samples)} samples were missing from the graphs dictionary and were skipped."
            )
            print(missing_samples["unique_id"].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PyTorch datasets for AEV-PLIG")
    parser.add_argument(
        "--csv_path", type=str, default="datasets/raw_datasets/bindingdb_processed.csv"
    )
    parser.add_argument("--output_name", type=str, default="trial_run")
    parser.add_argument("--output_folder", type=str, default="data")
    parser.add_argument(
        "--no_regenerate",
        action="store_true",
        help="Do not regenerate the datasets even if they already exist",
    )
    args = parser.parse_args()

    training_data = pd.read_csv(args.csv_path)
    save_splits(
        training_data,
        args.output_name,
        output_folder=args.output_folder,
        regenerate=not args.no_regenerate,
    )
    print("Done!")
