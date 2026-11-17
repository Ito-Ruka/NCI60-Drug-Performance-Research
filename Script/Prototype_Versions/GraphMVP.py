import numpy as np
import pandas as pd
from rdkit import Chem
import os
import sys
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


def acessing_graphmvp_files():
    imports_dir = r"C:\Development\NCI-60_Work\Main_Code\GraphMVP\GraphMVP_github_imports"

    if imports_dir not in sys.path:
        sys.path.append(imports_dir)
    
    from molecule_datasets import mol_to_graph_data_obj_simple
    from molecule_gnn_model import GNNComplete

    return mol_to_graph_data_obj_simple, GNNComplete



def load_input_data():
    df_DOSERESP = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\DOSERESP.csv")
    df_NCI60_SMILES = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\nsc_smiles.csv")
    return df_DOSERESP, df_NCI60_SMILES



def get_unique_NSCs (df_DOSERESP):
    df_NSCs = df_DOSERESP[["NSC"]]
    df_unique_NSCs = df_NSCs.drop_duplicates().reset_index(drop = True)

    unique_NSC_count = len(df_unique_NSCs)
    
    return df_unique_NSCs, unique_NSC_count



def merge_SMILES_strings (df_unique_NSCs, df_NCI60_SMILES):
    df_NSC_with_SMILES = pd.merge(df_unique_NSCs, df_NCI60_SMILES, on = 'NSC', how = 'left')
    df_dropped_NSC_Numbers = df_NSC_with_SMILES[df_NSC_with_SMILES["SMILES"].isna()]["NSC"].unique()
    
    df_NSC_with_SMILES.dropna(subset=["SMILES"], inplace=True)
    df_NSC_with_SMILES.reset_index(drop=True, inplace=True)
    return df_NSC_with_SMILES, df_dropped_NSC_Numbers



def graphmvp_featurizer (df_NSC_with_SMILES):
    pyg_graph_list = []
    valid_nsc_list = []

    for idx, row in df_NSC_with_SMILES.iterrows():
        nscs = row["NSC"]
        smiles = str(row["SMILES"]).strip()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        data = mol_to_graph_data_obj_simple(mol)
        if (data is not None and data.x is not None and data.x.size(0) > 0):
            pyg_graph_list.append(data)
            valid_nsc_list.append(nscs)
    
    return pyg_graph_list, valid_nsc_list



def load_graphmvp_GNN (device, num_layer, emb_dim):
    num_layer = num_layer
    emb_dim = emb_dim
    JK = "last"
    gnn_type = "gin"

    graphmvp_model = GNNComplete(
        num_layer = num_layer, 
        emb_dim = emb_dim, 
        JK = JK, 
        gnn_type =  gnn_type
        ).to(device)
    
    weights = torch.load(r"C:\Development\NCI-60_Work\Main_Code\GraphMVP\GraphMVP_Intermeddiate_Output\pretraining_model_complete.pth", map_location = device, weights_only = False)

    def filter_graphmvp_weights (weights):
        if isinstance(weights, dict):
            if "model" in weights:
                state_dict = weights["model"]
            elif "molecule_model_2d" in weights:
                state_dict = weights["molecule_model_2d"]
            else:
                state_dict = weights
        else:
            state_dict = weights
        
        cleaned_dict = {}
        
        for k, v in state_dict.items():
            new_k = (
                k.replace("molecule_model_2d.", "")
                 .replace("molecule_model.", "")
                 .replace("encoder.", "")
            )
            
            cleaned_dict[new_k] = v

        return cleaned_dict
    
    cleaned_weights = filter_graphmvp_weights(weights)
    missing_keys, unexpected_keys = graphmvp_model.load_state_dict(cleaned_weights, strict = False)
    
    graphmvp_model.eval()
    return graphmvp_model
    


def extract_embeddings(model, pyg_graph_list, valid_nsc_list, device, batch_size = 64):
    loader = DataLoader(pyg_graph_list, batch_size = batch_size, shuffle = False)
    embedding_batches = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
            graph_reps = global_mean_pool(node_reps, batch.batch)
            embedding_batches.append(graph_reps.cpu().numpy())

    matrix = np.concatenate(embedding_batches, axis=0)

    col_names = [f"emb_{i}" for i in range(matrix.shape[1])]
    df_embeddings = pd.DataFrame(matrix, columns=col_names)
    df_embeddings.insert(0, "NSC", valid_nsc_list)
    return df_embeddings
    


print("\nLoading externial GraphMVP imports.")
mol_to_graph_data_obj_simple, GNNComplete = acessing_graphmvp_files()


print("\nLoading input data.")
df_DOSERESP, df_NCI60_SMILES = load_input_data()
print(
    f"\ndf_DOSERESP: {df_DOSERESP.shape}"
    f"\ndf_NCI60_SMILES: {df_NCI60_SMILES.shape}"
)


print("\nRetreiving unique NSCs.")
df_unique_NSCs, unique_NSC_count = get_unique_NSCs(df_DOSERESP)
print(f"Retrieved {unique_NSC_count} NSCs.")


print("\nRetrieving and merging SMILES strings with unique NSCs.")
df_NSC_with_SMILES, df_dropped_NSC_Numbers = merge_SMILES_strings(df_unique_NSCs, df_NCI60_SMILES)
print(
    f"Dropped {len(df_dropped_NSC_Numbers)} NSC due to invalid data."
    f"\nNCI-60 SMILES string dictionary: {df_NSC_with_SMILES.shape}"
    f"\n\n{df_NSC_with_SMILES.head(10)}"
)


print(
    f"\nInitiating GraphMVP scripts"
    f"\nConverting SMILES strings into pyg graphs."
)
pyg_graph_list, valid_nsc_list = graphmvp_featurizer(df_NSC_with_SMILES)
print(f"\nGenerated {len(valid_nsc_list)} molecular representations.")


print("\nLoading GraphMVP GNN.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nComputing on {device}.")
graphmvp_model = load_graphmvp_GNN(device, num_layer = 5, emb_dim = 300)


print("Starting Embedding generation.")

df_graphmvp_embeddings = extract_embeddings(
    model=graphmvp_model,
    pyg_graph_list=pyg_graph_list,
    valid_nsc_list=valid_nsc_list,
    device=device,
    batch_size=64
)

print("\nSaving molecular embeddings to CSV.")
output_dir = r"C:\Development\NCI-60_Work\Main_Code\GraphMVP\GraphMVP_Embedding_Output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "graphmvp_nsc_embeddings.csv")

df_graphmvp_embeddings.to_csv(output_path, index=False)
print(f"Final output shape: {df_graphmvp_embeddings.shape}")


    


