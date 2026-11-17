import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
import os
from rdkit.Chem import BRICS
from itertools import combinations
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn


def import_input_data():
    df_DOSERESP = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\DOSERESP.csv")
    df_NCI60_SMILES = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\nsc_smiles.csv")
    df_doubling = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\NCI-60_Cell_Line_Doubling_Times.csv")
    return df_DOSERESP, df_NCI60_SMILES, df_doubling



def remove_non_M_units(df_DOSERESP):
    unit = "M"
    df_DOSERESP = df_DOSERESP[df_DOSERESP["CONCENTRATION_UNIT"] == unit].copy()
    return df_DOSERESP



def convert_to_viability(df_DOSERESP, df_doubling):
    df_doubling = df_doubling.set_index("CELL_NAME")
    doubling_time = df_doubling["Doubling Time (hours)"]
    td = df_DOSERESP["CELL_NAME"].map(doubling_time)
    pg = df_DOSERESP["AVERAGE_GIPRCNT"]
    
    G = 2**(48/td)

    viability_values = []
    
    for p, growth_factor in zip(pg, G):
        if pd.isna(p) or pd.isna(growth_factor):
            viability_values.append(np.nan)
        
        elif p >= 0:
            val = (((p / 100.0) * (growth_factor - 1.0)) + 1.0) / growth_factor
            viability_values.append(val)
        
        else:
            val = ((p / 100.0) + 1.0) / growth_factor
            viability_values.append(val)
    
    df_DOSERESP["VIABILITY"] = viability_values

    return df_DOSERESP

 

def aggregate_experimental_duplicates (df_DOSERESP):
    group = ["NSC", "CONCENTRATION", "CELL_NAME"]
    duplicates = df_DOSERESP.duplicated(subset = group, keep = False)
    
    df_duplicates = df_DOSERESP[duplicates]
    df_uniques = df_DOSERESP[~duplicates]

    df_aggregated = df_duplicates.groupby(group, as_index = False)["VIABILITY"].median()
    df_uniques = df_uniques[group + ["VIABILITY"]]
    
    df_DOSERESP = (pd.concat([df_uniques, df_aggregated], ignore_index = True))

    return df_DOSERESP, df_duplicates, df_uniques



def available_cell_lines(df_DOSERESP):
    cell_line_options = (df_DOSERESP["CELL_NAME"].value_counts().reset_index())
    cell_line_options.columns = ["CELL_NAME", "COUNT"]
    return cell_line_options



def select_cell_line(df_DOSERESP, cell_line_input):
    df_DOSERESP = df_DOSERESP[df_DOSERESP["CELL_NAME"] == cell_line_input].reset_index(drop = True)
    return df_DOSERESP



def analyze_conc_distribution(df_DOSERESP):
    conc_distribution = df_DOSERESP["CONCENTRATION"].value_counts()
    return conc_distribution



def drop_invalid_conc (df_DOSERESP, selected_conc_list):
    required_concs = list(selected_conc_list)
    expected_count = len(required_concs)

    df_filtered = df_DOSERESP[df_DOSERESP["CONCENTRATION"].isin(required_concs)].copy()
    points_per_curve = df_filtered.groupby(["NSC", "CELL_NAME"])["CONCENTRATION"].transform("nunique")
    df_DOSERESP = df_filtered[points_per_curve == expected_count].copy()
    return df_DOSERESP



def get_unique_NSCs (df_DOSERESP):
    df_NSCs = df_DOSERESP[["NSC"]]
    df_unique_NSCs = df_NSCs.drop_duplicates().reset_index(drop = True)

    unique_NSC_count = len(df_unique_NSCs)
    
    return df_unique_NSCs, unique_NSC_count



#--------------------------------------------------------------------------------------------------------------------------
def r_group_format (df_NCI60_SMILES):
    df_NCI60_SMILES = df_NCI60_SMILES.copy()

    has_r_group = df_NCI60_SMILES["SMILES"].astype(str).str.contains(r"\[R\]", regex = True)
    replacement_count = int(has_r_group.sum())
    
    df_NCI60_SMILES["SMILES"] = df_NCI60_SMILES["SMILES"].str.replace("[R]", "[*]", regex = False)
    df_NCI60_SMILES["SMILES"] = df_NCI60_SMILES["SMILES"].str.replace("[RH]", "[*]", regex = False)
    df_NCI60_SMILES["SMILES"] = df_NCI60_SMILES["SMILES"].str.replace("[R1]", "[*]", regex = False)
    df_NCI60_SMILES["SMILES"] = df_NCI60_SMILES["SMILES"].str.replace("[Z]", "[*]", regex = False)
    df_NCI60_SMILES["SMILES"] = df_NCI60_SMILES["SMILES"].str.replace("[ZH]", "[*]", regex = False)

    return df_NCI60_SMILES, replacement_count



def invalid_SMILES_filter (df_NCI60_SMILES):
    invalid_smiles = []

    for index, smiles in df_NCI60_SMILES["SMILES"].items():
        if not isinstance(smiles, str) or Chem.MolFromSmiles(smiles) is None:
            invalid_smiles.append(index)
    
    df_NCI60_SMILES.drop(index = invalid_smiles, inplace = True)
    df_NCI60_SMILES.reset_index(drop = True, inplace = True)
    return df_NCI60_SMILES, invalid_smiles



def remove_salts (df_NCI60_SMILES):
    remover = SaltRemover()
    
    for indx, row in df_NCI60_SMILES.iterrows():
        smiles = row["SMILES"]
        
        if isinstance (smiles, str) and smiles.strip():
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                stripped = remover.StripMol(mol, dontRemoveEverything = True)
                df_NCI60_SMILES.at[indx, "SMILES"] = Chem.MolToSmiles(stripped)

    return df_NCI60_SMILES



def merge_SMILES_to_DOSERESP (df_DOSERESP, df_NCI60_SMILES):
    df_DOSERESP = pd.merge(df_DOSERESP, df_NCI60_SMILES[["NSC", "SMILES"]], on = "NSC", how = "left")
    df_dropped_NSCs = df_DOSERESP.loc[df_DOSERESP["SMILES"].isna(), "NSC"].unique().tolist()
    df_DOSERESP = df_DOSERESP.dropna(subset=["SMILES"]).reset_index(drop=True)
    return df_DOSERESP, df_dropped_NSCs



#--------------------------------------------------------------------------------------------------------------------------
def visual_check_render(df_DOSERESP):
    df_DOSERESP["SMILES"] = df_DOSERESP["SMILES"].apply(lambda x: x[0] if isinstance(x, tuple) else str(x))

    smiles_to_render = df_DOSERESP["SMILES"].unique().tolist()
    nsc_to_render = df_DOSERESP["NSC"].unique().tolist()

    output_dir = (r"C:\Development\NCI-60_Work\Main_Code\Molecular_Structures\analyzing_substructures_output_data\rendered 2D png")
    os.makedirs(output_dir, exist_ok = True)

    for smile, nsc in zip(smiles_to_render, nsc_to_render):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        
        Chem.rdDepictor.Compute2DCoords(mol)

        img = Draw.MolToImage(mol, size = (1000, 1000))

        file_name = os.path.join(output_dir, f"NSC_{nsc}.png")
        img.save(file_name)
    return



#--------------------------------------------------------------------------------------------------------------------------
def molecule_analysis (df_DOSERESP):
    smiles_list = df_DOSERESP["SMILES"].unique().tolist()
    atom_graph_data = {}
    
    for smiles in smiles_list:
        if not isinstance(smiles, str) or not smiles.strip():
            atom_graph_data[smiles] = None
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            atom_graph_data[smiles] = None
            continue

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(int(atom.GetHybridization())),
                float(atom.GetFormalCharge()),
                1.0 if atom.GetIsAromatic() else 0.0,
                float(atom.GetTotalNumHs())
                ])
        
        atom_edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            atom_edges.extend([[i, j], [j, i]])
        
        atom_graph_data[smiles] = {
            "num_atoms": mol.GetNumAtoms(),
            "atom_features": atom_features,
            "atom_edges": atom_edges
            }
    
    df_DOSERESP["ATOM_GRAPH_DATA"] = df_DOSERESP["SMILES"].map(atom_graph_data)
    return df_DOSERESP



def BRICS_motif (df_DOSERESP):
    smiles_list = df_DOSERESP["SMILES"].unique().tolist()
    motif_graph_data = {}

    for smiles in smiles_list:
        if not isinstance(smiles, str) or not smiles.strip():
            motif_graph_data[smiles] = None
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            motif_graph_data[smiles] = None
            continue

        num_atoms = mol.GetNumAtoms()

        broken_mol = BRICS.BreakBRICSBonds(mol)
        frags_atom_indices = Chem.GetMolFrags(broken_mol, asMols=False)
        num_frags = len(frags_atom_indices)

        atom2frag = [0] * num_atoms
        for frag_idx, atom_indices in enumerate(frags_atom_indices):
            for a_idx in atom_indices:
                if a_idx < num_atoms:
                    atom2frag[a_idx] = frag_idx
        
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        frag_edges = []
        for (a1, a2), _ in brics_bonds:
            f1 = atom2frag[a1]
            f2 = atom2frag[a2]
            if f1 != f2:
                frag_edges.extend([[f1, f2], [f2, f1]])
        
        if not frag_edges:
            frag_edges = [[0, 0]]

        motif_graph_data[smiles] = {
        "num_frags": num_frags,
        "atom2frag": atom2frag,
        "frag_edges": frag_edges
        }

    df_DOSERESP["MOTIF_GRAPH_DATA"] = df_DOSERESP["SMILES"].map(motif_graph_data)
    return df_DOSERESP


def molecular_graph (df_DOSERESP):
    hierarchical_graphs = []

    for data, row in df_DOSERESP.iterrows():
        atom_graph = row.get("ATOM_GRAPH_DATA")
        motif_graph = row.get("MOTIF_GRAPH_DATA")
        viability = row.get("VIABILITY")

        if (atom_graph is None or not isinstance(atom_graph, dict)or motif_graph is None or not isinstance(motif_graph, dict)or pd.isna(viability)):
            continue

        x = torch.tensor(atom_graph["atom_features"], dtype=torch.float)
        if atom_graph["atom_edges"]:
            edge_index = (torch.tensor(atom_graph["atom_edges"], dtype=torch.long).t().contiguous())
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        atom2frag = torch.tensor(motif_graph["atom2frag"], dtype=torch.long)
        num_frags = torch.tensor([motif_graph["num_frags"]], dtype=torch.long)
        if motif_graph["frag_edges"]:
            frag_edge_index = (torch.tensor(motif_graph["frag_edges"], dtype=torch.long).t().contiguous())
        else:
            frag_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        y = torch.tensor([float(viability)], dtype=torch.float)

        data = Data(
            x = x,
            edge_index = edge_index,
            atom2frag = atom2frag,
            frag_edge_index = frag_edge_index,
            num_frags = num_frags,
            y = y
            ) 
        
        hierarchical_graphs.append(data)

    print(f"\nGenerated {len(hierarchical_graphs)} hierarchical graphs.")
    return hierarchical_graphs



#--------------------------------------------------------------------------------------------------------------------------  
class HierarchicalData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ["frag_edge_index", "atom2frag"]:
            return self.num_frags.item()
        return super().__inc__(key, value, *args, **kwargs)

class HierarchicalGNN(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64, out_channels=1, dropout=0.1):
        super().__init__()

        self.atom_conv1 = GCNConv(in_channels, hidden_dim)
        self.atom_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.motif_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.motif_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, out_channels)

    def extract_embedding(self, batch_data):
        h_atom = self.relu(self.atom_conv1(batch_data.x, batch_data.edge_index))
        h_atom = self.dropout(h_atom)
        h_atom = self.relu(self.atom_conv2(h_atom, batch_data.edge_index))

        h_motif = scatter(h_atom, batch_data.atom2frag, dim=0, reduce="mean")

        h_motif = self.relu(self.motif_conv1(h_motif, batch_data.frag_edge_index))
        h_motif = self.dropout(h_motif)
        h_motif = self.relu(self.motif_conv2(h_motif, batch_data.frag_edge_index))

        num_frags = batch_data.num_frags
        if num_frags.dim() > 1:
            num_frags = num_frags.squeeze(-1)

        motif_batch = torch.repeat_interleave(
            torch.arange(batch_data.num_graphs, device=h_motif.device, dtype=torch.long),num_frags)
        return global_mean_pool(h_motif, motif_batch)

    def forward(self, batch_data):
        h_graph = self.extract_embedding(batch_data)
        out = self.relu(self.fc1(h_graph))
        return self.fc2(out)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)
        target = batch.y.squeeze(-1)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).squeeze(-1)
        target = batch.y.squeeze(-1)
        loss = criterion(pred, target)
        total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)



#--------------------------------------------------------------------------------------------------------------------------
data_choice = input("\nDo you have [substructure_data_frame_output.csv] saved? (y/n): ").strip().lower()
if (data_choice == "y"):
    print("Importing substructure_data_frame_output.csv from previous session.")
    df_DOSERESP = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Molecular_Structures\analyzing_substructures_output_data\substructure_data_frame_output.csv")
    print(f"\ndf_DOSERESP: {df_DOSERESP.shape}") 
    print(df_DOSERESP.head(10))

else: 
    print("Proceeding to create new file.")

    print("\nImporting input data.")
    df_DOSERESP, df_NCI60_SMILES, df_doubling = import_input_data()
    print(
        f"df_DOSERESP:        {df_DOSERESP.shape}\n"
        f"df_NCI60_SMILES:    {df_NCI60_SMILES.shape}\n"
        f"df_doubling:        {df_doubling.shape}"
    ) 

    print(f"\nDropping data points with non M (log molar) concenctration units.")
    df_DOSERESP = remove_non_M_units(df_DOSERESP)
    print(f"Data size after unit filteration: {df_DOSERESP.shape}")


    print("\nStanderdizing experimental outputs to viability values.")
    df_DOSERESP = convert_to_viability(df_DOSERESP, df_doubling)
    print(f"Data size after VIABILITY unit conversion: {df_DOSERESP.shape}")


    print(
        "\nAggregating experimental duplicates."
        "\nAggregation method = med"
        )
    df_DOSERESP, df_duplicates, df_uniques = aggregate_experimental_duplicates(df_DOSERESP)
    print(
        f"Found {len(df_duplicates)} total experimental duplicates."
        f"\nFound {len(df_uniques)} unique data points."
        f"\nData size after aggregation: {df_DOSERESP.shape}"
        )


    cell_line_isolation = input("\nExtract screening profiles for an individual cell line? (y/n): ")
    is_single_cell = (cell_line_isolation.strip().lower() == "y")
    if (cell_line_isolation == "y"):
        cell_line_options = available_cell_lines(df_DOSERESP)
        with pd.option_context("display.max_rows", None):
            print(cell_line_options.to_string(index = False))
        
        cell_line_input = input("\nEnter target cell line: ")
        print(f"Isolating {cell_line_input} cell line response data.")    
        
        df_DOSERESP = select_cell_line(df_DOSERESP, cell_line_input)
        print(f"Data size after isolating {cell_line_input} cell line: {df_DOSERESP.shape}")
        
    else:
        print("Continuing foward.")


    print("\nStanderdizing dosage for drug response curve.")
    df_conc_distribution = analyze_conc_distribution(df_DOSERESP)
    print(df_conc_distribution)
    print("\nNCI-60 standard 5 concentration values (log molar): -4.0000, -5.0000, -6.0000, -7.000, -8.0000")
    
    conc_use_question = input(
        "[1]: Standard 5 Concentration Values, [2]: Custom Concentration Value Set"
        "\nEnter [1] or [2]: "
    ).strip()

    if (conc_use_question == "1"):
        selected_conc_list = sorted([-4.0000, -5.0000, -6.0000, -7.000, -8.0000])
        print(f"\nExtracting the follwoing concentration(s): {selected_conc_list}")
        df_DOSERESP = drop_invalid_conc(df_DOSERESP, selected_conc_list)

    elif (conc_use_question == "2"):
        selected_conc_list = [float(x.strip()) for x in input("Enter standardized concentrations (separated by comma): ").strip().split(",")]
        selected_conc_list = sorted(selected_conc_list)
        print(f"\nExtracting the follwoing concentration(s): {selected_conc_list}")
        df_DOSERESP = drop_invalid_conc(df_DOSERESP, selected_conc_list)
    
    
    print("\nRetrieving unique NSCs in subset.")
    df_unique_NSCs, unique_NSC_count = get_unique_NSCs(df_DOSERESP)
    print(f"Retrieved {unique_NSC_count} NSCs.")


    print("\nChanging r-group format from [R] to [*].")
    df_NCI60_SMILES, replacement_count = r_group_format(df_NCI60_SMILES)
    print(f"Total SMILES strings modified: {replacement_count}")


    print("\nFiltering invalid SMILES strings.")
    df_NCI60_SMILES, invalid_smiles =  invalid_SMILES_filter(df_NCI60_SMILES)
    print(f"Found and dropped {len(invalid_smiles)} SMILES strings.")

    
    remove_salts_choice = str(input("\nRemove salts? (y/n): "))
    if (remove_salts_choice == "y"):
        print("Removing salts")
        df_NCI60_SMILES = remove_salts(df_NCI60_SMILES)
    else:
        print("Proceeding to next")
    
    
    print("\nRetrieving and merging SMILES strings with unique NSCs.")
    df_DOSERESP, df_dropped_NSCs = merge_SMILES_to_DOSERESP(df_DOSERESP, df_NCI60_SMILES)
    print(f"Dropped {len(df_dropped_NSCs)} NSC due to invalid data.")
    

    preview_choice = input("\nPreview df_DOSERESP? (y/n): ").strip().lower()
    if (preview_choice == "y"):
        print(f"df_DOSERESP: {df_DOSERESP.shape}") 
        print(df_DOSERESP.head(10))
    else: 
        print("Continuing to exports")
    

    export_choice = (input("\nExport df_DOSERESP CSV files? (y/n): ").strip().lower())
    if export_choice == "y":
        df_DOSERESP.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Molecular_Structures\analyzing_substructures_output_data\substructure_data_frame_output.csv", index=False)
        print("Exported file")

    else:
        print("Export sequence cancelled")


manual_visual_check = str(input("\nVisual Check: Render a molecule? (y/n): "))
if (manual_visual_check == "y"):
    
    confirmation = str(input("This takes a very long time to complete rendering. Are you sure you would like to render rn? (y/n): "))
    if (confirmation == "y"):
        visual_check_render(df_DOSERESP)
    else:
        print("Proceeding to next")

else:
    print("Proceeding to next")


execute_gnn_choice = str(input("\nProceed to GNN? (y/n): ")).strip().lower()
if execute_gnn_choice == "y":
    processed_dir = r"C:\Development\NCI-60_Work\Main_Code\Molecular_Structures\analyzing_substructures_output_data"
    save_path = os.path.join(processed_dir, "hierarchical_graphs.pt")
    os.makedirs(processed_dir, exist_ok=True)

    data_choice = input("\nDo you have [hierarchical_graphs.pt] saved? (y/n): ").strip().lower()
    if data_choice == "y":
        print(f"\nLoading cached hierarchical graphs from: {save_path}")
        raw_graphs = torch.load(save_path, weights_only=False)
        print(f"Loaded {len(raw_graphs)} hierarchical graphs.")
    else:
        print("Proceeding to create new graph objects.")
        print("\nRunning molecular data analysis.")
        df_DOSERESP = BRICS_motif(df_DOSERESP)
        df_DOSERESP = molecule_analysis(df_DOSERESP)
        raw_graphs = molecular_graph(df_DOSERESP)

        print(f"\nSaving {len(raw_graphs)} graphs to disk.")
        torch.save(raw_graphs, save_path)
        print(f"Saved successfully to: {save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nFormatting data structures into HierarchicalData.")
    hierarchical_graphs = [
        HierarchicalData(
            x=g.x,
            edge_index=g.edge_index,
            atom2frag=g.atom2frag,
            frag_edge_index=g.frag_edge_index,
            num_frags=g.num_frags,
            y=g.y,
        )
        for g in raw_graphs
    ]

    total = len(hierarchical_graphs)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(hierarchical_graphs, [train_size, val_size, test_size], generator=generator)

    batch_size = 256
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Data split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    model = HierarchicalGNN(in_channels=6, hidden_dim=64, out_channels=1, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    epochs = 30
    best_val_loss = float("inf")
    best_model_path = os.path.join(processed_dir, "best_hierarchical_gnn.pt")

    print("\nBeginning training loop.")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\nLoading best checkpoint: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test MSE:  {test_loss:.5f}")
    print(f"Final Test RMSE: {test_loss**0.5:.5f}")

else:
    print("Proceeding to next")



 