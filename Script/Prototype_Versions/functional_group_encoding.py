import numpy as np 
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import torch
from torch_geometric.data import Data
from rdkit.Chem import Descriptors
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



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



def invalid_SMILES_filter (df_NCI60_SMILES):
    invalid_smiles = []

    for index, smiles in df_NCI60_SMILES["SMILES"].items():
        if not isinstance(smiles, str) or Chem.MolFromSmiles(smiles) is None:
            invalid_smiles.append(index)
    
    df_NCI60_SMILES.drop(index = invalid_smiles, inplace = True)
    df_NCI60_SMILES.reset_index(drop = True, inplace = True)
    return df_NCI60_SMILES, invalid_smiles



def remove_salt(df_NCI60_SMILES):
    remover = SaltRemover()
    for idx, row in df_NCI60_SMILES.iterrows():
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            stripped = remover.StripMol(mol, dontRemoveEverything = True)
            df_NCI60_SMILES.at[idx, "SMILES"] = Chem.MolToSmiles(stripped)
    
    return df_NCI60_SMILES



def get_mol_weight (df_NCI60_SMILES):
    mol_weight = []
    smiles_list = df_NCI60_SMILES["SMILES"].tolist()

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_weight.append(None)
        else:
            mol_weight.append(Descriptors.MolWt(mol))
            print(f"Completed calculating molecular weight for: {smiles}.")

    df_NCI60_Mol_Weight = pd.DataFrame()
    df_NCI60_Mol_Weight["SMILES"] = smiles_list
    df_NCI60_Mol_Weight["MOLECULAR_WEIGHT"] = mol_weight
    
    return df_NCI60_Mol_Weight



def get_mol_weight_stats (df_NCI60_SMILES, df_NCI60_Mol_Weight):
    mol_weight = df_NCI60_Mol_Weight["MOLECULAR_WEIGHT"].astype(float).tolist()
    mol_weight_arr = np.array(mol_weight)

    mol_weight_stats = {
        "mean": np.mean(mol_weight_arr),
        "min": np.min(mol_weight_arr),
        "median": np.median(mol_weight_arr),
        "max": np.max(mol_weight_arr),
        "standard deviation": np.std(mol_weight_arr),
        "25th %": np.percentile(mol_weight_arr, 25),
        "50th %": np.percentile(mol_weight_arr, 50),
        "75th %": np.percentile(mol_weight_arr, 75),
        "IQR": np.percentile(mol_weight_arr, 75) - np.percentile(mol_weight_arr, 25)
        }
    
    df_mol_weight_stats = pd.DataFrame.from_dict(mol_weight_stats, orient = 'index', columns = ['value'])
    return df_mol_weight_stats 



def get_TPSA (df_NCI60_SMILES):
    smiles_list = df_NCI60_SMILES["SMILES"].astype(str).tolist()
    tpsa_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            tpsa_list.append(None)
        else:
            tpsa_list.append(Descriptors.TPSA(mol))
            print(f"Completed calculating TPSA for: {smiles}")
    
    df_NCI60_TPSA = pd.DataFrame()
    df_NCI60_TPSA["SMILES"] = smiles_list
    df_NCI60_TPSA["TPSA"] = tpsa_list        

    return df_NCI60_TPSA



def get_MolLogP (df_NCI60_SMILES):
    smiles_list = df_NCI60_SMILES["SMILES"].astype(str).tolist()
    mollogp_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mollogp_list.append(None)
        else:
            mollogp_list.append(Descriptors.MolLogP(mol))
            print(f"Completed calculating MolLogP for: {smiles}")
    
    df_NCI60_MolLogP = pd.DataFrame()
    df_NCI60_MolLogP["SMILES"] = smiles_list
    df_NCI60_MolLogP["MOLLOGP"] = mollogp_list
    
    return df_NCI60_MolLogP



def mol_weight_plot (df_DOSERESP, df_NCI60_Mol_Weight):
    df = pd.merge(df_DOSERESP, df_NCI60_Mol_Weight, on = "SMILES")

    plt.figure(figsize = (12, 8))
    plt.scatter(df["MOLECULAR_WEIGHT"], df["VIABILITY"], s = 1)
    plt.title("Molecular Weight Vs Viability")
    plt.xlabel("g/mol")
    plt.ylabel("Viability")
    plt.show(block = "True")
    return



def TPSA_plot (df_DOSERESP, df_NCI60_TPSA):
    df = pd.merge(df_DOSERESP, df_NCI60_TPSA, on = "SMILES")

    plt.figure(figsize = (12, 8))
    plt.scatter(df["TPSA"], df["VIABILITY"], s = 1)
    plt.title("TPSA Vs Viability")
    plt.xlabel("TPSA")
    plt.ylabel("Viability")
    plt.show(block = "True")
    return



def MolLogP_plot (df_DOSERESP, df_NCI60_MolLogP):
    df = pd.merge(df_DOSERESP, df_NCI60_MolLogP, on = "SMILES")

    plt.figure(figsize = (12, 8))
    plt.scatter(df["MOLLOGP"], df["VIABILITY"], s = 1)
    plt.title("MOLLOGP Vs Viability")
    plt.xlabel("MOLLOGP")
    plt.ylabel("Viability")
    plt.show(block = "True")
    return



def get_functional_grp(df_NCI60_SMILES):
    smiles_list = df_NCI60_SMILES["SMILES"].astype(str).tolist()

    def find_heteroatoms(mol):
        fg_atom_indices = set()

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()

            if symbol not in ["C", "H"]:
                fg_atom_indices.add(atom.GetIdx())

            elif symbol == "C":
                for bond in atom.GetBonds():
                    if bond.GetBondTypeAsDouble() > 1.0:
                        nbr = bond.GetOtherAtom(atom)
                        if nbr.GetSymbol() not in ["C", "H"]:
                            fg_atom_indices.add(atom.GetIdx())
                            break

        return fg_atom_indices

    def cluster_and_extract(mol, fg_atom_indices):
        visited = set()
        functional_groups = []

        for idx in fg_atom_indices:
            if idx in visited:
                continue

            cluster = set()
            queue = [idx]
            visited.add(idx)

            while queue:
                curr = queue.pop(0)
                cluster.add(curr)
                for nbr in mol.GetAtomWithIdx(curr).GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx in fg_atom_indices and nbr_idx not in visited:
                        visited.add(nbr_idx)
                        queue.append(nbr_idx)

            rw_mol = Chem.RWMol()
            sub_map = {}

            for c_idx in cluster:
                atom = mol.GetAtomWithIdx(c_idx)
                new_idx = rw_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                rw_mol.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
                sub_map[c_idx] = new_idx

            for c_idx in cluster:
                atom = mol.GetAtomWithIdx(c_idx)
                for bond in atom.GetBonds():
                    nbr_idx = bond.GetOtherAtomIdx(c_idx)
                    if nbr_idx in cluster and c_idx < nbr_idx:
                        rw_mol.AddBond(
                            sub_map[c_idx],
                            sub_map[nbr_idx],
                            bond.GetBondType(),
                        )

            for c_idx in cluster:
                atom = mol.GetAtomWithIdx(c_idx)
                for nbr in atom.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx not in cluster:
                        cap_idx = rw_mol.AddAtom(Chem.Atom(0))
                        bond_type = mol.GetBondBetweenAtoms(c_idx, nbr_idx).GetBondType()
                        rw_mol.AddBond(sub_map[c_idx], cap_idx, bond_type)

            frag_mol = rw_mol.GetMol()
            Chem.SanitizeMol(frag_mol, catchErrors=True)
            functional_groups.append(Chem.MolToSmiles(frag_mol))

        return functional_groups

    mol_fg_results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_fg_results.append([])
        else:
            fg_atom_indices = find_heteroatoms(mol)
            fg_list = cluster_and_extract(mol, fg_atom_indices)
            mol_fg_results.append(fg_list)

    df_result = pd.DataFrame({"SMILES": smiles_list, "FUNCTIONAL_GROUPS": mol_fg_results,})

    if "NSC" in df_NCI60_SMILES.columns:
        df_result.insert(0, "NSC", df_NCI60_SMILES["NSC"].values)

    return df_result


def merge_SMILES_to_DOSERESP (df_DOSERESP, df_NCI60_SMILES, df_NCI60_Mol_Weight, df_NCI60_TPSA, df_NCI60_MolLogP, df_NCI60_FG):
    df_features = df_NCI60_SMILES[["NSC", "SMILES"]].copy()
    df_features = pd.merge(df_features, df_NCI60_Mol_Weight[["SMILES", "MOLECULAR_WEIGHT"]], on = "SMILES", how = "left")
    df_features = pd.merge(df_features, df_NCI60_TPSA[["SMILES", "TPSA"]], on = "SMILES", how = "left")
    df_features = pd.merge(df_features, df_NCI60_MolLogP[["SMILES", "MOLLOGP"]], on = "SMILES", how = "left")
    df_features = pd.merge(df_features, df_NCI60_FG[["SMILES", "FUNCTIONAL_GROUPS"]], on = "SMILES", how = "left")

    df_DOSERESP = pd.merge(df_DOSERESP, df_features, on = "NSC", how = "left")
    
    df_dropped_NSCs = df_DOSERESP.loc[df_DOSERESP["SMILES"].isna(), "NSC"].unique().tolist()
    df_DOSERESP = df_DOSERESP.dropna(subset = ["SMILES"]).reset_index(drop = True)

    sort_cols = [col for col in ["NSC", "CELL_NAME", "CONCENTRATION"] if col in df_DOSERESP.columns]
    df_DOSERESP = df_DOSERESP.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    return df_DOSERESP, df_dropped_NSCs



def execution_script ():
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
    

    print("\nFiltering invalid SMILES strings.")
    df_NCI60_SMILES, invalid_smiles =  invalid_SMILES_filter(df_NCI60_SMILES)
    print(f"Found and dropped {len(invalid_smiles)} SMILES strings.")


    preview_choice = input("\nRemove salts? (y/n): ").strip().lower()
    if (preview_choice == "y"):
        print("Removing salts from SMILES strings.")
        df_NCI60_SMILES = remove_salt(df_NCI60_SMILES)
    else: 
        print("Continuing foward.")


    print("\nCalculating molecular weights.")
    df_NCI60_Mol_Weight = get_mol_weight(df_NCI60_SMILES)
    print(
        f"\ndf_NCI60_Mol_Weight: {df_NCI60_Mol_Weight.shape}"
        f"\n{df_NCI60_Mol_Weight.head(10)}"
    )
    
    print("\nNCI60 Molecular Weight Stats:")
    df_mol_weight_stats = get_mol_weight_stats(df_NCI60_SMILES, df_NCI60_Mol_Weight)
    print(df_mol_weight_stats)


    print("Calculating Topological Polar Surface Area")
    df_NCI60_TPSA = get_TPSA(df_NCI60_SMILES)
    print(f"df_NCI60_TPSA: {df_NCI60_TPSA.shape}")


    print("Calculating Mol Log P")
    df_NCI60_MolLogP = get_MolLogP(df_NCI60_SMILES)
    print(f"df_NCI60_MolLogP: {df_NCI60_MolLogP.shape}")


    print("Searching for functional groups.")
    df_NCI60_FG = get_functional_grp(df_NCI60_SMILES)
    print(df_NCI60_FG.head(10))


    print("\nRetrieving and merging SMILES strings with unique NSCs.")
    df_DOSERESP, df_dropped_NSCs = merge_SMILES_to_DOSERESP(df_DOSERESP, df_NCI60_SMILES, df_NCI60_Mol_Weight, df_NCI60_TPSA, df_NCI60_MolLogP, df_NCI60_FG)
    print(f"Dropped {len(df_dropped_NSCs)} NSC due to invalid data.")


    preview_choice = input("\nPreview df_DOSERESP? (y/n): ").strip().lower()
    if (preview_choice == "y"):
        print(f"df_DOSERESP: {df_DOSERESP.shape}") 
        print(df_DOSERESP.head(10))
    else: 
        print("Continuing foward.")
    

    plot_choice = input("\nView viability relationships? (y/n): ").strip().lower()
    if (plot_choice == "y"):
        mol_weight_plot(df_DOSERESP, df_NCI60_Mol_Weight)
        TPSA_plot(df_DOSERESP, df_NCI60_TPSA)
        MolLogP_plot(df_DOSERESP, df_NCI60_MolLogP)
    else: 
        print("Continuing foward.")
    

    df_DOSERESP.to_csv(r"C:\Development\NCI-60_Work\Input_Data\df_DOSERESP_merged.csv", index=False)
    print(f"Saved merged DOSERESP data.")

    return

execution_script()



