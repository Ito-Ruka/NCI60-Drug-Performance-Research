import os
from chemprop import data, featurizers, models, nn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm



df_DOSERESP = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\DOSERESP.csv")
df_NCI60_SMILES = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\nsc_smiles.csv")
print("df_DOSERESP:", df_DOSERESP.shape)
print("df_NCI60_SMILES:", df_NCI60_SMILES.shape)



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



def biuld_molecular_datapoints(df_clean):
    datapoints = []
    valid_nscs = []
    failed_nscs = []

    for index, row in tqdm(df_clean.iterrows()):
        smiles = row["SMILES"]
        nsc = row["NSC"]

        try:
            dp = data.MoleculeDatapoint.from_smi(smiles)
            datapoints.append(dp)
            valid_nscs.append(nsc)
            
        except Exception:
            failed_nscs.append(nsc)
    
    return datapoints, valid_nscs



def chemprop_model(d_h = 400, depth = 3):
    message_passing = nn.BondMessagePassing(d_h = d_h, d_vd = d_h, depth = depth)
    aggregation = nn.MeanAggregation()
    predictor = nn.RegressionFFN(input_dim = message_passing.output_dim, n_tasks = 1)

    model = models.MPNN(
        message_passing = message_passing,
        agg = aggregation,
        predictor = predictor
        )
    
    return model



def get_embeddings(model, dataloader, device):
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Generating embeddings"):
            bmg = batch.bmg[0] if isinstance(batch.bmg, (list, tuple)) else batch.bmg
            res = bmg.to(device)
            
            if res is not None:
                bmg = res

            batch_emb = model.fingerprint(bmg)
            all_embeddings.append(batch_emb.cpu())

    final_embeddings = torch.cat(all_embeddings, dim = 0)
    return final_embeddings



def save_embeddings(embeddings, valid_nscs, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    pt_path = os.path.join(output_dir, "chemprop_dmpnn_embeddings.pt")
    torch.save({"nsc": valid_nscs, "embeddings": embeddings}, pt_path)
    print(f"Saved PyTorch tensor to:\n{pt_path}")

    feature_cols = [f"dmpnn_{i}" for i in range(embeddings.shape[1])]
    df_embeddings = pd.DataFrame(embeddings.numpy(), columns = feature_cols)
    df_embeddings.insert(0, "NSC", valid_nscs)

    csv_path = os.path.join(output_dir, "chemprop_dmpnn_embeddings.csv")
    df_embeddings.to_csv(csv_path, index = False)
    print(f"Saved embeddings CSV to:\n{csv_path}")

    return df_embeddings



print("\nRetrieving unique NSCs.")
df_unique_NSCs, unique_NSC_count = get_unique_NSCs(df_DOSERESP)
print(f"Retrieved {unique_NSC_count} NSCs.")


print("\nRetrieving and merging SMILES strings with unique NSCs.")
df_NSC_with_SMILES, df_dropped_NSC_Numbers = merge_SMILES_strings(
    df_unique_NSCs, df_NCI60_SMILES
)
print(
    f"Dropped {len(df_dropped_NSC_Numbers)} NSC due to invalid data."
    f"SMILES string dictionary: {df_NSC_with_SMILES.shape}"
)

print(df_NSC_with_SMILES.head(10))


proceed_choice = (input("\nProceed to Chemprop embedding generation? (y/n): ").strip().lower())
if (proceed_choice) == "y":
    print("Proceeding with embedding generation.")


    print("\nBuilding Chemprop molecule datapoints.")
    datapoints, valid_nscs = biuld_molecular_datapoints(df_NSC_with_SMILES)
    print(f"Constructed {len(datapoints)} datapoints.")


    print("\nFeaturizing graphs and assembling DataLoader.")
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer = featurizer)
    mol_loader = data.build_dataloader(dataset, batch_size = 64, shuffle = False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = chemprop_model(d_h = 400, depth = 3)


    final_embeddings = get_embeddings(model, mol_loader, device)
    print(f"Extracted embeddings matrix shape: {final_embeddings.shape}")
    

    output_dir = r"C:\Development\NCI-60_Work\Main_Code\Chemprop\Chemprop_embedding_output"
    df_embeddings = save_embeddings(final_embeddings, valid_nscs, output_dir)
    
    print(df_embeddings.head(5))


elif (proceed_choice) == "n":
    print("Skipped chemprop embedding generation.")