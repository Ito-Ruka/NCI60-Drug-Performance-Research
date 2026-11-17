import os
import numpy as np 
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

df_GI50 = pd.read_csv(r"C:\Users\lexha\Downloads\Visual_Studio_Code\NCI-60_Work\Input_Data\GI50.csv")
print("df_GI50:", df_GI50.shape, "\n", df_GI50.head(5))

NSC_column_name = ['NSC']
df_NSC_number = df_GI50[NSC_column_name]
df_unique_NSC_number = df_NSC_number.drop_duplicates().reset_index(drop=True)
print("\n\nUnique NSC Numbers Extracted from NCI60 Data:", df_unique_NSC_number.shape, "\n", df_unique_NSC_number)

df_NCI60_SMILES = pd.read_csv(r"C:\Users\lexha\Downloads\Visual_Studio_Code\NCI-60_Work\Output_Data\Canonical_SMILES_Output\nsc_canonical_smiles.csv")
print("\n\ndf_NCI60_SMILES:", df_NCI60_SMILES.shape, "\n", df_NCI60_SMILES.head(5))

df_NSC_with_SMILES = pd.merge(df_unique_NSC_number, df_NCI60_SMILES, on='NSC', how='left')
print("\n\ndf_NSC_with_SMILES:", df_NSC_with_SMILES.shape, "\n", df_NSC_with_SMILES.head(5))

df_Dropped_NSC_Numbers = df_NSC_with_SMILES[df_NSC_with_SMILES["SMILES"].isna()]["NSC"].unique()
df_NSC_with_SMILES.dropna(subset=["SMILES"], inplace=True)
df_NSC_with_SMILES.reset_index(drop=True, inplace=True)

print("\n\nTotal NSC numbers being dropped:", len(df_Dropped_NSC_Numbers), "\n", 
      df_Dropped_NSC_Numbers,
      "\n\nTotal unique SMILES strings to be processed through ChemBERTa:", len(df_NSC_with_SMILES))

unique_smiles_list = df_NSC_with_SMILES["SMILES"].unique().tolist()



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\nDevice being used: {device.upper()}\n")

model_name = "DeepChem/ChemBERTa-77M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

embedding_dict = {}
batch_size = 256

for i in range(0, len(unique_smiles_list), batch_size):
    batch = unique_smiles_list[i : i + batch_size]

    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()

    for smiles, emb in zip(batch, embeddings):
        embedding_dict[smiles] = emb

print(f"\nGenerated {len(embedding_dict)} unique chemical embeddings.")


mapped_embeddings = df_NSC_with_SMILES['SMILES'].map(embedding_dict)
emb_matrix = np.stack(mapped_embeddings.values)


emb_cols = [f'Emb_{i}' for i in range(emb_matrix.shape[1])]
df_embeddings_split = pd.DataFrame(emb_matrix, columns=emb_cols)


df_NSC_embedding_dict = pd.concat([df_NSC_with_SMILES[['NSC']], df_embeddings_split], axis=1)

print("\ndf_NSC_embedding_dict preview (Individual columns created):")
print(f"Shape: {df_NSC_embedding_dict.shape}")
print(df_NSC_embedding_dict.head(5))



exporting_choice = input("\nExport df_NSC_embedding_dict? (y/n): ").strip().lower()

if exporting_choice == "y":
    csv_output_path = r"C:\Users\lexha\Downloads\Visual_Studio_Code\NCI-60_Work\Output_Data\Transformer_Output\df_NSC_embedding_dict.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df_NSC_embedding_dict.to_csv(csv_output_path, index=False)
    print(f"Exported chemical embeddings to: {csv_output_path}")
elif exporting_choice == "n":
    print("df_NSC_embedding_dict export skipped.")