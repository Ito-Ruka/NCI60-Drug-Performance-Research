import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt



df_DOSERESP = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\DOSERESP.csv")
df_doubling = pd.read_csv(r"C:\Development\NCI-60_Work\Input_Data\NCI-60_Cell_Line_Doubling_Times.csv")



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



def remove_non_M_units(df_DOSERESP):
    unit = "M"
    df_DOSERESP = df_DOSERESP[df_DOSERESP["CONCENTRATION_UNIT"] == unit].copy()
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



def convert_to_AUC(df_DOSERESP):
        df_sorted = df_DOSERESP.sort_values(["NSC", "CELL_NAME", "CONCENTRATION"]).reset_index(drop=True)
        def calculate_AUC(group):
            log_conc = group["CONCENTRATION"].values.astype(np.float64)
            viability = group["VIABILITY"].values.astype(np.float64)
            
            span = log_conc[-1] - log_conc[0]

            if span <= 0:
                return np.nan
            
            raw_auc = np.sum(((viability[:-1] + viability[1:]) / 2.0) * np.diff(log_conc))

            return raw_auc / span
        
        df_DOSERESP = (df_sorted.groupby(["NSC", "CELL_NAME"]).apply(calculate_AUC, include_groups=False).reset_index(name="AUC"))
        df_DOSERESP = df_DOSERESP.dropna(subset=["AUC"]).reset_index(drop=True)
        
        return df_DOSERESP



def convert_to_AUC_nonlinear(df_DOSERESP, selected_conc_list, n_points = 100):

  def get_concentration_grid(concs, num_points):
    sorted_concs = sorted([float(c) for c in concs])
    min_conc = sorted_concs[0]
    max_conc = sorted_concs[-1]
    span = max_conc - min_conc
    dense_x = np.linspace(min_conc, max_conc, num_points)
    return dense_x, span

  def fit_pchip_and_integrate(concs, viab, dense_x, span):
    spline = PchipInterpolator(concs, viab)
    dense_y = spline(dense_x)
    raw_auc = np.trapezoid(dense_y, x=dense_x)
    return raw_auc / span

  def compute_group_auc(group, dense_x, span):
    concs = group["CONCENTRATION"].values.astype(np.float64)
    viab = group["VIABILITY"].values.astype(np.float64)
    return fit_pchip_and_integrate(concs, viab, dense_x, span)

  dense_x, span = get_concentration_grid(selected_conc_list, n_points)

  df_sorted = df_DOSERESP.sort_values(["NSC", "CELL_NAME", "CONCENTRATION"]).reset_index(drop=True)

  df_auc = (
      df_sorted.groupby(["NSC", "CELL_NAME"])
      .apply(compute_group_auc, dense_x=dense_x, span=span, include_groups=False)
      .reset_index(name="AUC")
      )

  return df_auc.dropna(subset=["AUC"]).reset_index(drop=True)



def plot_all_dose_response_curves(df_raw, df_auc, selected_conc_list, n_points=100):
  sorted_concs = sorted([float(c) for c in selected_conc_list])
  dense_x = np.linspace(sorted_concs[0], sorted_concs[-1], n_points)

  auc_lookup = df_auc.set_index(["NSC", "CELL_NAME"])["AUC"].to_dict()

  unique_curves = df_raw[["NSC", "CELL_NAME"]].drop_duplicates()
  print(
      f"\nDisplaying all {len(unique_curves)} dose-response curves"
      )

  for idx, (_, row) in enumerate(unique_curves.iterrows(), start=1):
    nsc = row["NSC"]
    cell = row["CELL_NAME"]

    group = df_raw[(df_raw["NSC"] == nsc) & (df_raw["CELL_NAME"] == cell)].sort_values("CONCENTRATION")

    concs = group["CONCENTRATION"].values.astype(np.float64)
    viab = group["VIABILITY"].values.astype(np.float64)

    spline = PchipInterpolator(concs, viab)
    dense_y = spline(dense_x)

    auc_val = auc_lookup.get((nsc, cell), np.nan)

    plt.figure(figsize=(7, 5))
    plt.plot(
        dense_x,
        dense_y,
        color="navy",
        linewidth=2.2,
        label="PCHIP Spline Interpolation"
    )
    plt.scatter(
        concs,
        viab,
        color="black",
        s=45,
        zorder=5,
        label="Experimental Viability"
    )

    plt.title(
        f"[{idx}/{len(unique_curves)}] Response: NSC {nsc} on {cell}\nNormalized"
        f" AUC: {auc_val:.4f}",
        fontsize=11
    )
    plt.xlabel("Log10 Concentration (M)", fontsize=10)
    plt.ylabel("Relative Cell Viability", fontsize=10)
    plt.ylim(-0.05, 1.4)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()



def data_frame_format(df_DOSERESP, is_single_cell_line):
  if is_single_cell_line:
    matrix_X = df_DOSERESP[["NSC"]].copy()
    matrix_Y = df_DOSERESP[["NSC", "AUC"]].copy()
  
  else:
    df_pivoted = df_DOSERESP.pivot(index="NSC", columns="CELL_NAME", values="AUC")
    matrix_X = pd.DataFrame({"NSC": df_pivoted.index})
    matrix_Y = df_pivoted.reset_index()

  return matrix_X, matrix_Y



def choose_embedding_version (embedding_version_choice):
    if (embedding_version_choice == 1):
        path = r"C:\Development\NCI-60_Work\Output_Data\Transformer_Output\df_NSC_embedding_dict.csv"
        print("Loading ChemBERTa embeddings.")
        return pd.read_csv(path)
    
    elif (embedding_version_choice == 2):
        path = r"C:\Development\NCI-60_Work\Main_Code\Chemprop\Chemprop_embedding_output\chemprop_dmpnn_embeddings.csv"
        print("Loading Chemprop embeddings.")
        return pd.read_csv(path)

    elif (embedding_version_choice == 3):
        path = r"C:\Development\NCI-60_Work\Main_Code\GraphMVP\GraphMVP_Embedding_Output\graphmvp_nsc_embeddings.csv"
        print("Loading GraphMVP embeddings.")
        return pd.read_csv(path)
    
    else:
        print("Invalid choice.")
        return None



def apply_embeddings(matrix_X, matrix_Y, df_embeddings):
    common_nscs = set(matrix_X["NSC"]).intersection(set(df_embeddings["NSC"]))
    missing_count = len(matrix_X) - len(common_nscs)

    valid_x = matrix_X["NSC"].isin(common_nscs)
    valid_y = matrix_Y["NSC"].isin(common_nscs)

    matrix_X = (matrix_X[valid_x].merge(df_embeddings, on="NSC", how="inner").reset_index(drop=True))
    matrix_Y = matrix_Y[valid_y].reset_index(drop=True)

    matrix_X = matrix_X.sort_values("NSC").reset_index(drop=True)
    matrix_Y = matrix_Y.sort_values("NSC").reset_index(drop=True)

    return matrix_X, matrix_Y, missing_count



def compress_latent_dimensions(matrix_X, latent_dim, epochs, lr = 1e-3):
    feature_cols = [c for c in matrix_X.columns if c != "NSC"]
    input_dim = len(feature_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Autoencoder(nn.Module):
        def __init__(self, in_dim, lat_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, lat_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(lat_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, in_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    x_data = torch.tensor(matrix_X[feature_cols].values, dtype=torch.float32).to(device)

    model = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"\nTraining Autoencoder ({input_dim}D into {latent_dim}D) for {epochs} epochs on {device}")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        recon = model(x_data)
        loss = criterion(recon, x_data)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}] - Reconstruction MSE: {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        latent_matrix = model.encoder(x_data).cpu().numpy()

    latent_cols = [f"latent_{i}" for i in range(latent_dim)]
    df_latent = pd.DataFrame(latent_matrix, columns=latent_cols)
    df_latent.insert(0, "NSC", matrix_X["NSC"].values)

    return df_latent



def get_prop():
    while True:
        try:
            test_prop = float(input("Enter proportion for test set: "))
            val_prop = float(input("Enter proportion for validation set: "))

            if not (0 < test_prop < 1) or not (0 < val_prop < 1):
                print("Error: Both proportions must staisfy range: (0, 1].")

            elif (test_prop + val_prop) >= 1:
                print(f"Error: Sum of Test ({test_prop}) + Val ({val_prop}) must be less than 1.0 so train set has data remaining.")
          
            else:
                train_prop = 1.0 - (test_prop + val_prop)
                print(
                    f"\nTrain: {train_prop*100:.1f}% | Val:"
                    f" {val_prop*100:.1f}% | Test: {test_prop*100:.1f}%"
                    )
                return test_prop, val_prop

        except ValueError:
            print("Please enter valid numeric floating point values.")



def split_data(test_prop, val_prop, matrix_X, matrix_Y, random_state):
    np.random.seed(random_state)

    unique_nscs = np.array(matrix_X["NSC"].unique())
    np.random.shuffle(unique_nscs)

    unique_nsc_count = len(unique_nscs)
    n_test = int(unique_nsc_count * test_prop)
    n_val = int(unique_nsc_count * val_prop)
    
    test_nscs = set(unique_nscs[:n_test])
    val_nscs = set(unique_nscs[n_test : n_test + n_val])
    train_nscs = set(unique_nscs[n_test + n_val :])

    train_mask = matrix_X["NSC"].isin(train_nscs)
    val_mask = matrix_X["NSC"].isin(val_nscs)
    test_mask = matrix_X["NSC"].isin(test_nscs)

    matrix_X_train = matrix_X[train_mask].reset_index(drop=True)
    matrix_X_val = matrix_X[val_mask].reset_index(drop=True)
    matrix_X_test = matrix_X[test_mask].reset_index(drop=True)

    matrix_Y_train = matrix_Y[train_mask].reset_index(drop=True)
    matrix_Y_val = matrix_Y[val_mask].reset_index(drop=True)
    matrix_Y_test = matrix_Y[test_mask].reset_index(drop=True)

    return (matrix_X_train, matrix_X_val, matrix_X_test, matrix_Y_train, matrix_Y_val, matrix_Y_test)



print(f"\nDropping data points with non M concenctration units.")
df_DOSERESP = remove_non_M_units(df_DOSERESP)
print(f"Data size after unit filteration: {df_DOSERESP.shape}")


print("\nStanderdizing experimental outputs to viability values.")
df_DOSERESP = convert_to_viability(df_DOSERESP, df_doubling)
print(f"Data size after VIABILITY unit conversion: {df_DOSERESP.shape}")


print(
    "\nAggregating experimental duplicates."
    "\nAggregation method = Medium"
    )
df_DOSERESP, df_duplicates, df_uniques = aggregate_experimental_duplicates(df_DOSERESP)
print(
    f"\nFound {len(df_duplicates)} total experimental duplicates."
    f"\nFound {len(df_uniques)} unique data points."
    f"\nData size after aggregation: {df_DOSERESP.shape}"
    )


cell_line_isolation = input("\nExtract screening profiles for an individual cell line? (y/n): ")
is_single_cell = (cell_line_isolation.strip().lower() == "y")
if (cell_line_isolation == "y"):
    cell_line_options = available_cell_lines(df_DOSERESP)
    with pd.option_context("display.max_rows", None):
        print(cell_line_options.to_string(index = False))
    
    cell_line_input = input("Enter target cell line: ")
    print(f"Isolating {cell_line_input} cell line response data.")    
    
    df_DOSERESP = select_cell_line(df_DOSERESP, cell_line_input)
    print(f"Data size after isolating {cell_line_input} cell line: {df_DOSERESP.shape}")
    
else:
    print("Continuing foward.")


print("\nStanderdizing dosage for drug response curve.")
df_conc_distribution = analyze_conc_distribution(df_DOSERESP)
print(df_conc_distribution)
print("NCI-60 standard 5 concentration values (log molar): -4.0000, -5.0000, -6.0000, -7.000, -8.0000")
selected_conc_list = [float(x.strip()) for x in input("Enter standardized concentrations (separated by comma): ").strip().split(",")]
selected_conc_list = sorted(selected_conc_list)
df_DOSERESP = drop_invalid_conc(df_DOSERESP, selected_conc_list)

df_raw_doses = df_DOSERESP.copy()

df_sorted = df_DOSERESP.sort_values(["NSC", "CELL_NAME", "CONCENTRATION"]).reset_index(drop=True)
print("\n", df_sorted)

"""
print("\nCalculating the AUC of the dose response curves.")
while True:
    try:
        auc_choice = int(input("\nCalculate AUC using standard trapezoid (1) or piecewise cubic hermite interpolating polynomial (2): ").strip())
        if auc_choice in [1, 2]:
            break
        print("Enter 1 or 2")
    except ValueError:
        print("Invalid input")

if auc_choice == 1:
    print("Computing AUC via standard trapezoidal Riemann sum.")
    df_DOSERESP = convert_to_AUC(df_DOSERESP)

elif auc_choice == 2:
  print("Computing AUC via PCHIP non-linear spline integration.")
  df_DOSERESP = convert_to_AUC_nonlinear(df_DOSERESP, selected_conc_list, n_points=100)

  graph_choice = (input("\nPlot all dose-response curves in df_DOSERESP? (y/n): ").strip().lower())
  if graph_choice == "y":
    plot_all_dose_response_curves(df_raw_doses, df_DOSERESP, selected_conc_list, n_points=100)

print(f"\nData size after conversion to AUC value: {df_DOSERESP.shape}")
"""

print("\nFormatting data frame.")
matrix_X, matrix_Y = data_frame_format(df_DOSERESP, is_single_cell_line=is_single_cell)
print(
    f"Matrix X size: {matrix_X.shape}\n"
    f"Matrix Y size: {matrix_Y.shape}"
    )


print("Selecting embedding versions: ChemBERTa [1], Chemprop [2], GraphMVP [3]")
embedding_version_choice = int(input("Enter selection: "))
df_embeddings = choose_embedding_version(embedding_version_choice)


print("\nAttaching chemical embeddings.")
matrix_X, matrix_Y, missing_count = apply_embeddings(matrix_X, matrix_Y, df_embeddings)
print(
    f"{missing_count} data points with missing embeddings detected."
    f"\nDropping missing data points"
    f"\nmatrix_X data size after applying embeddings: {matrix_X.shape}"
    f"\nmatrix_Y data size after applying embeddings: {matrix_Y.shape}"
    )


compress_choice = (input("\nCompress embeddings to 128D latent space? (y/n): ").strip().lower())
if compress_choice == "y":
    latent_dim = int(input("Enter latent dimension: "))
    epoch = int(input("Enter number of epochs for autoencoder: "))
    matrix_X = compress_latent_dimensions(matrix_X, latent_dim = latent_dim, epochs = epoch)
    print(f"Compressed matrix_X shape: {matrix_X.shape}")

else:
    print("Keeping original embeddings.")


print("\nSplitting feature and target matrices into train, val, and test sets")
test_prop, val_prop = get_prop()
matrix_X_train, matrix_X_val, matrix_X_test, matrix_Y_train, matrix_Y_val, matrix_Y_test = split_data(test_prop, val_prop, matrix_X, matrix_Y, random_state=10)
print(f"Train sizes | Features: {matrix_X_train.shape}, Targets:"f" {matrix_Y_train.shape}")
print(f"Val sizes   | Features: {matrix_X_val.shape}, Targets:"f" {matrix_Y_val.shape}")
print(f"Test sizes  | Features: {matrix_X_test.shape}, Targets:"f" {matrix_Y_test.shape}")


preview_choice = input("\n\nPreview Train, Validation, and Test CSV files? (y/n): ").strip().lower()
if (preview_choice == "y"):
  splits = [
    ("Train", matrix_X_train, matrix_Y_train),
    ("Validation", matrix_X_val, matrix_Y_val),
    ("Test", matrix_X_test, matrix_Y_test)
    ]

  for name, df_x, df_y in splits:
    print(f"\n{name} Matrix X")
    print(df_x.head(5))
    print(f"\n{name} Matrix Y")
    print(df_y.head(5))

else: 
    print("Continuing to exports.")


export_choice = (input("\nExport Train, Validation, and Test CSV files? (y/n): ").strip().lower())
if export_choice == "y":
    matrix_X_train.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_train_matrix_X.csv", index=False)
    matrix_Y_train.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_train_matrix_Y.csv", index=False)

    matrix_X_val.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_val_matrix_X.csv", index=False)
    matrix_Y_val.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_val_matrix_Y.csv", index=False)

    matrix_X_test.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_test_matrix_X.csv", index=False)
    matrix_Y_test.to_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_test_matrix_Y.csv", index=False)

    print("\nExported Train, Val, and Test CSVs")
else:
    print("\nExport sequence cancelled.")

