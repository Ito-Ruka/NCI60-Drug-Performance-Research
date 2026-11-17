import csv
import os
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df_test_X = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_test_matrix_X.csv")
df_test_Y = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_test_matrix_Y.csv")

print("df_test_X Shape:", df_test_X.shape)
print("df_test_Y Shape:", df_test_Y.shape)

feature_cols = [c for c in df_test_X.columns if c != "NSC"]
target_cols = [c for c in df_test_Y.columns if c != "NSC"]

X_test = df_test_X[feature_cols].values.astype(np.float32)
Y_test_actual = df_test_Y[target_cols].values.astype(np.float32)



class DrugResponseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out1=0.1, drop_out2=0.0, drop_out3=0.0):
        super(DrugResponseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 720),
            nn.ReLU(),
            nn.Dropout(drop_out1),
            nn.Linear(720, 220),
            nn.ReLU(),
            nn.Dropout(drop_out2),
            nn.Linear(220, 128),
            nn.ReLU(),
            nn.Dropout(drop_out3),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)



class SmallDrugResponseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out1=0.1, drop_out2=0.0):
        super(SmallDrugResponseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_out1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop_out2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(feature_cols)
output_dim = len(target_cols)

model_path = input("\nPaste the .pt file path: ").strip().strip('"').strip("'")

filename_lower = os.path.basename(model_path).lower()
if "small" in filename_lower:
    arch_name = "Small MLP"
    model = SmallDrugResponseMLP(input_dim=input_dim, output_dim=output_dim).to(device)
elif "regular" in filename_lower or "mlp" in filename_lower:
    arch_name = "Regular MLP"
    model = DrugResponseMLP(input_dim=input_dim, output_dim=output_dim).to(device)
else:
    print("\nCould not automatically determine architecture from file name.")
    print("Architectures: [1] Regular MLP  [2] Small MLP")
    arch_choice = int(input("Select architecture (1/2): "))
    if arch_choice == 1:
        arch_name = "Regular MLP"
        model = DrugResponseMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    else:
        arch_name = "Small MLP"
        model = SmallDrugResponseMLP(input_dim=input_dim, output_dim=output_dim).to(device)

arch_slug = arch_name.lower().replace(" ", "_")
print(f"Using architecture: {arch_name}")

checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)
if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
    model.load_state_dict(checkpoint_data["model_state_dict"])
else:
    model.load_state_dict(checkpoint_data)

model.eval()
print(f"Model successfully loaded from: {model_path}")

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds_actual = model(X_test_tensor).cpu().numpy()


cell_line_metrics = []
all_true_valid = []
all_pred_valid = []

for idx, col in enumerate(target_cols):
    y_true = Y_test_actual[:, idx]
    y_pred = preds_actual[:, idx]

    mask = ~np.isnan(y_true)

    if mask.sum() > 1:
        yt = y_true[mask]
        yp = y_pred[mask]

        mse = np.mean((yt - yp) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(yt - yp))
        corr = np.corrcoef(yt, yp)[0, 1]

        cell_line_metrics.append({
            "cell_line": col,
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "pearson_r": float(corr),
        })

        all_true_valid.extend(yt)
        all_pred_valid.extend(yp)

overall_mse = np.mean((np.array(all_true_valid) - np.array(all_pred_valid)) ** 2)
overall_rmse = np.sqrt(overall_mse)
overall_mae = np.mean(np.abs(np.array(all_true_valid) - np.array(all_pred_valid)))
overall_pearson_r = np.corrcoef(all_true_valid, all_pred_valid)[0, 1]

print(f"\n--- {arch_name} Evaluation Summary ---")
print(f"Overall Test MSE: {overall_mse:.4f}")
print(f"Overall Test RMSE: {overall_rmse:.4f}")
print(f"Overall Test MAE: {overall_mae:.4f}")
print(f"Overall Pearson Correlation (r): {overall_pearson_r:.4f}")

mean_cell_line_mse = np.mean([m["mse"] for m in cell_line_metrics])
mean_cell_line_rmse = np.mean([m["rmse"] for m in cell_line_metrics])
mean_cell_line_mae = np.mean([m["mae"] for m in cell_line_metrics])
mean_cell_line_pearson_r = np.mean([m["pearson_r"] for m in cell_line_metrics])

print(f"Mean Cell-Line MSE: {mean_cell_line_mse:.4f}")
print(f"Mean Cell-Line RMSE: {mean_cell_line_rmse:.4f}")
print(f"Mean Cell-Line MAE: {mean_cell_line_mae:.4f}")
print(f"Mean Cell-Line Pearson Correlation (r): {mean_cell_line_pearson_r:.4f}")


eval_out_dir = r"C:\Development\NCI-60_Work\Main_Code\Test\Test_Output"
os.makedirs(eval_out_dir, exist_ok=True)


user_choice = input("\nDo you want to export cell-line performance metrics to a CSV file? (y/n): ").strip().lower()

if user_choice == "y":
    csv_path = os.path.join(eval_out_dir, f"test_evaluation_metrics_{arch_slug}.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["cell_line", "mse", "rmse", "mae", "pearson_r"])
        writer.writeheader()
        writer.writerows(cell_line_metrics)
    print(f"Evaluation metrics exported to: {csv_path}")
else:
    print("Export skipped.")

user_choice_preds = input("\nExport actual vs predicted values to a CSV file? (y/n): ").strip().lower()

if user_choice_preds == "y":
    pred_dict = {"NSC": df_test_Y["NSC"]}
    if len(target_cols) == 1:
        target_name = target_cols[0]
        pred_dict[f"Actual_{target_name}"] = Y_test_actual[:, 0]
        pred_dict[f"Predicted_{target_name}"] = preds_actual[:, 0]
    else:
        for idx, col in enumerate(target_cols):
            pred_dict[f"Actual_{col}"] = Y_test_actual[:, idx]
            pred_dict[f"Predicted_{col}"] = preds_actual[:, idx]

    df_predictions = pd.DataFrame(pred_dict)

    csv_preds_path = os.path.join(eval_out_dir, f"test_predictions_actual_vs_predicted_{arch_slug}.csv")
    df_predictions.to_csv(csv_preds_path, index=False)
    print(f"Actual vs Predicted values exported to: {csv_preds_path}")
else:
    print("Export skipped.")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

all_true_arr = np.array(all_true_valid)
all_pred_arr = np.array(all_pred_valid)

max_display_points = 50000
if len(all_true_arr) > max_display_points:
    indices = np.random.choice(len(all_true_arr), max_display_points, replace=False)
    plot_true = all_true_arr[indices]
    plot_pred = all_pred_arr[indices]
else:
    plot_true = all_true_arr
    plot_pred = all_pred_arr

ax.scatter(
    plot_true,
    plot_pred,
    alpha=0.25,
    color="#20639B",
    edgecolors="none",
    s=12,
)

min_val = float(np.nanmin(plot_true)) - 0.05 if len(plot_true) > 0 else 0.0
max_val = float(np.nanmax(plot_true)) + 0.05 if len(plot_true) > 0 else 1.0
ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="#ED553B",
    linestyle="--",
    linewidth=2,
    label="Ideal Prediction (1:1)",
)

ax.set_title(
    f"NCI-60 {arch_name} Regression: Actual vs. Predicted AUC",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Actual AUC", fontsize=12, labelpad=10)
ax.set_ylabel("Predicted AUC", fontsize=12, labelpad=10)

stats_text = (
    f"Architecture: {arch_name}\n"
    f"MSE: {overall_mse:.4f}\n"
    f"RMSE: {overall_rmse:.4f}\n"
    f"MAE: {overall_mae:.4f}\n"
    f"Pearson r: {overall_pearson_r:.4f}\n"
    #f"Mean Line r: {mean_cell_line_pearson_r:.4f}"
)
ax.text(
    0.05,
    0.74,
    stats_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
)

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()

plot_output_path = os.path.join(eval_out_dir, f"{arch_slug}_regression_actual_vs_predicted.png")
plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Regression plot successfully saved to: {plot_output_path}")