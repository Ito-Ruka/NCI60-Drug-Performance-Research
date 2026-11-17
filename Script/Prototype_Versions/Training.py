import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import csv


df_train_X = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_train_matrix_X.csv")
df_train_Y = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_train_matrix_Y.csv")
df_val_X = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_val_matrix_X.csv")
df_val_Y = pd.read_csv(r"C:\Development\NCI-60_Work\Main_Code\Data_Frame\Data__Frame_Output\df_val_matrix_Y.csv")



def convert_to_numpy(df_train_X, df_train_Y, df_val_X, df_val_Y):
    feature_cols = df_train_X.columns.drop("NSC")
    target_cols = df_train_Y.columns.drop(["NSC"])
    
    X_train = df_train_X[feature_cols].values.astype(np.float32)
    Y_train = df_train_Y[target_cols].values.astype(np.float32)
    X_val = df_val_X[feature_cols].values.astype(np.float32)
    Y_val = df_val_Y[target_cols].values.astype(np.float32)

    return (X_train, Y_train, X_val, Y_val)



def convert_to_tensor(X_train, Y_train, X_val, Y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

    return (device, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor)



def masked_huber_loss(pred, target, delta=.5):
  mask = ~torch.isnan(target)
  if mask.sum() == 0:
    return torch.tensor(0.0, device=pred.device, requires_grad=True)

  diff = torch.abs(pred[mask] - target[mask])
  huber = torch.where(
      diff < delta, 0.5 * (diff**2), delta * (diff - 0.5 * delta)
  )
  return torch.mean(huber)



def masked_L1_loss(pred, target):
  mask = ~torch.isnan(target)
  if mask.sum() == 0:
    return torch.tensor(0.0, device=pred.device, requires_grad=True)

  return torch.mean(torch.abs(pred[mask] - target[mask]))



def masked_L2_loss(pred, target):
  mask = ~torch.isnan(target)
  if mask.sum() == 0:
    return torch.tensor(0.0, device=pred.device, requires_grad=True)

  return torch.mean((pred[mask] - target[mask]) ** 2)



class DrugResponseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out1, drop_out2, drop_out3):
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

  def __init__(self, input_dim, output_dim, drop_out1, drop_out2):
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



class Checkpoint:
  def __init__(self, patience, min_delta, checkpoint_dir, arch_name):
    self.patience = patience
    self.min_delta = min_delta
    self.checkpoint_dir = checkpoint_dir
    self.arch_name = arch_name.lower().replace(" ", "_")
    self.best_loss = float("inf")
    self.counter = 0
    self.best_model_wts = None
    self.best_epoch = 0

  def checkpoint(self, epoch, val_loss, model):
    if (val_loss < (self.best_loss - self.min_delta)):
      self.best_loss = val_loss
      self.counter = 0
      self.best_model_wts = copy.deepcopy(model.state_dict())
      self.best_epoch = epoch
    
    else:
      self.counter += 1
      if self.counter >= self.patience:
        save_path = os.path.join(self.checkpoint_dir, f"nci60_{self.arch_name}_checkpoint_epoch_{self.best_epoch}_loss_{self.best_loss:.4f}.pt",)
        torch.save(self.best_model_wts, save_path)
        print(
            f"\nPatience reached at Epoch [{epoch}]."
            f"\nSaved best weights from Epoch [{self.best_epoch}]"
            "Resetting patience counter and continuing training.\n"
            )
        self.counter = 0



def training_loop(model, optimizer, criterion, tracker, tensors, hyperparameters, periodic_dir, final_epoch_path, arch_name):
    X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor = tensors
    history = []
    max_epoch = hyperparameters["max_epoch"]
    periodic_saves = hyperparameters["periodic_saves"]
    arch_slug = arch_name.lower().replace(" ", "_")

    print(f"\nStarting traning for up to {max_epoch} epochs.")

    for epoch in range(1, max_epoch + 1):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, Y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, Y_val_tensor).item()

        train_loss_val = train_loss.item()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_val,
            "val_loss": val_loss,
        })

        print(
            f"Epoch [{epoch:04d}/{max_epoch}] - Train Loss: {train_loss_val:.5f} |"
            f" Val Loss: {val_loss:.5f}"
        )

        tracker.checkpoint(epoch, val_loss, model)

        if periodic_saves > 0 and epoch % periodic_saves == 0:
            periodic_path = os.path.join(
                periodic_dir, f"nci60_{arch_slug}_periodic_epoch_{epoch}.pt"
            )

            torch.save(model.state_dict(), periodic_path)
            print(f"[Periodic Save] Saved model state at Epoch [{epoch}].")

    torch.save(model.state_dict(), final_epoch_path)
    print(f"\nTraining complete. Saved final epoch ({max_epoch}) model state to: {final_epoch_path}")

    return history



print("\nLoading ML inputs.")
print(f"Train shapes | Features (X): {str(df_train_X.shape):<15} | Targets (Y): {str(df_train_Y.shape)}")
print(f"Val shapes   | Features (X): {str(df_val_X.shape):<15} | Targets (Y): {str(df_val_Y.shape)}")

X_train, Y_train, X_val, Y_val = convert_to_numpy(df_train_X, df_train_Y, df_val_X, df_val_Y)
device, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor = (convert_to_tensor(X_train, Y_train, X_val, Y_val))


print("\nLoss Functions: [1] Huber  [2] L1 (MAE)  [3] L2 (MSE)")
loss_choice = int(input("Select loss function (1/2/3): "))
if (loss_choice == 1):
  criterion = masked_huber_loss
  loss_name = "Masked Huber Loss"
elif (loss_choice == 2):
  criterion = masked_L1_loss
  loss_name = "Masked L1 Loss"
else:
  criterion = masked_L2_loss
  loss_name = "Masked L2 Loss"
print(f"Running MLP using: {loss_name}")


print("\nModel Architectures: [1] Regular MLP  [2] Small MLP")
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
arch_choice = int(input("Select MLP architecture (1/2): "))

if (arch_choice == 1):
  arch_name = "Regular MLP"

  drop_out1 = float(input("Enter dropout for Layer 1: ") or 0.1)
  drop_out2 = float(input("Enter dropout for Layer 2: ") or 0.0)
  drop_out3 = float(input("Enter dropout for Layer 3: ") or 0.0)

  model = DrugResponseMLP(
    input_dim = input_dim,
    output_dim = output_dim,
    drop_out1 = drop_out1,
    drop_out2 = drop_out2,
    drop_out3 = drop_out3
    ).to(device)

else:
  arch_name = "Small MLP"
  drop_out1 = float(input("Enter dropout for Layer 1: ") or 0.1)
  drop_out2 = float(input("Enter dropout for Layer 2: ") or 0.0)

  model = SmallDrugResponseMLP(
    input_dim = input_dim,
    output_dim = output_dim,
    drop_out1 = drop_out1,
    drop_out2 = drop_out2
    ).to(device)

print(f"Using Architecture: {arch_name}")


print("\nSetting up hyperparameters.")
max_epoch = int(input("Enter maximum epoch value: "))
patience = int(input("Enter patience value: "))
min_delta = float(input("Enter the minimum decrease in validation loss required to count as a genuine improvement: "))
lr = float(input("Enter learning rate value: "))
weight_decay = float(input("Enter weight decay value: "))
periodic_saves = int(input("Enter periodic save interval: "))


hyperparameters = {
    "max_epoch": max_epoch,
    "patience": patience,
    "min_delta": min_delta,
    "lr": lr,
    "weight_decay": weight_decay,
    "periodic_saves": periodic_saves,
    }


base_out_dir = (r"C:\Development\NCI-60_Work\Main_Code\Training\Training_Output")
checkpoint_dir = os.path.join(base_out_dir, "ML_Training(Checkpoints)")
periodic_dir = os.path.join(base_out_dir, "ML_Training(Periodic_Saves)")
logs_dir = os.path.join(base_out_dir, "Train_&_Validation_Loss_Logs")

for folder in [base_out_dir, checkpoint_dir, periodic_dir, logs_dir]:
  os.makedirs(folder, exist_ok=True)

arch_slug = arch_name.lower().replace(" ", "_")
final_epoch_path = os.path.join(base_out_dir, f"nci60_{arch_slug}_final_epoch.pt")


optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
tracker = Checkpoint(
  patience = patience,
  min_delta = min_delta,
  checkpoint_dir = checkpoint_dir,
  arch_name = arch_name
  )
print(f"\nModel initialized on: {device}")


tensors = (X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor)
history = training_loop(
    model = model,
    optimizer = optimizer,
    criterion = criterion,
    tracker = tracker,
    tensors = tensors,
    hyperparameters = hyperparameters,
    periodic_dir = periodic_dir,
    final_epoch_path = final_epoch_path,
    arch_name = arch_name
    )

exporting_choice = (input("\nExport train and validation loss logs to CSV? (y/n): "))
if exporting_choice == "y":
  csv_output_path = os.path.join(logs_dir, f"training_loss_history_{arch_slug}.csv")
  with open(csv_output_path, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
    writer.writeheader()
    writer.writerows(history)
  print(f"Exported loss history to: {csv_output_path}")
else:
  print("Export sequence cancelled.")