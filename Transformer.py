import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Loads the Sudoku dataset
df = pd.read_csv("SUDOKU.csv")

# Remove puzzles that are too hard (puzzles with very high solver attempts)
df = df[df["attempts"] <= 100000].copy()

# Apply log transform to reduce skew of the attempts
df["log_attempts"] = np.log1p(df["attempts"])

# Assign difficulty based on log attempts
df["difficulty"] = pd.cut(
    df["log_attempts"],
    bins=[df["log_attempts"].min(), 6.04, 6.98, df["log_attempts"].max()],
    labels=[0, 1, 2],  # 0 = Easy, 1 = Medium, 2 = Hard
    include_lowest=True
)
df["difficulty"] = df["difficulty"].astype(int) #declaring the difficulties as an integer

# Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["difficulty"], random_state=42)

# Preprocessing the dataset
class SudokuDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert the puzzle string to a grid of integers
        grid = np.array([int(c) for c in row["puzzle"]], dtype=np.int64)
        label = torch.tensor(row["difficulty"], dtype=torch.long)
        return torch.tensor(grid), label

# Loaders for batching and shuffling data
train_loader = DataLoader(SudokuDataset(train_df), batch_size=128, shuffle=True)
val_loader = DataLoader(SudokuDataset(val_df), batch_size=128)

# Transformer model 
class SudokuTransformerClassifier(nn.Module):
    def __init__(self, emb_dim=64, num_heads=8, num_layers=4, ff_dim=256, dropout=0.1): #setting the embedding vector dimension, attention heads, encoder layers, feed-forward network, and dropout rate
        super().__init__()
        self.embedding = nn.Embedding(10, emb_dim)  # For digits 0–9
        self.position_emb = nn.Parameter(torch.randn(82, emb_dim))  # 81 digits and the CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))   # CLS token that will store the information of the puzzle

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Layer
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Outputs to three classes: Easy, Medium, Hard
        )

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, 81, emb_dim)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # (batch_size, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Adds CLS token to the front
        x = x + self.position_emb     # Adds position encoding
        x = self.transformer(x)     # Run through transformer
        cls_output = x[:, 0, :]      # Use the CLS token output
        return self.fc(cls_output)

# Training the Model
def train_model(model, train_loader, val_loader, epochs=20, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) #Declaring the learning rate (AdamW)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)  #Cosine learning rate scheduler
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) #Cross entropy loss

    train_acc_list, val_acc_list = [], [] #Array where we will store training and validation accuracy

    for epoch in range(epochs): 
        model.train()  
        correct, total = 0, 0 #Declaring variables used for tallying correct puzzles guessed and total puzzles guessed
        loop = tqdm(train_loader, desc="Epoch " + str(epoch + 1) + "/" + str(epochs)) #Progress bar

        # Training pass
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y) #calculating loss
            loss.backward() #backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Gradient clipping to avoid exploding gradients
            optimizer.step()
            scheduler.step(epoch + loop.n / loop.total)

            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total #Calculating training accuracy
        train_acc_list.append(train_acc)

        # Validation pass
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        val_acc = correct_val / total_val
        val_acc_list.append(val_acc)

        print("")
        print("Epoch", epoch + 1)
        print("Train Accuracy:", train_acc)
        print("Validation Accuracy:", val_acc)

    # Plot training and validation accuracy
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()




# Run training and saving model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformerClassifier()
torch.save(model.state_dict(), "Transformer.pth")
train_model(model, train_loader, val_loader, epochs=20, device=device)
