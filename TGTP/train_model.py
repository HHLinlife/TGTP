import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# Define the 1D - CNN model with more layers and complexity
class OneDCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(64 * (input_channels // 4), 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define the Var - CNN model with more layers and complexity
class VarCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(VarCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.fc1 = nn.Linear(128 * (input_channels // 4), 256)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load the dataset with more data pre - processing steps
def load_dataset(dataset_name):
    if dataset_name == 'CIC-DDoS2019':
        data = pd.read_csv('CIC-DDoS2019.csv')
        # Handle missing values
        data = data.dropna()
        X = data.drop('label', axis=1).values
        y = data['label'].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, y_train, X_val, y_val
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# Training the model with validation and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=5):
    best_val_loss = float('inf')
    no_improvement_count = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), f'{type(model).__name__}_{args.dataset}_best.pth')
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break


def main():
    parser = argparse.ArgumentParser(description='Train a surrogate model')
    parser.add_argument('--model', type=str, required=True, help='Model name (1D-CNN, Var-CNN)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., CIC-DDoS2019)')
    args = parser.parse_args()

    # Load the dataset
    X_train, y_train, X_val, y_val = load_dataset(args.dataset)
    num_classes = len(torch.unique(y_train))
    input_channels = X_train.shape[1]

    # Select the model
    if args.model == '1D-CNN':
        model = OneDCNN(input_channels, num_classes)
    elif args.model == 'Var-CNN':
        model = VarCNN(input_channels, num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)


if __name__ == "__main__":
    main()
