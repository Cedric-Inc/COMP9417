import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


def compute_effective_number_weights(y, beta=0.9999):
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    class_weights = 1.0 / effective_num
    class_weights = class_weights / np.sum(class_weights) * len(classes)

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weight = np.array([class_weight_dict[label] for label in y])
    return sample_weight, class_weight_dict


class SimpleClassifier(nn.Module):

    def __init__(self, input_dim=300, num_classes=28,drouout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=300, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drouout),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=28)
        )


    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


import torch, torch.nn as nn, torch.nn.functional as F


class GRN(nn.Module):
    """
    Gated Residual Network block for tabular MLP
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)  #
        self.gate = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.elu(x)
        x = self.drop(x)
        x = self.fc2(x)

        g = torch.sigmoid(self.gate(residual))  #
        x = x * g  #

        x = x + residual  # 
        return self.ln(x)  # LayerNorm


class TabNetMLP(nn.Module):
    def __init__(self, in_dim=300, d_model=512, depth=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([GRN(d_model, dropout) for _ in range(depth)])
        self.head = nn.Linear(d_model, 28)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)  # logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


def default_neural_network(X_train, y_train, n_epochs=60, batch_size=64, learning_rate=0.001, dropout=0.3,
                           class_weight='balanced'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model = SimpleClassifier().to(device)
    # model = TabNetMLP(in_dim=300, d_model=512, depth=4, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if class_weight == 'balanced':
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif class_weight == 'effective':
        sample_weight, class_weight_dict = compute_effective_number_weights(y_train)
        class_weights_array = np.array([class_weight_dict[i] for i in range(num_classes)])
        class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif class_weight == 'label_smoothing':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(int(n_epochs)):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}] - Loss: {running_loss / len(train_loader):.4f}")
    return model

