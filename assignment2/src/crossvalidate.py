# Evaluate the model with K-Fold Cross Validation
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np

def cross_validate(instances, x_combined, edge_index, targets, model, lr=0.001):
    num_folds = 4
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Initialize lists to store the loss for each fold
    rmse_losses = []
    mae_losses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(instances)):
        # Split the dataset into training and validation sets
        train_data = Data(x=x_combined, edge_index=edge_index, y=targets)
        train_data.train_mask = torch.zeros(len(instances), dtype=torch.bool)
        train_data.train_mask[train_idx] = 1
        val_data = Data(x=x_combined, edge_index=edge_index, y=targets)
        val_data.test_mask = torch.zeros(len(instances), dtype=torch.bool)
        val_data.test_mask[test_idx] = 1

        # Training loop
        model.train()
        num_epochs = 500
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions = model(train_data.x, train_data.edge_index)
            loss = criterion(predictions[train_data.train_mask], train_data.y[train_data.train_mask])
            loss.backward()
            optimizer.step()

        # Compute the loss on the test set with RSME and MAE
        model.eval()
        with torch.no_grad():
            test_predictions = model(val_data.x, val_data.edge_index)
            rmse = torch.sqrt(criterion(test_predictions[val_data.test_mask], val_data.y[val_data.test_mask]))
            mae = torch.mean(torch.abs(test_predictions[val_data.test_mask] - val_data.y[val_data.test_mask]))
            #print(f"Fold {fold+1}/{num_folds}, RMSE: {rmse.item():.4f}, MAE: {mae.item():.4f}")
            rmse_losses.append(rmse.item())
            mae_losses.append(mae.item())

    mean_rmses = np.mean(rmse_losses)
    mean_maes = np.mean(mae_losses)
    print(f"Across folds, mean RMSE: {mean_rmses:.4f}, mean MAE: {mean_maes:.4f}")
    return mean_rmses, mean_maes