import numpy as np
import torch
import data_processing as dp
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from models.NRMSELoss import NRMSELoss


########################
# EVALUATING THE MODEL #
########################
def print_metrics(y_test, y_pred):
    print('Performance on the test set:')

    # NRMSE
    range = torch.quantile(y_test, 0.99) - torch.quantile(y_test, 0.01)
    loss = NRMSELoss(range)
    test_loss = loss(y_test, y_pred)
    print(f'NRMSE = {test_loss.item():.4f}')

    y_test = y_test.detach().squeeze().numpy()
    y_pred = y_pred.detach().squeeze().numpy()

    # Correlation
    r = np.corrcoef(y_test, y_pred)
    print('r =', r[0, 1])

    # Scatterplot
    plt.figure(figsize=(3, 3))
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.scatter(y_test, y_pred)
    plt.show()


####################
# CROSS-VALIDATION #
####################
def cross_validate(model, X, Y, strata, cv, pca=False):
    scores = []

    fold = 0
    for train_idx, val_idx in cv.split(X, strata):
        fold += 1

        # Make train-test split
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Perform PCA if necessary
        if pca:
            X_train, X_val = dp.perform_pca(X_train, X_val)

        # Train the model
        model.train_(X_train, Y_train)

        # Test the model
        mse = model.test(X_val, Y_val)
        scores.append(mse)

    return scores
