import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold

import data_processing as dp
from matplotlib import pyplot as plt

from loss_functions.nrmse_loss import NRMSELoss


########################
# EVALUATING THE MODEL #
########################
def print_metrics(y_test, y_pred):
    print('Performance on the test set:')

    # NRMSE
    range = torch.quantile(y_test, 0.99) - torch.quantile(y_test, 0.01)
    loss = NRMSELoss([range])
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
    plt.scatter(y_test, y_pred, s=5)
    plt.show()


####################
# CROSS-VALIDATION #
####################
def cross_validate(model, X, Y, strata, cv):
    mses, rs = [], []

    if isinstance(cv, StratifiedKFold):
        folds = cv.split(X, strata)
    elif isinstance(cv, GroupKFold):
        folds = cv.split(X, groups=strata)

    fold = 0
    for train_idx, val_idx in folds:
        fold += 1

        # Make train-test split
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Perform PCA
        X_train, X_val = dp.perform_pca(X_train, X_val)

        # Train the model
        model.train_(X_train, Y_train)

        # Test the model
        mse, r = model.test(X_val, Y_val)

        mses.append(np.mean(mse))
        rs.append(np.mean(r))

    return mses, rs


###############
# CORRELATION #
###############
def plot_correlations(Y_test, Y_pred):
    fig, axs = plt.subplots(2, 4, figsize=(12, 9), sharey='col')

    LABELS = ['Fx_l', 'Fy_l', 'Fz_l', 'Tz_l',
              'Fx_r', 'Fy_r', 'Fz_r', 'Tz_r']
    COLORS = ['red', 'green', 'blue', 'blue']

    for i, label in enumerate(LABELS):
        y_test = Y_test[:, i].detach().numpy()
        y_pred = Y_pred[:, i].detach().numpy()

        # Correlation
        r = np.corrcoef(y_test, y_pred)[0, 1]

        # Scatterplot
        scatterplot = axs[i // 4, i % 4]
        scatterplot.set_title(f'{label} {r}')
        scatterplot.set_xlabel(f'{label}_test')
        scatterplot.set_ylabel(f'{label}_pred')
        scatterplot.set_xlim(np.min(y_test), np.max(y_test))
        scatterplot.set_ylim(np.min(y_pred), np.max(y_pred))
        scatterplot.scatter(y_test, y_pred, s=5, color=COLORS[i % 4])

    plt.show()