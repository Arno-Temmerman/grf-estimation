import itertools

import pandas as pd
from pandas import DataFrame

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import os

import joblib

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from models.NRMSELoss import NRMSELoss
from models.base_regressor import BaseRegressor


#######################
# BUILDING THE HEADER #
#######################
LABELS = [
    # Kistler force plates
    'Fx_l','Fy_l','Fz_l', 'M_l',
    'Fx_r','Fy_r','Fz_r', 'M_r'
]
INPUTS = [
    # Moticon insoles
    'Ftot_l', 'CoPx_l', 'CoPy_l',
    'Ftot_r', 'CoPx_r', 'CoPy_r',

    'P1_l', 'P2_l', 'P3_l', 'P4_l', 'P5_l', 'P6_l', 'P7_l', 'P8_l', 'P9_l', 'P10_l', 'P11_l', 'P12_l', 'P13_l', 'P14_l', 'P15_l', 'P16_l',
    'ax_l', 'ay_l', 'az_l',
    'angx_l', 'angy_l', 'angz_l',

    'P1_r', 'P2_r', 'P3_r', 'P4_r', 'P5_r', 'P6_r', 'P7_r', 'P8_r', 'P9_r', 'P10_r', 'P11_r', 'P12_r', 'P13_r', 'P14_r', 'P15_r', 'P16_r',
    'ax_r', 'ay_r', 'az_r',
    'angx_r', 'angy_r', 'angz_r',

    # Cometa EMG sensors
    'vm_r',  'vm_l',  # vastus medialis
    'vr_r',  'vr_l',  # vastus radialis
    'gm_r',  'gm_l',  # gluteus medius
    'tfl_r', 'tfl_l' # tensor fasciae latae
]
MASKS = [
    'fp1', 'fp2', 'fp3', 'fp4',
    'valid_mask_feet', # 0 = occluded markers
    'correct_mask_fp', # 0 = force plate gave bad prediction
    'correct_mask_ins' # 0 = foot incorrectly placed on force plate
]
HEADER = LABELS + INPUTS + MASKS


####################
# LOADING THE DATA #
####################
DATA_DIR = "../segmented_data/"

# Reads the csv-files in ./segmented_data of the subjects-scenes-trials combinations specified above
def read_gait_cycles(subjects, scenes, trials):
    # Adds the features of several past and future timesteps to each row
    def expand_timeframe(gait_cycle):
        NR_OF_TIMESTEPS = 6

        past   = [ 2**i for i in range(NR_OF_TIMESTEPS)]
        future = [-2**i for i in range(NR_OF_TIMESTEPS)]

        for offset in past + future:
            # Shift the relevant columns back/forwards in time
            columns = [col for col in INPUTS if not col.startswith('P')]
            shifted = gait_cycle[columns].shift(offset)

            # Rename shifted columns and concat to gait_cycle
            timestep = offset * -2 # ms
            shifted.columns = [f'{col}_{timestep}ms' for col in columns]
            gait_cycle = pd.concat([gait_cycle, shifted], axis=1)

        return gait_cycle

    # Removes the first and final (25) rows for which no target label is available
    def remove_buffers(df):
        cond = df.iloc[:, 0:8].sum(axis=1) != 0
        first = df[cond].index[0]
        last = df[cond].index[-1] + 1
        return df[first:last]

    # Reads a single csv-file, expands the features and removes the buffers
    def read_gait_cycle(filepath):
        gait_cycle = pd.read_csv(filepath, header=0, names=HEADER)
        gait_cycle = expand_timeframe(gait_cycle)
        return remove_buffers(gait_cycle)


    # Build up DataFrame of gait cyles by traversing the specified TRIALS
    df = DataFrame()

    for subdirs in itertools.product(subjects, scenes):
        path = DATA_DIR + '/'.join(subdirs)

        for file in os.listdir(path):
            if file.endswith('.csv') and (file.startswith(trials) or trials == ('all')):
                print("Reading", file)
                filepath = path + '/' + file
                gait_cycle = read_gait_cycle(filepath)
                gait_cycle['trial'] = file
                df = pd.concat([df, gait_cycle])

    return df


##############################################
# LEFT VS. RIGHT FOOT -> MAIN VS. OTHER FOOT #
##############################################
# Removes the distinction between left and right foot by creating a new DataFrame in terms of main_foot and other_foot
def homogenize(df):
    def rename_columns(df, main_foot, other_foot):
        mapping = {col: col.replace("_" + main_foot, "")
                           .replace("_" + other_foot, "_o")
                   for col in df.columns}

        return df.rename(columns=mapping)

    # Create the new DataFrame
    df_l = rename_columns(df, 'l', 'r')
    df_r = rename_columns(df, 'r', 'l')
    df_combined = pd.concat([df_l, df_r])

    # Drop the target labels for other_foot
    df_combined = df_combined.drop(columns=['Fx_o','Fy_o','Fz_o', 'M_o'])
    return df_combined


#############
# FILTERING #
#############
def filter(df):
    # Ignore readings of force plate 1 (loose)
    df = df[df['fp1'] != 1]

    # # Only keep masked rows
    df = df[df['valid_mask_feet'] == 1]
    # df = df[df['correct_mask_fp'] == 1] #ignore this mask for now
    df = df[df['correct_mask_ins'] == 1]

    # Removes rows for which no target label is available
    df = df[(df['Ftot'] < 0.05) | (df['Fz'] != 0)]

    # Remove rows with non-available values
    df = df.dropna()

    return df


############################
# FEATURE/LABEL EXTRACTION #
############################
def extract_features(df):
    # Drop labels
    X = df.drop(columns=['Fx','Fy','Fz', 'M'], axis=1)

    # Drop pressure columns
    P_cols = [col for col in X if col.startswith('P')]
    X = X.drop(columns=P_cols, axis=1)

    # Drop masks
    X = X.drop(columns=MASKS, axis=1)

    # Drop source file
    X = X.drop(columns=['trial'], axis=1)

    return X


################################
# PRINCIPAL COMPONENT ANALYSIS #
################################
def perform_pca(X_train, X_test):
    # Fit PCA model to the training data for capturing 99% of the variance
    pca = PCA(n_components=0.99, svd_solver='full')
    pca.fit(X_train)

    # Save the PCA model to a file
    joblib.dump(pca, '../results/pca.pkl')

    # Project the data from the old features to their principal components
    print('Number of features before PCA', X_train.shape[1])

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    print('Number of features after PCA', X_train.shape[1])

    return X_train, X_test


########################
# EVALUATING THE MODEL #
########################
def print_metrics(y_test, y_pred):
    print('Performance on the test set:')

    # NRMSE
    range = torch.max(y_test) - torch.min(y_test)
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


