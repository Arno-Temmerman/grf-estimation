import itertools
import joblib
import os
import pandas as pd
import torch
from pandas import DataFrame
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import tensor


#######################
# BUILDING THE HEADER #
#######################
LABELS = [
    # Kistler force plates
    'Fx_l', 'Fy_l', 'Fz_l', 'Tz_l',
    'Fx_r', 'Fy_r', 'Fz_r', 'Tz_r'
]
INSOLES = [
    # Moticon insoles
    'Ftot_l', 'CoPx_l', 'CoPy_l',
    'Ftot_r', 'CoPx_r', 'CoPy_r',

    'P1_l', 'P2_l', 'P3_l', 'P4_l', 'P5_l', 'P6_l', 'P7_l', 'P8_l', 'P9_l', 'P10_l', 'P11_l', 'P12_l', 'P13_l', 'P14_l', 'P15_l', 'P16_l',
    'ax_l', 'ay_l', 'az_l',
    'angx_l', 'angy_l', 'angz_l',

    'P1_r', 'P2_r', 'P3_r', 'P4_r', 'P5_r', 'P6_r', 'P7_r', 'P8_r', 'P9_r', 'P10_r', 'P11_r', 'P12_r', 'P13_r', 'P14_r', 'P15_r', 'P16_r',
    'ax_r', 'ay_r', 'az_r',
    'angx_r', 'angy_r', 'angz_r']
EMGS = [
    # Cometa EMG sensors
    'vm_r',  'vm_l',  # vastus medialis
    'vr_r',  'vr_l',  # vastus radialis
    'gm_r',  'gm_l',  # gluteus medius
    'tfl_r', 'tfl_l' # tensor fasciae latae
]
MASKS = [
    'fp1_l', 'fp1_r', 'fp2_l', 'fp2_r', 'fp3_l', 'fp3_r', 'fp4_l', 'fp4_r',
    'valid_mask_feet', # 0 = occluded markers
    'correct_mask_fp', # 0 = force plate gave bad prediction
    'correct_mask_ins' # 0 = foot incorrectly placed on force plate
]
HEADER = LABELS + INSOLES + EMGS + MASKS

####################
# LOADING THE DATA #
####################
# Reads the csv-files in ./segmented_data of the subjects-scenes-trials combinations specified above
def read_gait_cycles(data_dir, subjects, scenes, trials, drop_emgs=False):
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
            timestep = offset * -2 # cs
            shifted.columns = [f'{col}_{timestep}cs' for col in columns]
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
        if drop_emgs:
            gait_cycle = gait_cycle.drop(columns=EMGS)
        gait_cycle = expand_timeframe(gait_cycle)
        return remove_buffers(gait_cycle)

    if drop_emgs: INPUTS = INSOLES
    else:         INPUTS = INSOLES + EMGS

    # Build up list of DataFrames of gait cyles by traversing the specified TRIALS
    dfs = []

    print(f'Reading {trials} trials of {subjects}.')

    for subdirs in itertools.product(subjects, scenes):
        path = data_dir + '/'.join(subdirs)

        for file in os.listdir(path):
            if file.endswith('.csv') and (file.startswith(trials) or trials == ('all')):
                filepath = path + '/' + file
                gait_cycle = read_gait_cycle(filepath)
                gait_cycle['subject'] = subdirs[0]
                gait_cycle['trial'] = file
                dfs.append(gait_cycle)

    return dfs

#############
# FILTERING #
#############
def filter_seperately(gait_cycles):
    def filter(gait_cycles, foot):
        df = DataFrame()

        # Trim each gait cycle
        for gait_cycle in gait_cycles:
            active = gait_cycle[gait_cycle[f'Fz_{foot}'] > 0]

            if not active.empty:
                first = max(gait_cycle.index[0],  active.index.min() - 15)
                last  = min(gait_cycle.index[-1], active.index.max() + 15)
                trimmed = gait_cycle[(first <= gait_cycle.index) & (gait_cycle.index <= last)]

                # Drop other invalid rows
                trimmed = trimmed[trimmed[f'fp1_{foot}'] != 1]      # invalid fp
                trimmed = trimmed[trimmed['valid_mask_feet'] == 1]  # occluded markers
                trimmed = trimmed[trimmed['correct_mask_ins'] == 1] # invalid foot placement

                # Check if trial has been synchronized poorly
                invalid_steps = trimmed[(trimmed[f'Fz_{foot}'] == 0) & (trimmed[f'Ftot_{foot}'] >= 0.40)]

                # Only add properly synchronized trials
                if invalid_steps.empty:
                    df = pd.concat([df, trimmed])
                else:
                    trial = trimmed.iloc[0]['trial']
                    subject = trimmed.iloc[0]['subject']
                    print(f'Dropped {trial} of subject {subject}.')

        # Drop samples with missing timestamp(s)
        df = df.dropna()

        return df

    # Trim each foot seperately
    df_l = filter(gait_cycles, 'l')
    df_r = filter(gait_cycles, 'r')

    return df_l, df_r

def filter_together(gait_cycles):
    df = DataFrame()

    # Trim each gait cycle
    for gait_cycle in gait_cycles:
        active_l = gait_cycle[gait_cycle[f'Fz_l'] > 0]
        active_r = gait_cycle[gait_cycle[f'Fz_r'] > 0]

        if not (active_l.empty or active_r.empty):
            first = max(active_l.index.min() - 15, active_r.index.min() - 15)
            last  = min(active_l.index.max() + 15, active_r.index.max() + 15)
            trimmed = gait_cycle[(first <= gait_cycle.index) & (gait_cycle.index <= last)]

            # Drop other invalid rows
            trimmed = trimmed[(trimmed['fp1_l'] != 1) & (trimmed['fp1_r'] != 1)]  # invalid fp
            trimmed = trimmed[trimmed['valid_mask_feet'] == 1]                    # occluded markers
            trimmed = trimmed[trimmed['correct_mask_ins'] == 1]                   # invalid foot placement

            # Check if trial has been synchronized poorly
            invalid_steps = trimmed[  ((trimmed[f'Fz_l'] == 0) & (trimmed[f'Ftot_l'] >= 0.40))
                                    | ((trimmed[f'Fz_r'] == 0) & (trimmed[f'Ftot_r'] >= 0.40))]

            # Only add properly synchronized trials
            if invalid_steps.empty:
                df = pd.concat([df, trimmed])
            else:
                trial = trimmed.iloc[0]['trial']
                subject = trimmed.iloc[0]['subject']
                print(f'Dropped {trial} of subject {subject}.')

            df = pd.concat([df, trimmed])

    # Drop samples with missing timestamp(s)
    df = df.dropna()

    return df


##############################################
# LEFT VS. RIGHT FOOT -> MAIN VS. OTHER FOOT #
##############################################
def reformat(df, main_foot, other_foot):
    mapping = {col: col.replace("_" + main_foot, "")
                       .replace("_" + other_foot, "_o")
               for col in df.columns}
    return df.rename(columns=mapping)

# Removes the distinction between left and right foot by creating a new DataFrame in terms of main_foot and other_foot
def homogenize(df_l, df_r=None):
    # Single parameter version
    if df_r is None:
        df = df_l
        df_l = reformat(df, 'l', 'r')
        df_r = reformat(df, 'r', 'l')
    # Two parameter version
    else:
        df_l = reformat(df_l, 'l', 'r')
        df_r = reformat(df_r, 'r', 'l')

    # Use same order of columns for both
    df_r = df_r[df_l.columns]
    return df_l, df_r


############################
# FEATURE/LABEL EXTRACTION #
############################
def extract_features(df):
    # Drop labels
    labels = [col for col in df
              if col.startswith('Fx')
              or col.startswith('Fy')
              or col.startswith('Fz')
              or col.startswith('Tz')]
    X = df.drop(columns=labels)

    # Drop pressure columns
    P_cols = [col for col in X if col.startswith('P')]
    X = X.drop(columns=P_cols)

    # Drop pressure columns
    fp_cols = [col for col in X if col.startswith('fp')]
    X = X.drop(columns=fp_cols)

    # Drop other masks
    OTHER = ['valid_mask_feet', 'correct_mask_fp', 'correct_mask_ins']
    X = X.drop(columns=OTHER)

    # Drop source file
    X = X.drop(columns=['trial'])

    # Drop subject
    X = X.drop(columns=['subject'])

    return X


################################
# PRINCIPAL COMPONENT ANALYSIS #
################################
def perform_pca(X_train, X_test, save_dir=None):
    # Normalize features so their variances are comparable
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save the scaler to a file
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir, 'scaler.pkl'), 'wb') as output_file:
            joblib.dump(scaler, output_file)


    # Fit PCA model to the training data for capturing 99% of the variance
    pca = PCA(n_components=0.99, svd_solver='full')
    pca.fit(X_train)

    # Project the data from the old features to their principal components
    X_pc_train = pca.transform(X_train)
    X_pc_test  = pca.transform(X_test)

    # Save the PCA model to a file
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir, 'PCA.pkl'), 'wb') as output_file:
            joblib.dump(pca, output_file)


    # Convert the result to tensor
    X_train_tensor = tensor(X_pc_train, dtype=torch.float32)
    X_test_tensor  = tensor(X_pc_test,  dtype=torch.float32)

    return X_train_tensor, X_test_tensor
