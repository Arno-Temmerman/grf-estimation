import data_processing as dp
import model_evaluation as me
import numpy as np
import pandas as pd
import torch
from models.mlp import MLP
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

INTER_SUBJECT = True
EXCL_EMG = True

if EXCL_EMG: FEATURES = 'excl_emg'
else:        FEATURES = 'incl_emg'

####################
# LOADING THE DATA #
####################
DATA_DIR = "../segmented_data/"
if INTER_SUBJECT:
    if EXCL_EMG: SUBJECTS = [['AT', 'EL', 'MS', 'RB', 'RL', 'TT']]           # train one model on all training  subjects
    else:        SUBJECTS = [['AT', 'MS', 'RB', 'RL', 'TT']]                 # noisy EMG signals for 'EL'
else:            SUBJECTS = [['AT'], ['EL'], ['MS'], ['RB'], ['RL'], ['TT']] # train model for each subject individually
SCENES = ['FlatWalkStraight', 'FlatWalkCircular', 'FlatWalkStatic']
TRIALS = ('all')


for subject in SUBJECTS:
    gait_cycles = dp.read_gait_cycles(DATA_DIR, subject, SCENES, TRIALS, drop_emgs=EXCL_EMG)


    ############################
    # TRIMMING THE GAIT CYCLES #
    ############################
    df_l, df_r = dp.filter_seperately(gait_cycles)

    ##############################################
    # LEFT VS. RIGHT FOOT -> MAIN VS. OTHER FOOT #
    ##############################################
    df_l, df_r = dp.homogenize(df_l, df_r)
    df_homogenous = pd.concat([df_l, df_r])


    ############################
    # FEATURE/LABEL EXTRACTION #
    ############################
    # Features
    X = dp.extract_features(df_homogenous)

    # Labels
    LABELS = ['Fx', 'Fy', 'Fz', 'Tz']
    Y = df_homogenous[LABELS]

    # Strata
    if INTER_SUBJECT: strata = df_homogenous['subject']
    else:             strata = df_homogenous['trial']

    ######################
    # CONVERT TO TENSORS #
    ######################
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy().reshape((-1, 4)), dtype=torch.float32)


    ####################
    # CROSS VALIDATION #
    ####################
    if INTER_SUBJECT:
        K = len(subject)
        kf = LeaveOneGroupOut()
    else:
        K = 5
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    models = {
        'Fx': MLP([61]),
        'Fy': MLP([60]),
        'Fz': MLP([58]),
        'Tz': MLP([62]),
    }

    for i, (label, model) in enumerate(models.items()):
        y = Y_tensor[:, i].reshape(-1, 1)
        range = (torch.quantile(y, 0.99) - torch.quantile(y, 0.01)).item()
        mses, rs = me.cross_validate(model, X_tensor, y, strata, kf)
        nrmses = np.sqrt(mses) / range

        print(label)
        print(f'Errors: {np.mean(nrmses) * 100:.2f} ({np.std(nrmses) * 100:.2f})')
        print(f'Correlations: {np.mean(rs) * 100:.2f} ({np.std(rs) * 100:.2f})')
        if INTER_SUBJECT:
            print(nrmses)
            print(rs)


