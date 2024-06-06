import data_processing as dp
import joblib
import model_evaluation as me
import numpy as np
import optuna
import pandas as pd
import time
import torch
from models.mlp import MLP
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch import tensor


EXCL_EMG = True

if EXCL_EMG: FEATURES = 'excl_emg'
else:        FEATURES = 'incl_emg'

####################
# LOADING THE DATA #
####################
DATA_DIR = "../segmented_data/"
if EXCL_EMG: SUBJECTS = ['AT', 'EL', 'MS', 'RB', 'RL', 'TT']
else:        SUBJECTS = ['AT', 'MS', 'RB', 'RL', 'TT']
SCENES = ['FlatWalkStraight', 'FlatWalkCircular', 'FlatWalkStatic']
TRIALS = ('all')

gait_cycles = dp.read_gait_cycles(DATA_DIR, SUBJECTS, SCENES, TRIALS, drop_emgs=EXCL_EMG)


############################
# TRIMMING THE GAIT CYCLES #
############################
df = dp.filter_together(gait_cycles)

del gait_cycles


##############################################
# LEFT VS. RIGHT FOOT -> MAIN VS. OTHER FOOT #
##############################################
df_mirror = dp.reformat(df, 'r', 'l')

df_l, df_r = dp.homogenize(df)
df_augmented = pd.concat([df_l, df_r])

del df, df_l, df_r


############################
# FEATURE/LABEL EXTRACTION #
############################
# Features
X = dp.extract_features(df_augmented)

# Labels
LABELS = ['Fx', 'Fy', 'Fz', 'Tz',
          'Fx_o', 'Fy_o', 'Fz_o', 'Tz_o',]
Y = df_augmented[LABELS]

# Subjects
subjects = df_augmented['subject']

del df_augmented


######################
# CONVERT TO TENSORS #
######################
X_tensor = torch.tensor(X.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.to_numpy().reshape((-1, 8)), dtype=torch.float32)

del X, Y


# ##########################
# # CROSS-VALIDATION SPLIT #
# ##########################
K = len(SUBJECTS)
kf = LeaveOneGroupOut()


# #########################
# # HYPERPARAMETER TUNING #
# #########################
# 1. Define an objective function to be maximized
def objective(trial):
    # 2. Suggest values for the hyperparameters using a trial object
    ## a. Number of layers
    n_layers = trial.suggest_int('hidden_layers', 1, 2)

    ## b. Number of neurons per layer
    hidden_sizes = []
    for i in range(n_layers):
        size = trial.suggest_int(f'hidden_size_{i}', 32 * 8, 64 * 8)
        hidden_sizes.append(size)

    # 3. Instantiate a model with suggested hyperparameters
    model = MLP(hidden_sizes)
    trial.set_user_attr('hidden_sizes', hidden_sizes)

    # 4. Cross-validate the suggested model
    mses, corrs = me.cross_validate(model, X_tensor, Y_tensor.reshape(-1, 8), subjects, kf)
    print('Out-of-sample errors:', mses)
    print('Correlations:', corrs)
    return np.mean(mses)


# 5. Create a study object and optimize the objective function.
study = optuna.create_study(study_name='Y', direction='minimize')
study.set_user_attr('best_score', float('inf'))
study.optimize(objective, n_trials=50)

# Remember the best model
best_hyperparams = study.best_trial.user_attrs['hidden_sizes']


######################
# PERSIST BEST MODEL #
######################
DIR = f'results/{FEATURES}/{TRIALS}/' + time.strftime("%Y%m%d-%H%M%S/", time.localtime())


# PERFORM PCA #
# Normalize features so their variances are comparable
scaler = StandardScaler()
scaler.fit(X_tensor)
X_scaled = scaler.transform(X_tensor)

# Fit PCA model to the training data for capturing 99% of the variance
pca = PCA(n_components=0.99, svd_solver='full')
pca.fit(X_scaled)
X_pc = pca.transform(X_scaled)
X_pc_tensor = tensor(X_pc, dtype=torch.float32)

# Persist both models
Path(DIR).mkdir(parents=True, exist_ok=True)
# Save the scaler to a file
with open(Path(DIR, 'scaler.pkl'), 'wb') as output_file:
    joblib.dump(scaler, output_file)

# Save the PCA model to a file
with open(Path(DIR, 'PCA.pkl'), 'wb') as output_file:
    joblib.dump(pca, output_file)

del X_tensor, scaler, X_scaled, pca, X_pc

# TRAIN AND SAVE THE  MODELS #
model = MLP(best_hyperparams)
model.train_(X_pc_tensor, Y_tensor.reshape(-1, 8))
model.save(DIR, 'Y')