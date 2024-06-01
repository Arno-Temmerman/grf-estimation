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


EXCL_EMG = False

if EXCL_EMG: FEATURES = 'excl_emg'
else:        FEATURES = 'incl_emg'

####################
# LOADING THE DATA #
####################
DATA_DIR = "../segmented_data/"
SUBJECTS = ['AT', 'EL', 'MS', 'RB', 'RL', 'TT']
SCENES = ['FlatWalkStraight', 'FlatWalkCircular', 'FlatWalkStatic']
TRIALS = ('all')

gait_cycles = dp.read_gait_cycles(DATA_DIR, SUBJECTS, SCENES, TRIALS, drop_emgs=(FEATURES == 'excl_emg'))

DIR = f'results/{FEATURES}/{TRIALS}/' + time.strftime("%Y%m%d-%H%M%S/", time.localtime())


############################
# TRIMMING THE GAIT CYCLES #
############################
df_l, df_r = dp.filter_seperately(gait_cycles)

del gait_cycles


##############################################
# LEFT VS. RIGHT FOOT -> MAIN VS. OTHER FOOT #
##############################################
df_l, df_r = dp.homogenize(df_l, df_r)
df_homogenous = pd.concat([df_l, df_r])

del df_l, df_r

############################
# FEATURE/LABEL EXTRACTION #
############################
# Features
X = dp.extract_features(df_homogenous)

# Labels
LABELS = ['Fx', 'Fy', 'Fz', 'Tz']
Y = df_homogenous[LABELS]

# Subjects
subjects = df_homogenous['subject']

del df_homogenous


######################
# CONVERT TO TENSORS #
######################
X_tensor = torch.tensor(X.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.to_numpy().reshape((-1, 4)), dtype=torch.float32)

del X, Y


##########################
# CROSS-VALIDATION SPLIT #
##########################
K = len(subjects.unique())
kf = LeaveOneGroupOut()


#########################
# HYPERPARAMETER TUNING #
#########################
for i, label in enumerate(LABELS):
    # 1. Define an objective function to be maximized
    def objective(trial):
        # 2. Suggest values for the hyperparameters using a trial object
        ## a. Number of layers
        n_layers = trial.suggest_int('hidden_layers', 1, 2)

        ## b. Number of neurons per layer
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f'hidden_size_{i}', 16, 64)
            hidden_sizes.append(size)

        # 3. Instantiate a model with suggested hyperparameters
        model = MLP(hidden_sizes)
        trial.set_user_attr('model', model)

        # 4. Cross-validate the suggested model
        mses, corrs = me.cross_validate(model, X_tensor, Y_tensor[:, i].reshape(-1, 1), subjects, kf)
        print('Out-of-sample errors:', mses)
        print('Correlations:', corrs)
        return np.mean(mses)


    # 5. Create a study object and optimize the objective function.
    study = optuna.create_study(study_name=label, direction='minimize')
    study.set_user_attr('best_score', float('inf'))
    study.optimize(objective, n_trials=1)


    ##########################
    # PERSIST THE BEST MODEL #
    ##########################
    best_model = study.best_trial.user_attrs['model']

    # PERFORM PCA #
    # Normalize features so their variances are comparable
    scaler = StandardScaler()
    scaler.fit(X_tensor)
    X_full_train_scaled = scaler.transform(X_tensor)

    # Save the scaler to a file
    Path(DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(DIR, 'scaler.pkl'), 'wb') as output_file:
        joblib.dump(scaler, output_file)

    # Fit PCA model to the training data for capturing 99% of the variance
    pca = PCA(n_components=0.99, svd_solver='full')
    pca.fit(X_full_train_scaled)

    # Project the data from the old features to their principal components
    X_pc_full_train = pca.transform(X_full_train_scaled)

    # Save the PCA model to a file
    Path(DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(DIR, 'PCA.pkl'), 'wb') as output_file:
        joblib.dump(pca, output_file)

    # Convert the result to tensor
    X_pc_full_train = tensor(X_pc_full_train, dtype=torch.float32)

    del scaler, X_full_train_scaled , pca

    best_model.train_(X_pc_full_train, Y_tensor[:, i].reshape(-1, 1))
    best_model.save(DIR, label)