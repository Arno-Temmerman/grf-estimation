import data_processing as dp
import model_evaluation as me
import numpy as np
import optuna
import pandas as pd
import time
import torch
from models.mlp import MLP
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, GroupShuffleSplit

from models.stm_regressor import STMRegressor

META1 = ['incl_emg', 'excl_emg']

for FEATURES in META1:
    ####################
    # LOADING THE DATA #
    ####################
    SUBJECTS = ['AT', 'EL', 'HH', 'MS', 'RB', 'RL', 'TT']
    SCENES = ['FlatWalkStraight', 'FlatWalkCircular', 'FlatWalkStatic']
    #TRIALS = ('FW walking')
    TRIALS = ('all')

    DIR = f'results/{FEATURES}/{TRIALS}/' + time.strftime("%Y%m%d-%H%M%S/", time.localtime())


    gait_cycles = dp.read_gait_cycles(SUBJECTS, SCENES, TRIALS, drop_emgs=(FEATURES == 'excl_emg'))


    ##############################################
    # LEFT VS. RIGHT FOOT -> MAIN VS. OTHER FOOT #
    ##############################################
    df_l, df_r = dp.homogenize(gait_cycles)
    df_homogenous = pd.concat([df_l, df_r])


    #############
    # FILTERING #
    #############
    df_filtered = dp.filter(df_homogenous)


    ############################
    # FEATURE/LABEL EXTRACTION #
    ############################
    # Features
    X = dp.extract_features(df_filtered)

    # Labels
    LABELS = ['Fx', 'Fy', 'Fz', 'Tz', 'Fx_o', 'Fy_o', 'Fz_o', 'Tz_o' ]
    Y = df_filtered[LABELS]

    # Groups
    subjects = df_filtered['subject']


    ######################
    # CONVERT TO TENSORS #
    ######################
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy().reshape((-1, 8)), dtype=torch.float32)


    ####################
    # TRAIN-TEST SPLIT #
    ####################
    train_idx = np.where(subjects != 'HH')[0]
    test_idx  = np.where(subjects == 'HH')[0]

    X_full_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    Y_full_train, Y_test = Y_tensor[train_idx], Y_tensor[test_idx]
    subjects_full_train, subjects_test = subjects[subjects != 'HH'], subjects[subjects == 'HH']


    K = len(subjects_full_train.unique())
    kf = GroupKFold(n_splits=K)


    ###############
    # PERFORM PCA #
    ###############
    X_pc_full_train, X_pc_test = dp.perform_pca(X_full_train, X_test, DIR)


    #########################
    # HYPERPARAMETER TUNING #
    #########################
    # 1. Define an objective function to be maximized
    def objective(trial):
        # 2. Suggest values for the hyperparameters using a trial object
        ## a. Number of layers
        n_layers = trial.suggest_int('hidden_layers', 2, 2)

        ## b. Number of neurons per layer
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f'hidden_size_{i}', 55, 55)
            hidden_sizes.append(size)

        # 3. Instantiate a model with suggested hyperparameters
        model = MLP(hidden_sizes)
        trial.set_user_attr('model', model)

        # 4. Cross-validate the suggested model
        scores = me.cross_validate(model, X_full_train, Y_full_train, subjects_full_train, kf, pca=True)
        print('Out-of-sample errors:', scores)
        return np.mean(scores)


    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(study_name='Y', direction='minimize')
    study.set_user_attr('best_score', float('inf'))
    study.optimize(objective, n_trials=1)

    best_model = study.best_trial.user_attrs['model']
    best_model.train_(X_pc_full_train, Y_full_train)
    best_model.save(DIR, 'Y')