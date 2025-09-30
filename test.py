import argparse

from data_processing import *
from models.mlp import MLP
from models.stm_regressor import STMRegressor
from model_evaluation import *


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-me", "--method", type=str, default="single_target", help="The multi-output regression method used in the architecture. Choose between single_target and multi_task.")
parser.add_argument("-mo", "--model", type=str, default="latest", help="Name of the directory of the model to be tested (format: YYYYMMDD-HHMMSS-[inter|intra]_subject-insoles[_emg])")
parser.add_argument("-d", "--data", type=str, default="./segmented_data", help="Root directory of the data")
parser.add_argument("-su", "--subjects", type=str, default=["HH"], nargs="+", help="List of strings specifying the subject data directories to use for training. Default is [\"AT\", \"EL\", \"MS\", \"RB\", \"RL\", \"TT\"].")
parser.add_argument("-sc", "--scenes", type=str, default=["FlatWalkStraight", "FlatWalkCircular", "FlatWalkStatic"], nargs="+", help="List of strings specifying the scenes to use for training.")
parser.add_argument("-t", "--trials", type=str, default=["all"], nargs="+", help="Tuple of strings specifying the trials to use for training.")
parser.add_argument("-emg", "--include-emg", action="store_true", default=False, help="Flag used to in-/exclude data measured by sEMG sensors in feature selection.")


LABELS = ['Fx_l', 'Fy_l', 'Fz_l', 'Tz_l',
          'Fx_r', 'Fy_r', 'Fz_r', 'Tz_r']


if __name__ == '__main__':
    args = parser.parse_args()

    # Load Data
    gait_cycles = read_gait_cycles(data_dir=args.data,
                                   subjects=args.subjects,
                                   scenes=args.scenes,
                                   trials=args.trials[0] if len(args.trials) == 1 else tuple(args.trials),
                                   drop_emgs=not args.include_emg)
    
    df_filtered = dp.filter_together(gait_cycles)
    
    # Feature/Label Selection
    X_test = dp.extract_features(df_filtered)
    Y_test = df_filtered[LABELS]
    Y_test = tensor(Y_test.to_numpy().reshape((-1, 8)), dtype=torch.float32)


    # Load Model
    if args.model == "latest":
        DIR = os.path.join("results", args.method)
        DIR = os.path.join(DIR, sorted(os.listdir(DIR))[-1])
    else:
        DIR = os.path.join("results", args.method, args.model)

    if args.method == "single_target":
        full_grf_estimator = STMRegressor(DIR)
        Y_pred = full_grf_estimator(X_test)
    else:
        scaler = joblib.load(os.path.join(DIR, 'standard_scaler.pkl'))
        pca    = joblib.load(os.path.join(DIR, 'pca.pkl'))
        full_grf_estimator = MLP.load(DIR, 'mlp_Y')

        # Make Predictions
        X_scaled = scaler.transform(X_test.values)
        X_pc = pca.transform(X_scaled)
        X_pc_tensor = tensor(X_pc, dtype=torch.float32)
        Y_pred = full_grf_estimator(X_pc_tensor)

    # Show Results
    nr_of_vars = Y_test.shape[1]
    for i in range(nr_of_vars):
        print(LABELS[i])
        log_metrics(Y_test[:, i].reshape(-1, 1), Y_pred[:, i].reshape(-1, 1), scatterplot=False, log_file=os.path.join(DIR, "results.txt"))
    plot_correlations(Y_test, Y_pred, DIR)