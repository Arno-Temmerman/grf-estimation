import argparse

from data_processing import *
from models.mlp import MLP
from model_evaluation import *


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="latest",)
parser.add_argument("--data", type=str, default="./segmented_data", help="Root directory of the data")
parser.add_argument("--subjects", type=str, default=["HH"], nargs="+")
parser.add_argument("--scenes", type=str, default=['FlatWalkStraight', 'FlatWalkCircular', 'FlatWalkStatic'], nargs="+")
parser.add_argument("--trials", type=str, default=('all'), nargs="+")
parser.add_argument("--include-emg", action="store_true", default=False, help="Include data measured by sEMG sensors in features.")


if __name__ == '__main__':
    args = parser.parse_args()

    # Load Data
    gait_cycles = read_gait_cycles(data_dir=args.data,
                                   subjects=args.subjects,
                                   scenes=args.scenes,
                                   trials=args.trials,
                                   drop_emgs=not args.include_emg)
    df_filtered = dp.filter_together(gait_cycles)
    
    # Feature/Label Selection
    X_test = dp.extract_features(df_filtered)
    Y_test = df_filtered[['Fx_l', 'Fy_l', 'Fz_l', 'Tz_l',
                          'Fx_r', 'Fy_r', 'Fz_r', 'Tz_r']]
    Y_test = tensor(Y_test.to_numpy().reshape((-1, 8)), dtype=torch.float32)


    # Load Model
    DIR = os.path.join("results", "insoles", "all", args.model)
    scaler = joblib.load(os.path.join(DIR, 'standard_scaler.pkl'))
    pca    = joblib.load(os.path.join(DIR, 'pca.pkl'))
    full_grf_estimator = MLP.load(DIR, 'mlp_Y')

    # Make Predictions
    X_scaled = scaler.transform(X_test.values)
    X_pc = pca.transform(X_scaled)
    X_pc_tensor = tensor(X_pc, dtype=torch.float32)
    Y_pred = full_grf_estimator(X_pc_tensor)

    # Show Results
    LABELS =  ['Fx_l', 'Fy_l', 'Fz_l', 'Tz_l', 'Fx_r', 'Fy_r', 'Fz_r', 'Tz_r']
    nr_of_vars = Y_test.shape[1]
    for i in range(nr_of_vars):
        print(LABELS[i])
        print_metrics(Y_test[:, i].reshape(-1, 1), Y_pred[:, i].reshape(-1, 1), scatterplot=False)
    plot_correlations(Y_test, Y_pred)