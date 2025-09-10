import argparse
import time

from data_processing import *
from models.mlp import MLP


LABELS = ['Fx',   'Fy',   'Fz',   'Tz',
          'Fx_o', 'Fy_o', 'Fz_o', 'Tz_o',]


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--hidden-layers", type=int, default=[241], nargs="+")
parser.add_argument("--data", type=str, default="./segmented_data", help="Root directory of the data")
parser.add_argument("--subjects", type=str, default=["AT", "EL", "MS", "RB", "RL", "TT"], nargs="+")
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
    df = filter_together(gait_cycles)
    df_l, df_r = homogenize(df)
    df_augmented = pd.concat([df_l, df_r])

    # Feature/Label Selection
    X = extract_features(df_augmented)
    Y = df_augmented[LABELS]
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy().reshape((-1, 8)), dtype=torch.float32)

    # Normalize features so their variances are comparable
    scaler = StandardScaler()
    scaler.fit(X_tensor)
    X_scaled = scaler.transform(X_tensor)

    # Fit PCA model to the training data for capturing 99% of the variance
    pca = PCA(n_components=0.99, svd_solver='full')
    pca.fit(X_scaled)
    X_pc = pca.transform(X_scaled)
    X_pc_tensor = tensor(X_pc, dtype=torch.float32)

    # Train the model
    model = MLP(args.hidden_layers)
    model.train_(X_pc_tensor, Y_tensor.reshape(-1, 8))
   
    # Persist model
    FEATURES = "insoles_emg" if args.include_emg else "insoles"
    DIR = os.path.join("./results", FEATURES, args.trials, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    Path(DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(DIR, 'config.txt'), 'w') as output_file:
        for arg, value in vars(args).items():
            output_file.write(f"{arg}: {value}\n")
    with open(Path(DIR, 'standard_scaler.pkl'), 'wb') as output_file:
        joblib.dump(scaler, output_file)
    with open(Path(DIR, 'pca.pkl'), 'wb') as output_file:
        joblib.dump(pca, output_file)
    model.save(DIR, f"mlp_Y")
