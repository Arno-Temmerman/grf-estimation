import argparse
import time

from data_processing import *
from models.mlp import MLP


LABELS = ["Fx",   "Fy",   "Fz",   "Tz",
          "Fx_o", "Fy_o", "Fz_o", "Tz_o",]


# Argument Parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--method", type=str, default="single_target", help="")
parser.add_argument("-Fx", "--hidden-layers-Fx", type=int, default=[61], nargs="+", help="List of integers specifying the number of neurons in each hidden layer for Fx MLP.")
parser.add_argument("-Fy", "--hidden-layers-Fy", type=int, default=[60], nargs="+", help="List of integers specifying the number of neurons in each hidden layer for Fy MLP.")
parser.add_argument("-Fz", "--hidden-layers-Fz", type=int, default=[58], nargs="+", help="List of integers specifying the number of neurons in each hidden layer for Fz MLP.")
parser.add_argument("-Tz", "--hidden-layers-Tz", type=int, default=[62], nargs="+", help="List of integers specifying the number of neurons in each hidden layer for Tz MLP.")
parser.add_argument("-Y", "--hidden-layers-Y", type=int, default=[241], nargs="+", help="List of integers specifying the number of neurons in each hidden layer for Y MLP.")
parser.add_argument("-d", "--data", type=str, default="segmented_data", help="Root directory of the data. Default is segmented_data.")
parser.add_argument("--subjects", type=str, default=["AT", "EL", "MS", "RB", "RL", "TT"], nargs="+", help="List of strings specifying the subject data directories to use for training. Default is [\"AT\", \"EL\", \"MS\", \"RB\", \"RL\", \"TT\"].")
parser.add_argument("--scenes", type=str, default=["FlatWalkStraight", "FlatWalkCircular", "FlatWalkStatic"], nargs="+", help="List of strings specifying the scenes to use for training.")
parser.add_argument("-t", "--trials", type=str, default=("all"), nargs="+", help="Tuple of strings specifying the trials to use for training.")
parser.add_argument("-emg", "--include-emg", action="store_true", default=False, help="Flag used to in-/exclude data measured by sEMG sensors in feature selection.")


if __name__ == "__main__":
    args = parser.parse_args()

    # Load Data
    gait_cycles = read_gait_cycles(data_dir=args.data, 
                                   subjects=args.subjects,
                                   scenes=args.scenes, 
                                   trials=args.trials,
                                   drop_emgs=not args.include_emg)
    match args.method:
        case "single_target":
            df_l, df_r = filter_seperately(gait_cycles)
            df_l, df_r = homogenize(df_l, df_r)
        case "multi_task":
            df = filter_together(gait_cycles)
            df_l, df_r = homogenize(df)
    df_augmented = pd.concat([df_l, df_r])

    # Feature/Label Selection
    X = extract_features(df_augmented)
    Y = df_augmented[LABELS]
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy().reshape((-1, 8)), dtype=torch.float32)

    # Make directory for model persistence
    SUBJECTS = "intra_subject" if len(args.subjects) == 1 else "inter_subject"
    FEATURES = "insoles_emg" if args.include_emg else "insoles"
    DIR = os.path.join("results", SUBJECTS, args.method, FEATURES, args.trials, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    Path(DIR).mkdir(parents=True, exist_ok=True)

    # Normalize features so their variances are comparable
    scaler = StandardScaler()
    scaler.fit(X_tensor)
    with open(Path(DIR, "standard_scaler.pkl"), "wb") as output_file:
        joblib.dump(scaler, output_file)
    X_scaled = scaler.transform(X_tensor)

    # Fit PCA model to the training data for capturing 99% of the variance
    pca = PCA(n_components=0.99, svd_solver="full")
    pca.fit(X_scaled)
    with open(Path(DIR, "pca.pkl"), "wb") as output_file:
        joblib.dump(pca, output_file)
    X_pc = pca.transform(X_scaled)
    X_pc_tensor = tensor(X_pc, dtype=torch.float32)


    # Train the model
    match args.method:
        case "single_target":
            hidden_layers = {
                "Fx": args.hidden_layers_Fx,
                "Fy": args.hidden_layers_Fy,
                "Fz": args.hidden_layers_Fz,
                "Tz": args.hidden_layers_Tz,
            }
            for i, (label, hidden_sizes) in enumerate(hidden_layers.items()):
                model = MLP(hidden_sizes)
                model.train_(X_pc_tensor, Y_tensor[:, i].reshape(-1, 1))
                model.save(DIR, label)

        case "multi_task":
            model = MLP(args.hidden_layers_Y)
            model.train_(X_pc_tensor, Y_tensor.reshape(-1, 8))
            model.save(DIR, f"mlp_Y")

    with open(Path(DIR, "config.txt"), "w") as output_file:
        for arg, value in vars(args).items():
            output_file.write(f"{arg}: {value}\n")
