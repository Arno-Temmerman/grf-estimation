import joblib
import torch
import torch.nn as nn

from models.mlp import MLP
from torch import tensor

class STMRegressor(nn.Module):
    def __init__(self, DIR):
        super(STMRegressor, self).__init__()

        # Load scaler and PCA model
        self.scaler = joblib.load(f'{DIR}/scaler.pkl')
        self.pca = joblib.load(f'{DIR}/PCA.pkl')

        # Load submodel of each component
        self.Fx = MLP.load(DIR, 'Fx')
        self.Fy = MLP.load(DIR, 'Fy')
        self.Fz = MLP.load(DIR, 'Fz')
        self.Tz = MLP.load(DIR, 'Tz')


    def forward(self, input_features):
        def rename_columns(df, main_foot, other_foot):
            mapping = {col: col.replace("_" + main_foot, "")
                               .replace("_" + other_foot, "_o")
                       for col in df.columns}

            return df.rename(columns=mapping)

        # Split input_features by left and right foot
        X_l = rename_columns(input_features, 'l', 'r')
        X_r = rename_columns(input_features, 'r', 'l')
        X_r = X_r[X_l.columns]

        # Normalize features
        X_l = self.scaler.transform(X_l.values)
        X_r = self.scaler.transform(X_r.values)

        # Perform PCA
        X_l_pc = self.pca.transform(X_l)
        X_r_pc = self.pca.transform(X_r)

        # Convert to tensors
        X_pc_l_tensor = tensor(X_l_pc, dtype=torch.float32)
        X_pc_r_tensor = tensor(X_r_pc, dtype=torch.float32)


        # Make predictions for left foot
        Fx_l = self.Fx(X_pc_l_tensor)
        Fy_l = self.Fy(X_pc_l_tensor)
        Fz_l = self.Fz(X_pc_l_tensor)
        Tz_l = self.Tz(X_pc_l_tensor)

        # Make predictions for right foot
        Fx_r = self.Fx(X_pc_r_tensor)
        Fy_r = self.Fy(X_pc_r_tensor)
        Fz_r = self.Fz(X_pc_r_tensor)
        Tz_r = self.Tz(X_pc_r_tensor)

        Y = torch.cat([Fx_l, Fy_l, Fz_l, Tz_l,
                              Fx_r, Fy_r, Fz_r, Tz_r], dim=1)

        return Y


