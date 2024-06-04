import joblib
import torch
import torch.nn as nn
import data_processing as dp
from torch import tensor

from models.mlp import MLP


class StackedRegressor(nn.Module):
    def __init__(self, DIR):
        super(StackedRegressor, self).__init__()

        # Load scaler and PCA model
        self.scaler = joblib.load(f'{DIR}/scaler.pkl')
        self.pca    = joblib.load(f'{DIR}/PCA.pkl')

        # Load base model of each component
        self.Fx = MLP.load(DIR, 'Fx_base')
        self.Fy = MLP.load(DIR, 'Fy_base')
        self.Fz = MLP.load(DIR, 'Fz_base')
        self.Tz = MLP.load(DIR, 'Tz_base')

        # Load meta model of each component
        self.Fx = MLP.load(DIR, 'Fx_meta')
        self.Fy = MLP.load(DIR, 'Fy_meta')
        self.Fz = MLP.load(DIR, 'Fz_meta')
        self.Tz = MLP.load(DIR, 'Tz_meta')


    def forward(self, X):
        # Homogenize features
        X_l, X_r = dp.homogenize(X)

        # Normalize features
        X_l = self.scaler.transform(X_l.values)
        X_r = self.scaler.transform(X_r.values)

        # Perform PCA
        X_l_pc = self.pca.transform(X_l)
        X_r_pc = self.pca.transform(X_r)

        # Convert to tensors
        X_pc_l_tensor = tensor(X_l_pc, dtype=torch.float32)
        X_pc_r_tensor = tensor(X_r_pc, dtype=torch.float32)


        ###################
        # BASE REGRESSORS #
        ###################
        # Make initial predictions for left foot
        Fx_l = self.Fx(X_pc_l_tensor)
        Fy_l = self.Fy(X_pc_l_tensor)
        Fz_l = self.Fz(X_pc_l_tensor)
        Tz_l = self.Tz(X_pc_l_tensor)

        # Make initial predictions for right foot
        Fx_r = self.Fx(X_pc_r_tensor)
        Fy_r = self.Fy(X_pc_r_tensor)
        Fz_r = self.Fz(X_pc_r_tensor)
        Tz_r = self.Tz(X_pc_r_tensor)


        ###############
        # META MODELS #
        ###############
        # Add inital predictions to input
        X_pc_l_tensor = torch.cat((X_pc_l_tensor, Fx_l, Fy_l, Fz_l, Tz_l, Fx_r, Fy_r, Fz_r, Tz_r), dim=1)
        X_pc_r_tensor = torch.cat((X_pc_r_tensor, Fx_r, Fy_r, Fz_r, Tz_r, Fx_l, Fy_l, Fz_l, Tz_l), dim=1)

        # Make final predictions with meta models
        Fx_l = self.Fx_meta(X_pc_l_tensor)
        Fy_l = self.Fy_meta(X_pc_l_tensor)
        Fz_l = self.Fz_meta(X_pc_l_tensor)
        Tz_l  = self.M_meta(X_pc_l_tensor)

        Fx_r = self.Fx_meta(X_pc_r_tensor)
        Fy_r = self.Fy_meta(X_pc_r_tensor)
        Fz_r = self.Fz_meta(X_pc_r_tensor)
        Tz_r  = self.M_meta(X_pc_r_tensor)

        Y = torch.cat([Fx_l, Fy_l, Fz_l, Tz_l,
                              Fx_r, Fy_r, Fz_r, Tz_r], dim=1)

        return Y