import joblib
import torch
import torch.nn as nn
import numpy as np

from models.base_regressor import BaseRegressor


class CombinedRegressor(nn.Module):
    def __init__(self):
        super(CombinedRegressor, self).__init__()

        DIR = '20240502-170624'

        def load_model(component):
            sizes = []

            state_dict = torch.load(f'../results/{DIR}/{component}.pt')
            for key, value in state_dict.items():
                if 'weight' in key:
                    sizes.append(np.shape(value)[1])  # Number of neurons in the layer

            input_size, *hidden_sizes = sizes
            model = BaseRegressor(input_size, hidden_sizes)
            model.load_state_dict(state_dict)
            return model


        # PCA
        self.pca = joblib.load(f'../results/{DIR}/PCA.pkl')

        # Load the base regressors
        self.Fx_base = load_model('Fx_base')
        self.Fy_base = load_model('Fy_base')
        self.Fz_base = load_model('Fz_base')
        self.M_base  = load_model('M_base')
        
        # Load the meta models
        self.Fx_meta = load_model('Fx_meta')
        self.Fy_meta = load_model('Fy_meta')
        self.Fz_meta = load_model('Fz_meta')
        self.M_meta  = load_model('M_meta')


    def forward(self, input_features):
        def rename_columns(df, main_foot, other_foot):
            mapping = {col: col.replace("_" + main_foot, "")
                               .replace("_" + other_foot, "_o")
                       for col in df.columns}

            return df.rename(columns=mapping)

        # Split input_features by left and right foot
        X_l = rename_columns(input_features, 'l', 'r')
        X_r = rename_columns(input_features, 'r', 'l')
        X_r = X_r[self.pca.feature_names_in_]

        # Perform PCA
        X_l_pc = self.pca.transform(X_l)
        X_r_pc = self.pca.transform(X_r)

        # Convert to tensors
        X_pc_l_tensor = torch.tensor(X_l_pc, dtype=torch.float32)
        X_pc_r_tensor = torch.tensor(X_r_pc, dtype=torch.float32)


        ###################
        # BASE REGRESSORS #
        ###################
        # Make initial predictions with base regressors
        Fx_l = self.Fx_base(X_pc_l_tensor)
        Fy_l = self.Fy_base(X_pc_l_tensor)
        Fz_l = self.Fz_base(X_pc_l_tensor)
        M_l  = self.M_base(X_pc_l_tensor)

        Fx_r = self.Fx_base(X_pc_r_tensor)
        Fy_r = self.Fy_base(X_pc_r_tensor)
        Fz_r = self.Fz_base(X_pc_r_tensor)
        M_r  = self.M_base(X_pc_r_tensor)


        ###############
        # META MODELS #
        ###############
        # Add inital predictions to input
        X_pc_l_tensor = torch.cat((X_pc_l_tensor, Fx_l, Fy_l, Fz_l, M_l, Fx_r, Fy_r, Fz_r, M_r), dim=1)
        X_pc_r_tensor = torch.cat((X_pc_r_tensor, Fx_l, Fy_l, Fz_l, M_l, Fx_r, Fy_r, Fz_r, M_r), dim=1)

        # Make final predictions with meta models
        Fx_l = self.Fx_meta(X_pc_l_tensor)
        Fy_l = self.Fy_meta(X_pc_l_tensor)
        Fz_l = self.Fz_meta(X_pc_l_tensor)
        M_l  = self.M_meta(X_pc_l_tensor)

        Fx_r = self.Fx_meta(X_pc_r_tensor)
        Fy_r = self.Fy_meta(X_pc_r_tensor)
        Fz_r = self.Fz_meta(X_pc_r_tensor)
        M_r  = self.M_meta(X_pc_r_tensor)

        return Fx_l, Fy_l, Fz_l, M_l, Fx_r, Fy_r, Fz_r, M_r


