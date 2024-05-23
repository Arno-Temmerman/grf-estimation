import torch
import torch.nn as nn

class NRMSELoss(nn.Module):
    def __init__(self, norm_factor):
        super().__init__()
        self.mse = nn.MSELoss()
        self.norm_factor = norm_factor

    def forward(self, y, y_pred):
        rmse = torch.sqrt(self.mse(y, y_pred))
        return rmse / self.norm_factor