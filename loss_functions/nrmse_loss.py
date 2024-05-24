import torch
import torch.nn as nn

class NRMSELoss(nn.Module):
    def __init__(self, norm_factors):
        super().__init__()
        self.mse = nn.MSELoss()
        self.norm_factors = norm_factors

    def forward(self, Y, Y_pred):
        loss = 0
        for i in range(Y.shape[1]):
            mse = self.mse(Y[:, i].reshape((-1, 1)), Y_pred[:, i].reshape((-1, 1)))
            nrmse = torch.sqrt(mse) / self.norm_factors[i]
            loss += nrmse
        return loss


