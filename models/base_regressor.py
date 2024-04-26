from pathlib import Path

import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.NRMSELoss import NRMSELoss


class BaseRegressor(nn.Module):
    def __init__(self, component, input_size, hidden_sizes):
        super().__init__()

        OUTPUT_SIZE = 1

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.Sigmoid())
            prev_size = size

        layers.append(nn.Linear(prev_size, OUTPUT_SIZE))
        self.linear_sigmoid_stack = nn.Sequential(*layers)

        self.component = component
        self.timestamp = time.strftime("%Y%m%d/%H%M%S", time.localtime())

        self.writer = SummaryWriter(Path(f'../results/{self.component}/{self.timestamp}'))


    def forward(self, x):
        return self.linear_sigmoid_stack(x)


    def train_(self, X, y):
        self.train() # train mode

        # Instantiate the loss function
        norm_factor = (torch.max(y).item() - torch.min(y).item())
        loss_function = NRMSELoss(norm_factor)

        # Instantiate the optimizer
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)

        for epoch in range(100):
            # Don't accumulate gradients
            optimizer.zero_grad()

            # Compute the loss and its gradients
            y_pred = self(X)
            loss = loss_function(y, y_pred)
            loss.backward()  # back propagation

            # Adjust the NN's weights and biases accordingly
            optimizer.step()

            if epoch % 20 == 0:
                # Log the loss to TensorBoard
                self.writer.add_scalar(f'Training/NRMSE', loss.item(), epoch)
                # print(f'[epoch:{epoch}]: NRMSE = {loss}')

                # Log the weights and biases to TensorBoard
                for name, param in self.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu(), epoch)
                    self.writer.add_scalar(name + '/grad_norm', param.grad.data.norm(), epoch)


    def save(self):
        with open(Path(f'../results/{self.component}/{self.timestamp}/{self.component}.pt'), 'wb') as output_file:
            torch.save(self.state_dict(), output_file)