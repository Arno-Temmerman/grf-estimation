from pathlib import Path

import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class BaseRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, component):
        super().__init__()

        OUTPUT_SIZE = 1

        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, OUTPUT_SIZE),
        )

        self.component = component
        self.timestamp = time.strftime("%Y%m%d/%H%M%S", time.localtime())

        self.writer = SummaryWriter(Path(f'../results/{self.component}/{self.timestamp}'))


    def forward(self, x):
        return self.linear_tanh_stack(x)


    def train_(self, X, y):
        self.train() # train mode
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)

        for epoch in range(100):
            # Don't accumulate gradients
            optimizer.zero_grad()

            # Compute the loss and its gradients
            y_pred = self(X)
            loss = loss_function(y_pred, y)
            loss.backward()  # back propagation

            # Adjust the NN's weights and biases accordingly
            optimizer.step()

            # Log the loss to TensorBoard
            self.writer.add_scalar(f'Training/MSE', loss.item(), epoch)

            # Log the weights and biases to TensorBoard
            for name, param in self.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu(), epoch)
                self.writer.add_histogram(name + '/grad_norm', param.grad.data.norm(), epoch)

            if epoch % 10 == 0:
                print(f'[epoch:{epoch}]: MSE = {loss}')

    def save(self):
        with open(Path(f'../results/{self.component}/{self.timestamp}/{self.component}.pt'), 'wb') as output_file:
            torch.save(self.state_dict(), output_file)