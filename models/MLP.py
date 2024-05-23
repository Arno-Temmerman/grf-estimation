from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import Linear, Sigmoid, MSELoss

from models.NRMSELoss import NRMSELoss


class MLP(nn.Module):
    """Generic implementation of a multiplayer perceptron (MLP) with sigmoid activation functions

    Args:
        hidden_sizes (list): list of #neurons in the hidden layer(s).
    """

    def __init__(self, hidden_sizes):
        super().__init__()
        self.hidden_sizes = hidden_sizes

        # Instantiate hidden layers with sigmoid activations
        hidden_layers = [Sigmoid()]
        for i in range(len(hidden_sizes) - 1):
            hidden_layers.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            hidden_layers.append(Sigmoid())

        self.__hidden_stack = nn.Sequential(*hidden_layers)

    def forward(self, X):
        '''Passes input features forward through the MLP's layers

        Args:
            X (torch.Tensor): input features

        Returns:
            torch.Tensor: output of the MLP
        '''
        X = self.__input_stack(X)
        X = self.__hidden_stack(X)
        X = self.__output_stack(X)
        return X

    def train_(self, X, Y):
        '''Trains the MLP on the training set [X, Y] through gradient descent

        Args:
            X (torch.Tensor): input features of the training set
            Y (torch.Tensor): target labels of the training set
        '''
        # Tailor in- and output layer to shape of X and Y
        self.__input_stack = Linear(X.shape[1], self.hidden_sizes[0])
        self.__output_stack = Linear(self.hidden_sizes[-1], Y.shape[1])

        self.train()  # train mode

        # Instantiate the optimizer
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)

        # Instantiate the loss function
        norm_factor = (torch.max(Y).item() - torch.min(Y).item())
        loss_function = NRMSELoss(norm_factor)

        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()  # don't accumulate gradients

            # Compute the loss and its gradient
            Y_pred = self(X)
            loss = loss_function(Y, Y_pred)
            loss.backward()  # back propagation

            optimizer.step()  # adjust weights and biases accordingly

    def test(self, X, Y):
        '''Computes the MSE of predictions on the test set

        Args:
            X (torch.Tensor): input features of the test set
            Y (torch.Tensor): target labels of the test set

        Returns:
            float: MSE between predicted and target labels
        '''
        self.eval()  # evaluation mode

        # Compare Y to predictions on X
        Y_pred = self(X)
        loss_function = MSELoss()
        mse = loss_function(Y, Y_pred).item()

        return mse

    # Retrieved from SoftDecisionTree by Youri Coppens
    # https://github.com/endymion64/SoftDecisionTree/blob/master/sdt/model.py#L153
    def save(self, folder_path, save_name):
        '''Saves current state of MLP to .pt file

        Args:
            folder_path (str): path to destination folder
            save_name (str): name of the .pt file
        '''
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        with open(Path(folder_path, save_name + '.pt'), 'wb') as output_file:
            torch.save(self.state_dict(), output_file)

    @classmethod
    def load(cls, folder_path, save_name):
        '''Loads state from previously saved .pt file

        Args:
            folder_path (str): path to folder
            save_name (str): name of the .pt file

        Returns:
            MLP: MLP with restored state
        '''
        state_dict = torch.load(f'{folder_path}/{save_name}.pt')

        # Derive size of all layers
        hidden_sizes = []
        for key, value in state_dict.items():
            if 'weight' in key:
                if 'input' in key:  input_size = np.shape(value)[1]
                if 'hidden' in key: hidden_sizes.append(np.shape(value)[1])
                if 'output' in key:
                    hidden_sizes.append(np.shape(value)[1])
                    output_size = np.shape(value)[0]

        # Reconstruct MLP object
        mlp = cls(hidden_sizes)
        mlp.__input_stack = Linear(input_size, mlp.hidden_sizes[0])
        mlp.__output_stack = Linear(mlp.hidden_sizes[-1], output_size)
        return mlp

