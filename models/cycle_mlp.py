from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F


class mlp(nn.Module):
    def __init__(self, 
                 input_features : int , 
                 hidden_features : int = 0, 
                 output_features : int = 0,
                 drop = 0.0):
        """ Initializes  MLP layer

        Args:
            input_features (int): Number of input features
            hidden_features (int, optional): Number of features of intermediate layer. Defaults to 0.
            output_features (int, optional): Number of outpput features Defaults to 0.
            drop (float, optional): Dropout probability . Defaults to 0.0.
        """
        output_features = output_features or input_features
        hidden_features = hidden_features or input_features
        self.fc_1 = nn.Linear(in_features=input_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc_2 = nn.Linear(in_features=hidden_features, out_features=output_features)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x : torch.Tensor):
        """ Forward function for MLP

        Args:
            x (torch.Tensor): Input tensor.
        """
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        x = self.dropout(x)