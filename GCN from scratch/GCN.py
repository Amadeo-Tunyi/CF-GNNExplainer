import warnings; warnings.filterwarnings('ignore')
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from IPython.display import HTML 
from sklearn.preprocessing import OneHotEncoder
import community
import torch
import torch.nn as nn



class GCN(nn.Module):

    def __init__(self, Graph: nx.Graph, hidden :int = 4, d_classes: int = 2) -> None:
        super().__init__()
        self.G = Graph
        self.relu = nn.ReLU()
        self.hidden = hidden
        self.d_classes = d_classes
        self.linear_1 = nn.Linear(self.G.number_of_nodes(), self.hidden)
        self.linear_2 = nn.Linear(self.hidden, self.d_classes)	


    def Adjacency_matrix(self):
        order = sorted(list(self.G.nodes()))
        Adj_mat = nx.to_numpy_matrix(self.G, nodelist= order)
        return Adj_mat

    def NomralizedAdjMat(self):
        A = self.Adjacency_matrix()
        I = torch.eye(*A.shape)
        A_tilde = torch.tensor(A.copy()) + I
        D_tilde = torch.tensor(torch.sum(A_tilde, dim = 0)) 
        D_inv = torch.tensor(torch.diag(D_tilde**-0.5))
        return D_inv@A_tilde@D_inv
    

    def gcn_layer(self,H, X, FeedForward: nn.Linear):
        H, X = torch.tensor(H, dtype = torch.float32), torch.tensor(X, dtype = torch.float32)
        B_hat = H@X
        C_hat = FeedForward(B_hat)
        return C_hat
    

    def forward(self, X):
    
        H = self.NomralizedAdjMat()
        x = self.gcn_layer(H, X,self.linear_1)
        x = self.relu(x)
        x = self.gcn_layer(H, x, self.linear_2)
        return x


       

