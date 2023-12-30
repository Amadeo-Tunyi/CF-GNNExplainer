import torch
import torch.nn as nn
import torch.nn.functional as F
from params import get_params
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from GCN import GCN
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from data import get_ds
import warnings

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # Compute softmax cross-entropy loss
    loss = F.cross_entropy(preds, labels, reduction='none')

    # Apply mask
    mask = mask.float()
    mask /= mask.mean()
    loss *= mask

    # Calculate mean loss


    return loss.mean()


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # Calculate predictions
    _, predicted = torch.max(preds, 1)
    lab = torch.tensor(np.argmax(labels, axis = 1))

    # Compare predictions with true labels
    correct_prediction = torch.eq(predicted, lab).float()

    # Apply mask
    mask = mask.float()
    mask /= mask.mean()
    correct_prediction *= mask

    # Calculate mean accuracy
    return correct_prediction.mean()

def loss_fn(preds, labels, mask, norm_adj, Z_out):
    fro = Z_out@Z_out.t() - norm_adj
    reg = torch.sqrt(torch.sum(fro@fro.t()))

    lambdaa = 0.1
    cost = masked_softmax_cross_entropy(preds, labels, mask) + lambdaa * reg
    return cost

G, train_loader, mask = get_ds()



def get_model(params):
    model = GCN(G, params['d_classes'])
    return model


def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

def train_model(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = get_model(params).to(device)


    optimizer = optim.Adam(model.parameters(), lr = params['lr'] , eps = 1e-4)

    initial_epoch = 0
    global_step = 0



    # Step 2: Define your loss function
    loss_function = loss_fn  
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Example usage in the training loop
    # Assuming train_loader is your data loader and model is in training mode
    
    for epoch in tqdm(range(params['num_epochs']), desc="Processing", unit="epoch"):
        model.train()
        for batch in tqdm(range(1), desc='Batches', unit='batch', leave=False):
            
            
            optimizer.zero_grad() 
            inputs = train_loader['feature']# Zero the gradients

            Z_out = model.forward(inputs)
            norm_adj = model.NomralizedAdjMat()
            preds = torch.softmax(Z_out, dim = -1)
            y = torch.tensor(np.argmax(train_loader['label'], axis = 1))
            acc = accuracy(torch.argmax(preds, dim = 1),y)

            loss = loss_function(preds, train_loader['label'], mask, norm_adj, Z_out)
        

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights#
            #

            global_step += 1
            tqdm.write(f'Epoch {epoch + 1}, Batch {batch + 1}, Accuracy {acc}, Loss = {loss:.4f}')

    # Ensure that the tqdm bars are properly closed
    tqdm.write("Training completed.")
if  __name__ == '__main__':
    warnings.filterwarnings('ignore')
    params = get_params()   
    train_model(params)
    
    



