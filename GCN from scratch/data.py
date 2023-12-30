import torch
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}

def get_ds():
    '''
    get data
    '''
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    X = torch.eye(N, N)
    
    node_labels = [1 if G.nodes[v]['club'] == 'Mr. Hi' else 0 for v in G]
    y_train = np.array(node_labels)
    Y_train = OneHotEncoder(categories='auto').fit_transform(y_train.reshape(-1, 1))
    Y_train = Y_train.toarray()
    Y = torch.tensor(Y_train)

    # Create a custom dataset
    dataset = {'feature':X, 'label': Y}

    # Create DataLoader
    data_loader = DataLoader(dataset, shuffle=True)
    mask_tr = torch.BoolTensor([0] * 34)
    mask_tr[[0, 33]] = 1

    return G, dataset, mask_tr

graph, data_loader,_ = get_ds()

features = data_loader['feature']
labels = data_loader['label']
print("Features:", features)
print("Labels:", labels)
