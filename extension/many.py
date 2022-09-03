import os.path as osp
import argparse
from torch_geometric.datasets import *
import torch
import torch.nn.functional as F

from torch_geometric.nn import SGConv, GCNConv

from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='facebook',
                    choices=['reddit', 'flickr','facebook','github','twitch','deezer'])
parser.add_argument('--model', type=str, default='GCN',
                    choices=['GCN','SGC'])
args = parser.parse_args()


if(args.dataset == "facebook"):
    dataset = FacebookPagePage("../dataset/facebook")
elif(args.dataset == "reddit"):
    dataset = Reddit("../dataset/reddit")
elif(args.dataset == "flickr"):
    dataset = Flickr("../dataset/flickr")
elif(args.dataset == "github"):
    dataset = GitHub("../dataset/github")
elif(args.dataset == "twitch"):
    dataset = Twitch("../dataset/twitch","EN")
elif(args.dataset == "deezer"):
    dataset = DeezerEurope("../dataset/deezer")

data = dataset[0]

split_idx = torch.arange(len(data.x))
train_idx, _, test_idx, _ = train_test_split(split_idx, split_idx, test_size=0.2, random_state=42)
train_idx, _, valid_idx, _ = train_test_split(train_idx, train_idx, test_size=0.25, random_state=42)

train_mask = torch.zeros(len(data.x), dtype=torch.bool)
valid_mask = torch.zeros(len(data.x), dtype=torch.bool)
test_mask = torch.zeros(len(data.x), dtype=torch.bool)

for position in train_idx.numpy():
    train_mask[position] = True
for position in valid_idx.numpy():
    valid_mask[position] = True
for position in test_idx.numpy():
    test_mask[position] = True

class SGC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=2,
                            cached=True)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(args.model=="SGC"):
    model, data = SGC().to(device), data.to(device)
elif(args.model=="GCN"):
    model, data = GCN().to(device), data.to(device)

train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
valid_mask = valid_mask.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.005)

def train():
    model.train()
    optimizer.zero_grad()
    if(args.model=="SGC"):
        F.nll_loss(model()[train_mask], data.y[train_mask]).backward()
    elif(args.model=="GCN"):
        F.nll_loss(model(data)[train_mask], data.y[train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    if(args.model=="SGC"):
        logits, accs = model(), []        
    elif(args.model=="GCN"):
        logits, accs = model(data), []

    for mask in train_mask,valid_mask,test_mask:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
for epoch in range(1, 101):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

