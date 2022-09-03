import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
data = dataset[0]

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

for position in train_idx.numpy():
    train_mask[position] = True
for position in valid_idx.numpy():
    valid_mask[position] = True
for position in test_idx.numpy():
    test_mask[position] = True

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)
model = GCN().to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
valid_mask = valid_mask.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_mask], torch.flatten(data.y)[train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(data), []
    for mask in train_mask,valid_mask,test_mask:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(torch.flatten(data.y)[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
for epoch in range(1, 501):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
