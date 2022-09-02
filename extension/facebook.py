import os.path as osp

from torch_geometric.datasets import FacebookPagePage
import torch
import torch.nn.functional as F

from torch_geometric.nn import SGConv

from sklearn.model_selection import train_test_split

dataset = FacebookPagePage("./dataset")
data = dataset[0]
split_idx = torch.arange(22470)
train_idx, _, test_idx, _ = train_test_split(split_idx, split_idx, test_size=0.2, random_state=42)
train_idx, _, valid_idx, _ = train_test_split(train_idx, train_idx, test_size=0.25, random_state=42)
data = dataset[0]

train_mask = torch.zeros(22470, dtype=torch.bool)
valid_mask = torch.zeros(22470, dtype=torch.bool)
test_mask = torch.zeros(22470, dtype=torch.bool)

for position in train_idx.numpy():
    train_mask[position] = True
for position in valid_idx.numpy():
    valid_mask[position] = True
for position in test_idx.numpy():
    test_mask[position] = True

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(128, 4, K=2,
                            cached=True)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
valid_mask = valid_mask.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.005)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], data.y[train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(), []
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
