from dataReading import SlicingDataset
import numpy as np

import torch
from torch.nn import Linear
import torch.nn as nn

import dgl
import dgl.nn as dglnn
import torch.optim as optim
from dgl.nn import GraphConv, HeteroGraphConv

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

dataset_path = '../V2.2.1-3k/'
save_model_path = '../V2.2.1-model-100-dataset-inputgraphqueues01302024075531'
save_dataset_path = '../V2.2.1-3k'
dataset_name="Slicing-V2.2.1-3k"

# define dataset
dataset = SlicingDataset(
    name=dataset_name,
    raw_dir=dataset_path,
    save_dir=save_dataset_path,
    force_reload=False)

def joinFeatures(fs):
    features = []
    for feature in fs:
        if feature in ("_ID", "nodeType", "delay", "jitter"):
            continue
        else:
            features.append(fs[feature])
    return features

def genNodeFeatures(graph):
    g = graph
    link_feats = torch.transpose(torch.stack(joinFeatures(g.nodes['link'].data)),0,1).contiguous()
    path_feats = torch.transpose(torch.stack(joinFeatures(g.nodes['path'].data)),0,1).contiguous()
    queue_feats = torch.transpose(torch.stack(joinFeatures(g.nodes['queue'].data)),0,1).contiguous()

    labels = g.nodes['path'].data['delay']

    node_features = {
        'link': link_feats,
        'path': path_feats,
        'queue': queue_feats
    }

    return node_features, labels


dataset.load()
print("Dataset loaded")

graphs = dataset.getGraphs()[0]
graphs = [g for g in graphs if (g.nodes['path'].data['delay'] <= 0.1).all()]


rel_names = set()     
raw_dims   = {}           
for g in graphs:           
    rel_names.update(g.etypes)
    for ntype in g.ntypes:
        if ntype not in raw_dims:      
            feats, _ = genNodeFeatures(g)
            raw_dims[ntype] = feats[ntype].shape[1]

class GATModel(nn.Module):
    def __init__(self, raw_dims, rel_names,
                 d_proj=32, hid_feats=32, out_feats=1, num_heads=4):
        super().__init__()

        self.proj = nn.ModuleDict({
            n: nn.Linear(raw_dims[n], d_proj, bias=False)
            for n in raw_dims
        })

        self.conv1 = dglnn.HeteroGraphConv(
            {r: dglnn.GATConv(d_proj, hid_feats, num_heads) for r in rel_names},
            aggregate="sum"
        )

        self.conv2 = dglnn.HeteroGraphConv(
            {r: dglnn.GATConv(hid_feats * num_heads, out_feats, num_heads=1)
             for r in rel_names},
            aggregate="sum"
        )

    def forward(self, g, inputs):
        h = {nt: torch.relu(self.proj[nt](x)) for nt, x in inputs.items()}

        h = self.conv1(g, h)
        h = {k: torch.relu(v.flatten(1)) for k, v in h.items()}

        h = self.conv2(g, h)
        h = {k: torch.relu(v.flatten(1)) for k, v in h.items()}
        return h

model = GATModel(raw_dims=raw_dims,
                 rel_names=rel_names,
                 d_proj=32, hid_feats=32, out_feats=1, num_heads=4)

scalers = {
    'offeredTrafficIntensity' : MinMaxScaler(),
    'avgPacketSize' : MinMaxScaler(),
    'utilization' : MinMaxScaler(),
    'capacity' : MinMaxScaler(),
    'pathLength' : MinMaxScaler(),
    'packets' : MinMaxScaler(),
    'delta' : MinMaxScaler(),
    'traffic' : MinMaxScaler(),
    'maxDelay' : MinMaxScaler(),
    'calculatedLosses' : MinMaxScaler(),
    'weight' : MinMaxScaler(),
    'queueUtilization' : MinMaxScaler(),
    'jitter' : MinMaxScaler(),
    'drops' : MinMaxScaler()
}

def FitFeatures(graphs, scalers):
    feature_values = {feature: [] for feature in scalers}
    for graph in graphs:
        for feature in feature_values:
            if feature in ['offeredTrafficIntensity','avgPacketSize','utilization','capacity']:
                feature_values[feature].extend(graph.nodes['link'].data[feature].flatten().tolist())
            elif feature in ['pathLength','packets','delta','traffic', 'jitter', 'drops']:
                feature_values[feature].extend(graph.nodes['path'].data[feature].flatten().tolist())
            elif feature in ['maxDelay','calculatedLosses','weight','queueUtilization']:
                feature_values[feature].extend(graph.nodes['queue'].data[feature].flatten().tolist())
    for feature, values in feature_values.items():
        scalers[feature].fit(np.array(values).reshape(-1,1))

def TransformFeatures(graphs, scalers):
    for graph in graphs:
        for name in ['offeredTrafficIntensity','avgPacketSize','utilization','capacity']:
            arr = graph.nodes['link'].data[name].reshape(-1,1).cpu().numpy()
            graph.nodes['link'].data[name] = torch.tensor(scalers[name].transform(arr)).flatten()
        for name in ['pathLength','packets','delta','traffic', 'jitter', 'drops']:
            arr = graph.nodes['path'].data[name].reshape(-1,1).cpu().numpy()
            graph.nodes['path'].data[name] = torch.tensor(scalers[name].transform(arr)).flatten()
        for name in ['maxDelay','calculatedLosses','weight','queueUtilization']:
            arr = graph.nodes['queue'].data[name].reshape(-1,1).cpu().numpy()
            graph.nodes['queue'].data[name] = torch.tensor(scalers[name].transform(arr)).flatten()

        graph.nodes['path'].data['delay'] *= 1000

total = len(graphs)
ntrain = int(total * 3/4)

FitFeatures(graphs[:ntrain], scalers)
TransformFeatures(graphs, scalers)

configs = [
    (0.0001, 300)
]

results = []

for lr, epochs in configs:
    print(f"\nRunning config: LR={lr}, Epochs={epochs}")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    progbar = tqdm(total=ntrain*epochs, desc=f"Training (lr={lr})")
    losses = []

    for epoch in range(epochs):
        epoch_losses = []
        for graph in graphs[:ntrain]:
    
            inputs, labels = genNodeFeatures(graph)
            prediction = model(graph, inputs)['path'].squeeze()
            loss_value = loss_fn(prediction, labels)

            epoch_losses.append(loss_value.detach().item())

            optimizer.zero_grad()
            loss_value.backward()

            optimizer.step()
            progbar.update(1)
        print(np.mean(epoch_losses))
        losses.append(np.mean(epoch_losses))


    losses = np.array(losses) / 1000
    plotLosses(epochs, losses)
    model.eval()
    loss_MAE = nn.L1Loss()
    test_mae, test_mse = [], []

    i = 1
    for graph in graphs[ntrain:]:
        inputs, labels = genNodeFeatures(graph)
        pred = model(graph, inputs)['path'].squeeze()

        test_mae.append(loss_MAE(pred, labels).item())
        test_mse.append(loss_fn(pred, labels).item())

        i+=1

    mean_mae = np.mean(test_mae)
    mean_mse = np.mean(test_mse)

    print(f"Results for LR={lr}, Epochs={epochs}")
    print("MAE:", mean_mae, "MSE:", mean_mse)
    
    results.append({"lr": lr, "epochs": epochs, "MAE": mean_mae, "MSE": mean_mse})

print("\nSummary of all configurations:")
for res in results:
    print(res)

