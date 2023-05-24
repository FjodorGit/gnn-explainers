import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Dataset
import yaml
from networkx.classes.graph import Graph
from torch_geometric.data.data import Data

from torch_geometric.datasets import ExplainerDataset, BA2MotifDataset, TUDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, ExplainerAlgorithm, ExplainerConfig, GNNExplainer, HeteroExplanation, ModelConfig, PGExplainer, DummyExplainer, ThresholdConfig
from torch_geometric.visualization import graph as graphvs

import networkx as nx
import numpy as np
from torch_geometric.explain.explanation import Explanation

from torch_geometric.utils import to_networkx, to_undirected

from collections import defaultdict

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Linear, global_mean_pool, GraphConv
from torch_geometric.explain.metric import fidelity, unfaithfulness

CLASSIFICATION_MODEL_PATH = "../models/graphclass.pt"
GIN_MODEL_PATH = "../models/gin.pt"

PGEXPLAINER_MODEL_PATH = "../models/pgexplainer.pt"

NUM_HIDDEN_CHANNELS = 64

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.lin(x))
        
        return x

    def get_emb(self, x, edge_index, batch, edge_attr=None):
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        return x


def train(train_loader, criterion, optimizer, model: GCN):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        # print(f"Out: {out} -- y: {data.y}")
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader, model):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def training_classification_model(model, train_dataset, test_dataset, state_dict_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
        
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    for epoch in range(1, 171):
        train(train_loader, criterion, optimizer, model)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    torch.save(model.state_dict(), state_dict_path)
    
def load_model(state_dict_path, model_type, **kargs):
    model = model_type(**kargs)
    model.load_state_dict(torch.load(state_dict_path))
    return model

def visualize_explanations_on_mutag(graphs: list[Data], explainers: list[tuple[Explainer, str]], seed: int | None, topk=0.25):      
    
    ATOM_MAP = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
    
    num_subplot = 0
    plt.figure(figsize=(20,40))
    plt.suptitle(f"Comparison of Explainers", size=50)
    
    for graph_num, graph in enumerate(graphs):
        
        target = torch.tensor([[0,1]]) if graph.y == 1 else torch.tensor([[1,0]])
        
        '''Set graph labels '''
        netx_graph: Graph = to_networkx(graph, node_attrs=['x'])
        node_labels = dict()
        for u, data in netx_graph.nodes(data=True):
            if data is None:
                continue
            data['name'] = ATOM_MAP[data['x'].index(1.0)]
            node_labels[u] = data['name']
            del data['x']
        netx_graph = netx_graph.copy().to_undirected()

        for explainer, explainer_name in explainers:
            num_subplot += 1
            batch = torch.zeros(len(graph.x), dtype=torch.int64)
            pred = explainer.get_prediction(graph.x, graph.edge_index, batch=batch)
            explanation: Explanation | HeteroExplanation = explainer(graph.x, graph.edge_index, target=target, batch=batch)
            
            topk_num = int(len(graph.edge_index[0]) * topk)
            threshold = torch.sort(explanation.get("edge_mask"))[0][-topk_num]
            
            edge_mask = explanation.get("edge_mask") * (explanation.get("edge_mask")>threshold)
            undirected_graph_edge_mask = aggregate_edge_directions(edge_mask, graph)

            pos = nx.kamada_kawai_layout(netx_graph)

            widths = [3*undirected_graph_edge_mask[key] for key in netx_graph.edges]
            edge_color = [(1,0,0) if undirected_graph_edge_mask[key] > 0 else (0,0,0) for key in netx_graph.edges]
            column_title = f"Explanaitions from {explainer_name}\n\n" if graph_num == 0 else ''
            visualization_title = column_title + f"{target.tolist()} vs [{pred[0][0]:.4f}, {pred[0][1].item():.4f}]" 
            plt.subplot(len(graphs),len(explainers), num_subplot, title=visualization_title).title.set_size(30)
            nx.draw(netx_graph, pos=pos, labels=node_labels, node_size=600, node_color = "white", font_color="black", font_size="30")
            nx.draw_networkx_edges(netx_graph, pos=pos, width=widths, edge_color=edge_color)
            
    plt.savefig(f"../explanation_visualizations/Explanations_visualization_seed_{seed}")
    
def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict

def test_explainers_unfaithfulness(dataset, *args):
    unfaithfulness_container = []
    for explainer, name in args:
        explainer_unfaitfullness = []
        for graph in dataset:
            target = torch.tensor([0,1]) if graph.y == 1 else torch.tensor([1,0])
            batch = torch.zeros(len(graph.x), dtype = torch.int64)

            explanation = explainer(graph.x, graph.edge_index, target=target, batch=batch) 
            unf = unfaithfulness(gnnexplainer, explanation)
            print(unf)
            
            explainer_unfaitfullness.append(unf)
        print(f"{name} finished.")
        unfaithfulness_container.append(explainer_unfaitfullness)
    plt.boxplot(unfaithfulness_container, labels=[name for _, name in args])
    plt.title(f"Unfaitfulness on Dataset {dataset.name}")
    plt.savefig(fname="unfaithfulness_boxplot.png")
    plt.show()

def pgexplainer_train(model,dataset, algorithm: PGExplainer, path_to_save_explainer=None):
    for epoch in range(algorithm.epochs):
        epoch_loss = 0
        for graph in dataset:
            target = torch.tensor([0,1]) if graph.y == 1 else torch.tensor([1,0])
            batch = torch.zeros(len(graph.x), dtype = torch.int64)
            epoch_loss += algorithm.train(epoch, model, graph.x, graph.edge_index, target=target, batch = batch)
        print(f"PGExplainer loss in epoch {epoch}: {epoch_loss/len(dataset)}")
        
    #saving model 
    if path_to_save_explainer is not None:
        torch.save(algorithm.state_dict(), path_to_save_explainer)

def run_explanation_visualization(seeds: list[int]):
    for seed in seeds:
        torch_geometric.seed_everything(seed)
        
        dataset = TUDataset('../data', name='MUTAG').shuffle()
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.2):]

        # model = GCN(hidden_channels=NUM_HIDDEN_CHANNELS, dataset = train_dataset)
        # 
        # training_classification_model(model, train_dataset, test_dataset, CLASSIFICATION_MODEL_PATH)
        
        model = load_model(CLASSIFICATION_MODEL_PATH, GCN, hidden_channels=NUM_HIDDEN_CHANNELS, dataset=dataset)
        
        gnnexplainer = Explainer(
            model=model,
            algorithm=GNNExplainer(),
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs'
            ),

            threshold_config = ThresholdConfig(
                    threshold_type = "topk",
                    value = 10
                )
        )
        dummyexplainer = Explainer(
            model=model,
            algorithm=DummyExplainer(),
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs'),
        )
        pgexplainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.0003),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs'),
            threshold_config = ThresholdConfig(
                    threshold_type = "topk",
                    value = 10
                )
        )

        # '''Initailize PGExplainer from saved file'''
        # pgexplainer_modelconfig: ModelConfig = ModelConfig(mode = "binary_classification", task_level="graph", return_type="probs") 
        # pgexplainer_explainerconfig: ExplainerConfig = ExplainerConfig(explanation_type= "phenomenon", node_mask_type=None, edge_mask_type="object") 
        # pgexplainer_algorithm = load_model(PGEXPLAINER_MODEL_PATH, PGExplainer, epochs=0, lr=0.0003)
        # pgexplainer_algorithm.connect(explainer_config=pgexplainer_explainerconfig, model_config=pgexplainer_modelconfig)
        # pgexplainer.algorithm = pgexplainer_algorithm
        
        '''Train PGExplainer'''
        pgexplainer_train(model, train_dataset, pgexplainer.algorithm, path_to_save_explainer=PGEXPLAINER_MODEL_PATH)
        
        visualize_explanations_on_mutag(test_dataset[:5], [(gnnexplainer, "GNNExplainer"), (pgexplainer, "PGExplainer")], seed=seed)

def test_gsatexplainer_visualization():
    torch_geometric.seed_everything(41)
    dataset = TUDataset('../data', name='MUTAG').shuffle()
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.2):]
    
    local_config = yaml.safe_load(Path("./GSAT/src/configs/GIN-mutag.yml").open('r'))
    method_config = local_config["GSAT_config"]
    shared_config = local_config["shared_config"]
    model_config = local_config["model_config"]
    multi_label = True
    
    model = GCN(hidden_channels=NUM_HIDDEN_CHANNELS, dataset = train_dataset)

    gsatexplainer = Explainer(
        model=model,
        algorithm=GSATExplainer(train_dataset, model),
        explanation_type='phenomenon',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs'),
        threshold_config = ThresholdConfig(
                threshold_type = "topk",
                value = 10
            )
    )
    
    # training_classification_model(model, train_dataset, test_dataset, GIN_MODEL_PATH)
    gsatexplainer.algorithm.train()
    
    visualize_explanations_on_mutag(test_dataset[5:10], [(gsatexplainer, "GSAT")], seed=41)
    
if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    run_explanation_visualization([41,42,43,44])
    # test_gsatexplainer_visualization()
