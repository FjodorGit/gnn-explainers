import torch
import sys
import networkx as nx
from networkx import Graph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.explain import Explainer, Explanation
from collections import defaultdict
import matplotlib.pyplot as plt


def create_explanations(explainer: Explainer,  explainer_name: str, dataset, dataset_name: str, topk=0.2):
    #creating 5 explanations of correctly classified graphs
    correctly_classified_graphs = []
    i = 0
    while len(correctly_classified_graphs) < 5:
        graph: Data = dataset[i]
        batch = torch.zeros(len(graph.x), dtype=torch.int64)
        raw_pred = explainer.get_prediction(graph.x, graph.edge_index, batch=batch)
        pred = (raw_pred.sigmoid() > 0.5).float()
        if pred.data == graph.y.data:
            correctly_classified_graphs.append((graph, pred))
        i += 1
        
    
    fig, axes = plt.subplots(5, 1, figsize=(7,7*5))
    plt.suptitle(f"Visualizations from {explainer_name} of {dataset_name}", size=20)
    for num_subplot, (graph, pred) in enumerate(correctly_classified_graphs):
        batch = torch.zeros(len(graph.x), dtype=torch.int64)
        explanation: Explanation = explainer(graph.x, graph.edge_index, target=graph.y, batch=batch)
        
        topk_num = int(len(graph.edge_index[0]) * topk)
        threshold = torch.sort(explanation.get("edge_mask"))[0][-topk_num]
        edge_mask = explanation.get("edge_mask") * (explanation.get("edge_mask")>threshold)
        print(f"edge_mask: {edge_mask}\n")

        graph.att = edge_mask
        
        visualize_explanation(graph, dataset_name, axes[num_subplot])

    fig.savefig(f"../explanation_visualizations/Visaulaizations_from_{explainer_name}_of_{dataset_name}")
    
def visualize_dataset(dataset ,num_graphs: int, dataset_name: str):

    fig, axes = plt.subplots(num_graphs, 1, figsize=(7,7*num_graphs))
    fig.suptitle(f"{dataset_name} visualized", fontsize=20)

    if dataset_name == "mutag":
        
        ATOM_MAP = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        for i ,ax in enumerate(axes):
        
            pos = nx.spring_layout(G)
           
            ax.set_title(f"Graphlabel: {graph.y}")
            widths = [1 for _ in graph.edge_index]
            edge_color = [(0,0,0) for _ in graph.edge_index]
            nodelist = range(len(graph.x))
            node_labels = {i: ATOM_MAP[(feature == 1).nonzero()] for i,feature in enumerate(graph.x)}
            
            nx.draw_networkx_nodes(G, pos, node_color='white', node_size=200, nodelist=nodelist, ax=ax)
            nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_color, arrows=False, ax=ax)
            nx.draw_networkx_labels(G, pos, node_labels, font_color="black", ax=ax)
    
    else:
        for i ,ax in enumerate(axes):
        
            graph = dataset[i]
            G = to_networkx(graph)
            pos = nx.kamada_kawai_layout(G)
           
            ax.set_title(f"Graphlabel: {graph.y}")
            widths = [1 for _ in graph.edge_index]
            edge_color = [(0,0,0) for _ in graph.edge_index]
            
            nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=200, ax=ax)
            nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_color, arrows=False, ax=ax)

    fig.savefig(f"../explanation_visualizations/Visaulaizations_from_{dataset_name}")
    plt.show()
    return
    
def visualize_explanation(graph, dataset_name, ax):

    G = to_networkx(graph)
    ax.set_title(f"Graphlabel: {graph.y}")
    edge_dict = defaultdict(float)
    for att ,(u,v) in zip(graph.att ,G.edges):
        edge_dict[(u,v)] += att
        edge_dict[(v,u)] += att

    if dataset_name == "mutag":
        
        ATOM_MAP = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        
        pos = nx.spring_layout(G)
       
        widths = [max(3*edge_dict[edge], 1) if edge_dict[edge] > 0 else 0.5 for edge in G.edges]
        edge_color = [(1,0,0) if edge_dict[edge] > 0 else (0,0,0) for edge in G.edges]
        nodelist = range(len(graph.x))
        node_labels = {i: ATOM_MAP[(feature == 1).nonzero()] for i,feature in enumerate(graph.x)}
        
        nx.draw_networkx_nodes(G, pos, node_color='white', node_size=200, nodelist=nodelist, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_color, arrows=False, ax=ax)
        nx.draw_networkx_labels(G, pos, node_labels, font_color="black", ax=ax)
    else:
        pos = nx.kamada_kawai_layout(G)
        
        widths = [max(3*edge_dict[edge], 1) if edge_dict[edge] > 0 else 0.5 for edge in G.edges]
        edge_color = [(1,0,0) if edge_dict[edge] > 0 else (0,0,0) for edge in G.edges]
        
        nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=200, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_color, arrows=False, ax=ax)
