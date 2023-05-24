import torch
import numpy as np
import networkx as nx
from networkx import Graph
from torch_geometric.utils import to_networkx
from torch_geometric.explain import Explainer, Explanation
import matplotlib.pyplot as plt


def create_explanations(explainer: Explainer,  explainer_name: str, dataset, topk=0.2):
    #creating 5 explanations of correctly classified graphs
    correctly_classified_graphs = []
    i = 0
    while len(correctly_classified_graphs) < 5:
        graph = dataset[i]
        batch = torch.zeros(len(graph.x), dtype=torch.int64)
        raw_pred = explainer.get_prediction(graph.x, graph.edge_index, batch=batch)
        pred = (raw_pred.sigmoid() > 0.5).float()
        print(f"pred: {pred} vs target: {graph.y}")
        if pred.data == graph.y.data:
            correctly_classified_graphs.append((graph, pred))
        i += 1
        
    
    fig, axes = plt.subplots(5, 1, figsize=(20,30))
    plt.suptitle(f"Visualizations from {explainer_name}", size=40)
    for num_subplot, (graph, pred) in enumerate(correctly_classified_graphs):
        batch = torch.zeros(len(graph.x), dtype=torch.int64)
        explanation: Explanation = explainer(graph.x, graph.edge_index, target=graph.y, batch=batch)
        
        topk_num = int(len(graph.edge_index[0]) * topk)
        threshold = torch.sort(explanation.get("edge_mask"))[0][-topk_num]
        edge_mask = explanation.get("edge_mask") * (explanation.get("edge_mask")>threshold)
        print(f"edge_mask: {edge_mask}\n")

        graph.att = edge_mask
        
        visualize_graph(graph, pred, axes[num_subplot], explainer_name)

def visualize_graph(graph, pred, ax, explainer_name, nodesize=300):

    G = to_networkx(graph)
    pos = nx.kamada_kawai_layout(G)

    widths = [max(10*att, 2) if att != 0 else 0.3 for att in graph.att]
    edge_color = [(1,0,0) if att > 0 else (0,0,0) for att in graph.att]
    
    ax.set_title(f"{graph.y.data} vs {pred.data}", fontsize=30)
    
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=nodesize, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_color, arrows=False, ax=ax)

    plt.savefig(f"../explanation_visualizations/Visaulaizations_{explainer_name}")
    
        
