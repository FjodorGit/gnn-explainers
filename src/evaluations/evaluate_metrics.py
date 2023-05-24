import torch
import torch.nn.functional as F
import sys
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm.utils import clear_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelReturnType
from torch_geometric.explain.metric import unfaithfulness

def evaluate_unfaithfulness(explainer: Explainer, explainer_name: str, evaluation_dataset):
    combined_unf = []
    benchmark = []
    for graph in evaluation_dataset:
        batch = torch.zeros(len(graph.x), dtype=torch.int64)
        explanation = explainer(graph.x, graph.edge_index, target=graph.y, batch=batch)
        unf, benchmark_unf = my_unfaithfulness(explainer, explanation)
        print(f"unfaithfulness: {unf}\n")
        combined_unf.append(unf)
        benchmark.append(benchmark_unf)
    plt.boxplot([combined_unf, benchmark],labels=[explainer_name, "Random"])
    plt.title(f"Unfaitfulness by {explainer_name}")
    plt.show()

def my_unfaithfulness(explainer: Explainer, explanation, top_k = 0.2, multi_label=False):
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    if top_k is not None and explainer.node_mask_type == MaskType.object:
        raise ValueError("Cannot apply top-k feature selection based on a "
                         "node mask of type 'object'")

    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    edge_mask = convert_to_thresholded_edge_mask(edge_mask, top_k)
    print(f"Edge mask: {edge_mask}")
    x, edge_index = explanation.x, explanation.edge_index
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.get('prediction')
    if y is None:  # == ExplanationType.phenomenon
        y = explainer.get_prediction(x, edge_index, **kwargs)

    if node_mask is not None and top_k is not None:
        feat_importance = node_mask.sum(dim=0)
        _, top_k_index = feat_importance.topk(top_k)
        node_mask = torch.zeros_like(node_mask)
        node_mask[:, top_k_index] = 1.0

    y_hat = explainer.get_masked_prediction(x, edge_index, node_mask, edge_mask, **kwargs)
    
    #random explanation
    random_node_mask = torch.rand(node_mask.shape) if node_mask is not None else None
    random_edge_mask = torch.rand(edge_mask.shape)
    random_edge_mask = convert_to_thresholded_edge_mask(random_edge_mask, top_k)
    print(f"Random mask: {random_edge_mask}")
    benchmark_y_hat = explainer.get_masked_prediction(x, edge_index, random_node_mask, random_edge_mask, **kwargs) 

    if explanation.get('index') is not None:
        y, y_hat = y[explanation.index], y_hat[explanation.index]

    if explainer.model_config.return_type == ModelReturnType.raw and not multi_label:
        print(f"y: {y} | y_hat: {y_hat} | benchmark_y_hat: {benchmark_y_hat}")
        y, y_hat = y.sigmoid(), y_hat.sigmoid()
        benchmark_y_hat = benchmark_y_hat.sigmoid()
        print(f"y: {y} | y_hat: {y_hat} | benchmark_y_hat: {benchmark_y_hat}")
    elif explainer.model_config.return_type == ModelReturnType.raw and multi_label:
        y, y_hat = y.sigmoid(dim=-1), y_hat.softmax(dim=-1)
    elif explainer.model_config.return_type == ModelReturnType.log_probs:
        y, y_hat = y.exp(), y_hat.exp()
    
    # kl_div = F.kl_div(y_hat.log(), y, reduction="batchmean")
    loss = F.mse_loss(y_hat, y)
    benchmark_loss = F.mse_loss(benchmark_y_hat, y)
    print(f"loss: {loss} vs benchmark: {benchmark_loss}")
    return loss, benchmark_loss

def convert_to_thresholded_edge_mask(mask, topk):
    topk_num = int(len(mask) * topk)
    threshold = torch.sort(mask)[0][-topk_num]
    edge_mask = (mask>threshold).float()
    return edge_mask
                           
    
