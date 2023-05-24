import torch
from classifiers.abstract_classifier import GraphClassifier
from torch_geometric.explain import Explainer, GNNExplainer

def get_gnnexplainer(model: GraphClassifier, explainer_configuration: dict) -> Explainer:
    
    lr = explainer_configuration['lr']
    epochs = explainer_configuration['epochs']
    
    gnnexplainer = Explainer(
        model=model,
        algorithm=GNNExplainer(lr=lr, epochs=epochs),
        explanation_type=explainer_configuration['explanation_type'],
        node_mask_type=explainer_configuration['node_mask_type'],
        edge_mask_type=explainer_configuration['edge_mask_type'],
        model_config=explainer_configuration['model_config'])    
    return gnnexplainer
