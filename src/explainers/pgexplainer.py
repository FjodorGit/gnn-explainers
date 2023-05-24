import torch
import sys
from torch_geometric.explain import Explainer, PGExplainer
from classifiers.abstract_classifier import GraphClassifier

def get_pgexplainer(model: GraphClassifier, train_loader, explainer_configuration: dict) -> Explainer:
    
    lr = explainer_configuration['lr']
    epochs = explainer_configuration['epochs']
    
    pgexplainer: Explainer = Explainer(
        model=model,
        algorithm=PGExplainer(lr=lr, epochs=epochs),
        explanation_type=explainer_configuration['explanation_type'],
        node_mask_type=explainer_configuration['node_mask_type'],
        edge_mask_type=explainer_configuration['edge_mask_type'],
        model_config=explainer_configuration['model_config'])    

    #train the explainer
    for epoch in range(epochs):
        epoch_loss = 0
        for data in train_loader:
            epoch_loss += pgexplainer.algorithm.train(epoch, model, data.x, data.edge_index, target=data.y, batch=data.batch)
        print(f"PGExplainer loss in epoch {epoch}: {epoch_loss/len(train_loader)}")
        
    return pgexplainer
