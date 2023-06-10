from pathlib import Path
from posixpath import exists
from graph_classification import load_model
import torch
import sys
from torch_geometric.explain import Explainer, ExplainerConfig, ModelConfig, PGExplainer
from classifiers.abstract_classifier import GraphClassifier

def get_pgexplainer(model: GraphClassifier, train_loader, configuration: dict) -> Explainer:
    
    explainer_configuration = configuration['pgexplainer']
    lr = explainer_configuration['lr']
    use_cached_explainer = explainer_configuration.get('use_cached_explainer', False)
    dataset_name = configuration['dataset_name']
    classifier_name = configuration['classifier_name']

    if use_cached_explainer:
        epochs = 0
    else:
        epochs = explainer_configuration['epochs']
        
    pgexplainer: Explainer = Explainer(
        model=model,
        algorithm=PGExplainer(lr=lr, epochs=epochs),
        explanation_type=explainer_configuration['explanation_type'],
        node_mask_type=explainer_configuration['node_mask_type'],
        edge_mask_type=explainer_configuration['edge_mask_type'],
        model_config=explainer_configuration['model_config'])    

    explainer_state_dict_path = f"../models/explainers/pgexplainer-{dataset_name}-{classifier_name}.pt"
    #load model
    if use_cached_explainer and Path(explainer_state_dict_path).exists():
        pgexplainer.algorithm = load_model(explainer_state_dict_path, PGExplainer, epochs=epochs, lr=lr)
        config = ExplainerConfig(explainer_configuration['explanation_type'],explainer_configuration['node_mask_type'],explainer_configuration['edge_mask_type'])
        pgexplainer.algorithm.connect(config, explainer_configuration['model_config'])
        return pgexplainer
        
    #train the explainer
    for epoch in range(epochs):
        epoch_loss = 0
        for data in train_loader:
            epoch_loss += pgexplainer.algorithm.train(epoch, model, data.x, data.edge_index, target=data.y, batch=data.batch)
        print(f"PGExplainer loss in epoch {epoch}: {epoch_loss/len(train_loader)}")

    #save model
    torch.save(pgexplainer.algorithm.state_dict(), explainer_state_dict_path)
    
    return pgexplainer
    

