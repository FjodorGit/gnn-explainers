from pathlib import Path
from evaluations.evaluate_metrics import evaluate_unfaithfulness
from evaluations.visualize_explanations import create_explanations
from explainers.gsatexplainer import get_gsatexplainer
from explainers.pgexplainer import get_pgexplainer
import yaml
import torch
from classifiers.gin import GIN
from classifiers.abstract_classifier import GraphClassifier
from classifiers.gcn import GCN
from classifiers.graphconv import GraphConvNet
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from classifiers.train_classfier import quick_eval_model, run_one_epoch, train
from explainers.gnnexplainer import get_gnnexplainer

def get_data_loaders(dataset_name: str, batch_size: int) -> tuple[Dataset, dict]:
    if dataset_name == "ba_2motifs":
        dataset = BA2MotifDataset("./../data/").shuffle()
    else:
        raise NotImplemented
    train_set = dataset[:int(len(dataset) * 0.8)]
    valid_set = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test_set: Dataset = dataset[int(len(dataset) * 0.9):]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return dataset, {"train": train_loader, "valid": valid_loader, "test": test_loader}

def get_explainer(explainer_name: str, classifier: GraphClassifier, configuration: dict, loaders=None):
    if explainer_name == "GNNExplainer":
        gnnexplainer_config = configuration['gnnexplainer']
        return get_gnnexplainer(classifier, gnnexplainer_config)
    if explainer_name == "PGExplainer":
        pgexplainer_config = configuration['pgexplainer']
        assert loaders is not None
        return get_pgexplainer(classifier, loaders['train'], pgexplainer_config)
    if explainer_name == "GSATExplainer":
        return get_gsatexplainer(classifier, loaders, configuration)
    else:
        raise NotImplemented

def get_classification_model(dataset:Dataset ,model_name: str, configuration: dict) -> tuple[GraphClassifier, dict]:
    if model_name == "GIN":
        gin_configuration = configuration['gin']
        return GIN(dataset.num_features, dataset.num_classes , dataset.num_edge_features,gin_configuration), gin_configuration
    if model_name == "GCN":
        gcn_configuration = configuration['gcn']
        return GCN(dataset.num_features, dataset.num_classes , dataset.num_edge_features, gcn_configuration), gcn_configuration
    if model_name == "GraphConv":
        graphconv_config = configuration['graphconv']
        return GraphConvNet(dataset.num_features, dataset.num_classes , dataset.num_edge_features, graphconv_config), graphconv_config
    else:
        raise NotImplemented

def main(dataset_name: str, classifier_name: str, explainer_name: str):
    with open(f'./config/{dataset_name}.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    batch_size = configuration['batch_size']
    dataset, loaders = get_data_loaders(dataset_name, batch_size)
    
    configuration['num_classes'] = dataset.num_classes
    classifier, model_configuration = get_classification_model(dataset, classifier_name, configuration)
    classifier_state_dict_path = f"../models/{dataset_name}-{classifier_name}.pt"
    if configuration['use_cached_classifier'] and Path(classifier_state_dict_path).exists():
        classifier.load_state_dict(torch.load(classifier_state_dict_path))
        quick_eval_model(classifier, dataset, loaders['test'], model_configuration)
    else:
        train(classifier, loaders, dataset, model_configuration)
    torch.save(classifier.state_dict(), classifier_state_dict_path)
    
    topk = configuration['topk']
    explainer = get_explainer(explainer_name, classifier, configuration, loaders)
    # create_explanations(explainer, explainer_name, dataset[int(0.8*len(dataset)):], topk=topk)
    evaluate_unfaithfulness(explainer, explainer_name, dataset[int(0.95*len(dataset)):])


if __name__ == "__main__":
    main("ba_2motifs", "GraphConv", "PGExplainer")
