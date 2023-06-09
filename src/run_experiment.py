import sys
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.circles_vs_houses import Circles_vs_Houses
from evaluations.evaluate_metrics import evaluate_unfaithfulness, evaluate_fidelity
from evaluations.visualize_explanations import create_explanations, visualize_dataset
from explainers.gsatexplainer import get_gsatexplainer
from explainers.pgexplainer import get_pgexplainer
import yaml
import torch
from classifiers.gin import GIN
from classifiers.abstract_classifier import GraphClassifier
from classifiers.gcn import GCN
from classifiers.graphconv import GraphConvNet
from torch_geometric.datasets import BA2MotifDataset, TUDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from classifiers.train_classfier import ClassifierTrainer
from explainers.gnnexplainer import get_gnnexplainer
from explainers.lars_gnnexplainer import get_larsgnnexplainer
from torch_geometric.seed import seed_everything


def get_data_loaders(dataset_name: str, batch_size: int) -> tuple[Dataset, dict]:
    if dataset_name == "ba_2motifs":
        dataset = BA2MotifDataset("./../data/").shuffle()
    elif dataset_name == "circles_vs_houses":
        dataset = Circles_vs_Houses("./../data/", "ba_2motifs").shuffle()
    elif dataset_name == "mutag":
        dataset = TUDataset("./../data/", "MUTAG").shuffle()
    elif dataset_name == "dd":
        dataset = TUDataset("./../data/", "DD").shuffle()
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
        assert loaders is not None
        return get_pgexplainer(classifier, loaders['train'], configuration)
    if explainer_name == "GSATExplainer":
        return get_gsatexplainer(classifier, loaders, configuration)
    if explainer_name == "LarsGNNExplainer":
        larsgnnexplainer_config = configuration["lars_gnnexplainer"]
        return get_larsgnnexplainer(classifier, larsgnnexplainer_config)
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

def main(dataset_name: str, classifier_name: str, explainer_names: list[str]):
    seed_everything(42)
    Path(f"./logs/{dataset_name}/").mkdir(parents=True, exist_ok=True)
    with open(f'./config/{dataset_name}.yml', 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    batch_size = configuration['batch_size']
    dataset, loaders = get_data_loaders(dataset_name, batch_size)
    
    configuration['num_classes'] = dataset.num_classes
    configuration['dataset_name'] = dataset_name
    configuration['classifier_name'] = classifier_name
    
    classifier, model_configuration = get_classification_model(dataset, classifier_name, configuration)
    classifier_state_dict_path = f"logs/{dataset_name}/{classifier_name}.pt"
    log_file_path = f"logs/{dataset_name}/{classifier_name}-train_log.txt"
    trainer = ClassifierTrainer(log_file_path)
    if configuration['use_cached_classifier'] and Path(classifier_state_dict_path).exists():
        classifier.load_state_dict(torch.load(classifier_state_dict_path))
        trainer.quick_eval_model(classifier, dataset, loaders['test'], model_configuration)
    else:
        trainer.train(classifier, loaders, dataset, model_configuration)
        torch.save(classifier.state_dict(), classifier_state_dict_path)
    
    # visualize_dataset(dataset, 5, dataset_name)
    
    topk = configuration['topk']
    explainers = []
    for explainer_name in explainer_names:
        explainer = get_explainer(explainer_name, classifier, configuration, loaders)
        create_explanations(explainer, explainer_name, dataset[int(0.8*len(dataset)):], f"{dataset_name} with {classifier_name}", topk=topk)
        explainers.append(explainer)
    # evaluate_unfaithfulness(explainer, explainer_name, dataset[int(0.95*len(dataset)):])
    evaluate_fidelity(explainers, explainer_names, dataset[int(0.95*len(dataset)):], f"{dataset_name} with {classifier_name}")


if __name__ == "__main__":
    EXPLAINERS = ["GNNExplainer", "PGExplainer", "GSATExplainer"]
    main("circles_vs_houses", "GCN", EXPLAINERS)
