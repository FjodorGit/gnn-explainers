from torch_geometric.utils import subgraph, is_undirected
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import ExplainerAlgorithm, Explanation
from GSAT.src.run_gsat import GSAT, ExtractorMLP
from GSAT.example.trainer import eval_one_batch
from GSAT.src.utils import Writer
from GSAT.src.models.gin import GIN
import torch
import yaml
from pathlib import Path


class GSATExplainer(ExplainerAlgorithm):
    def __init__(self, train_dataset: Dataset, epochs=100, lr=0.003, **kwargs) -> None:
        super().__init__()

        local_config = yaml.safe_load(Path("/home/fjk/Bachelorarbeit/GNNs-Code/src/GSAT/src/configs/GIN-mutag.yml").open('r'))
        method_config = local_config["GSAT_config"]
        shared_config = local_config["shared_config"]
        model_config = local_config["model_config"]
        multi_label = True
        
        model = GIN(train_dataset.num_features, 0, train_dataset.num_classes, multi_label, model_config)
        writer = Writer("logs")
        extractor_config = {"learn_edge_att": False, "extractor_dropout_p": 0.5}
        extractor = ExtractorMLP(64, extractor_config)
        device = 'cpu'
        dataset_name = "mutag"
        num_classes = train_dataset.num_classes
        seed = 41
        optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=0)
        
        self.dataset = train_dataset
        self.epochs = epochs
        self.gsat: GSAT = GSAT(model, extractor, optimizer, None, writer, device, "logs", dataset_name, num_classes, multi_label, seed, method_config, shared_config)

    def train(self):
        loader: DataLoader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        for epoch in range(self.epochs):
            train_results = self.gsat.run_one_epoch(loader, epoch, "train", False)
        
    def forward(self, model, x, edge_index, batch,*, target: torch.Tensor, index=None, **kwargs) -> Explanation:

        data = Data(x, edge_index, y=target.float(), batch=batch)
       
        batch_att, _, _ = self.gsat.eval_one_batch(data, 0)
        node_subset = data.batch == 0
        _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

        explanation = Explanation(x, edge_index, node_mask = None, edge_mask = edge_att)
        return explanation

    def supports(self) -> bool:
        return True
