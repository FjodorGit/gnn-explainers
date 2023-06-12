from classifiers.abstract_classifier import GraphClassifier
from classifiers.criterion import Criterion
from logs.trainingLogger import TrainingLogger
import torch
import numpy as np
from tqdm import tqdm

import sys

class ClassifierTrainer():

    def __init__(self, logPath: str):
        self.logger = TrainingLogger(logPath)
    
    def get_preds(self, logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def eval_one_batch(self, data, model, criterion, optimizer=None):
        assert optimizer is None
        model.eval()
        logits = model(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        targets = torch.reshape(data.y.data, logits.shape)
        loss = criterion(logits, targets)
        return loss.item(), logits.data, targets

    def train_one_batch(self, data, model, criterion, optimizer):
        model.train()

        logits = model(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        targets = torch.reshape(data.y.data, logits.shape)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), logits.data, targets

    def run_one_epoch(self, data_loader, model, criterion, optimizer, epoch, phase, multi_label):
        
        all_preds, all_targets, all_batch_losses, all_logits = [], [], [], []

        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        for data in data_loader:
            loss, logits, targets = run_one_batch(data, model, criterion, optimizer)
            preds = self.get_preds(logits, multi_label)

            acc = 0 if multi_label else (preds == targets).sum().item() / targets.shape[0]
            all_preds.append(preds), all_targets.append(targets), all_batch_losses.append(loss), all_logits.append(logits)

        all_preds, all_targets, all_logits = np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_logits)
        all_acc = (all_preds == all_targets).sum() / (all_targets.shape[0] * all_targets.shape[1]) if multi_label else (all_preds == all_targets).sum().item() / all_targets.shape[0]

        log_str = f'[Epoch: {epoch}]: Phase {phase} | Loss: {np.mean(all_batch_losses):.3f} | Acc: {all_acc:.3f}'
        self.logger.log(log_str)

        return all_acc, np.mean(all_batch_losses)

    def train(self, model: GraphClassifier, loaders, dataset, model_config):
        
        multi_label = model_config['multi_label']
        lr = model_config['pretrain_lr']
        epochs = model_config['pretrain_epochs']
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = Criterion(dataset.num_classes, multi_label)
        
        for epoch in range(epochs):
            train_res, _ = self.run_one_epoch(loaders['train'], model, criterion, optimizer, epoch, 'train', multi_label)
            valid_res, valid_loss = self.run_one_epoch(loaders['valid'], model, criterion, None, epoch, 'valid', multi_label)
            test_res, _ = self.run_one_epoch(loaders['test'], model, criterion, None, epoch, 'test', multi_label)
            
    def quick_eval_model(self, model: GraphClassifier, dataset, test_loader, model_config):
        multi_label = model_config['multi_label']
        criterion = Criterion(dataset.num_classes, multi_label)
        all_preds, all_targets, all_batch_losses, all_logits = [], [], [], []

        for data in test_loader:
            loss, logits, targets = self.eval_one_batch(data, model, criterion, None)
            preds = self.get_preds(logits, multi_label)

            all_preds.append(preds)
            all_targets.append(targets)
            all_batch_losses.append(loss)
            all_logits.append(logits)

        all_preds, all_targets, all_logits = np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_logits)
        all_acc = (all_preds == all_targets).sum() / (all_targets.shape[0] * all_targets.shape[1]) if multi_label else (all_preds == all_targets).sum().item() / all_targets.shape[0]

        self.logger.log(f'[Evaluating Model]: Classifier accuracy: {all_acc:.3f}')

        return all_acc, np.mean(all_batch_losses)
