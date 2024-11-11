import copy
import logging
import numpy as np
import torch
from torch import nn
#from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast


from fedml_api.dpfedsam.wsam import WeightedSAM, enable_running_stats, disable_running_stats
from fedml_api.model.cv.cnn_meta import Meta_net

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args or {}
        self.logger = logger
        self.device = args.device if hasattr(args, 'device') else 'cpu'
        self.scaler = GradScaler() if self.device == 'cuda' else None  # 混合精度训练器
        self.init_optimizer(0)  # 初始时round设为0

    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def init_optimizer(self, round):
        base_optimizer = torch.optim.SGD
        self.optimizer = WeightedSAM(
            model=self.model,
            base_optimizer=base_optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum,
                weight_decay=self.args.wd
            ),
            rho=self.args.rho,
            gamma=getattr(self.args, "gamma", 0.9),
            adaptive=self.args.adaptive
        )

    def set_masks(self, masks):
        self.masks = masks

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        params_dict = {}
        for name, param in self.model.named_parameters():
            params_dict[name] = param
        return params_dict

    def train(self, train_data, device, args, round):
        self.init_optimizer(round)  # 更新优化器
        model = self.model.to(device)
        metrics = {'train_correct': 0, 'train_loss': 0, 'train_total': 0}

        for epoch in range(args.epochs):
            epoch_metrics = self.train_epoch(train_data, device)
            metrics['train_correct'] += epoch_metrics['correct']
            metrics['train_loss'] += epoch_metrics['loss']
            metrics['train_total'] += epoch_metrics['total']
            self.log(f'Client Index = {self.id}\tEpoch: {epoch}\tLoss: {epoch_metrics["loss"] / epoch_metrics["total"]:.6f}')

        return metrics

    def train_epoch(self, train_data, device):
        metrics = {'correct': 0, 'loss': 0, 'total': 0}
        for batch_idx, (x, labels) in enumerate(train_data):
            batch_metrics = self.train_batch(x, labels, device)
            metrics['correct'] += batch_metrics['correct']
            metrics['loss'] += batch_metrics['loss']
            metrics['total'] += batch_metrics['total']
        return metrics

    def train_batch(self, x, labels, device):
        x, labels = x.to(device), labels.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        with autocast('cuda',enabled=bool(self.scaler)):  # 使用混合精度
            pred = self.model(x)
            if torch.isnan(pred).any():
                print("Model output contains NaN values.")
                return {'correct': 0, 'loss': 0, 'total': 0}
            loss = criterion(pred, labels.long())

        # 第一次优化步骤
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 加入梯度裁剪



        self.optimizer.first_step(zero_grad=True)

        # 第二次优化步骤
        with autocast('cuda',enabled=bool(self.scaler)):
            second_loss = criterion(self.model(x), labels.long())

        self.scaler.scale(second_loss).backward() if self.scaler else second_loss.backward()
        self.optimizer.second_step(zero_grad=True)

        # 更新缩放器
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        _, predicted = torch.max(pred, -1)
        correct = predicted.eq(labels).sum().item()
        return {'correct': correct, 'loss': loss.item() * labels.size(0), 'total': labels.size(0)}

    def test(self, test_data, device, args):
        model = self.model.to(device)
        model.eval()
        metrics = {'test_correct': 0, 'test_loss': 0, 'test_total': 0}
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                batch_metrics = self.evaluate_batch(x, target, device, criterion)
                metrics['test_correct'] += batch_metrics['correct']
                metrics['test_loss'] += batch_metrics['loss']
                metrics['test_total'] += batch_metrics['total']

        return metrics

    def evaluate_batch(self, x, target, device, criterion):
        x, target = x.to(device), target.to(device)
        with autocast('cuda',enabled=bool(self.scaler)):
            pred = self.model(x)
            loss = criterion(pred, target.long())

        _, predicted = torch.max(pred, -1)
        correct = predicted.eq(target).sum().item()
        return {'correct': correct, 'loss': loss.item() * target.size(0), 'total': target.size(0)}

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
