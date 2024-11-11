import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn

from fedml_api.dpfedsam.wsam import WeightedSAM, enable_running_stats, disable_running_stats  # 引入 WeightedSAM

from fedml_api.model.cv.cnn_meta import Meta_net

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger

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
        model = self.model
        model.to(device)
        model.train()
        metrics = {
            'train_correct': 0,
            'train_loss': 0,
            'train_total': 0
        }

        # 损失函数
        criterion = nn.CrossEntropyLoss().to(device)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)

        # 初始化 WeightedSAM 优化器
        base_optimizer = torch.optim.SGD
        optimizer = WeightedSAM(
            model=model.parameters(),
            base_optimizer=base_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr * (
                    args.lr_decay ** round), momentum=args.momentum, weight_decay=args.wd),
            rho=args.rho,
            gamma=0.9,  # 可以根据需求调整
            adaptive=args.adaptive
        )

        for epoch in range(args.epochs):
            epoch_loss, epoch_acc = [], []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                # 第一轮前向-后向传播
                pred = model(x)
                enable_running_stats(model)
                loss = criterion(pred, labels.long())
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # 第二轮前向-后向传播
                disable_running_stats(model)
                criterion(model(x), labels.long()).backward()
                optimizer.second_step(zero_grad=True)
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                # 记录损失和正确率
                epoch_loss.append(loss.item())
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(labels).sum()
                epoch_acc.append(correct.item())

                metrics['train_correct'] += correct.item()
                metrics['train_loss'] += loss.item() * labels.size(0)
                metrics['train_total'] += labels.size(0)

            print(f'Client Index = {self.id}\tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f}')

        return metrics

    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
