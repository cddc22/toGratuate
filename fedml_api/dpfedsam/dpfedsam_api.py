import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler  # 更新导入语句
from torch.utils.data import DataLoader
from fedml_api.dpfedsam.client import Client

class DPFedSAMAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        
        cudnn.benchmark = True
        cudnn.deterministic = False
        
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
         
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self, exper_index):
        # 更新 GradScaler 初始化
        scaler = GradScaler('cuda')
        
        w_global = self.model_trainer.get_model_params()
        # 确保全局模型参数在GPU上
        w_global = {k: v.to(self.device) for k,v in w_global.items()}
        nabala_w_global = {k: torch.zeros_like(v, device=self.device) for k,v in w_global.items()}

        self.logger.info("################Exper times: {}".format(exper_index))
        
        for round_idx in range(self.args.comm_round):
            w_locals = []
            nabala_w = []
            last_w_global = {k: v.clone() for k,v in w_global.items()}

            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            loss_locals, acc_locals, total_locals = [], [], []
            norm_list = []

            for cur_clnt in client_indexes:
                client = self.client_list[cur_clnt]
                
                # 更新 autocast 使用
                with autocast(device_type='cuda'):
                    w_per, training_flops, num_comm_params, metrics = client.train(
                        {k: v.clone() for k,v in w_global.items()}, 
                        round_idx
                    )
                
                # 确保 w_per 在GPU上
                w_per = {k: v.to(self.device) for k,v in w_per.items()}
                
                # 计算本地更新
                nabala = {k: w_per[k] - w_global[k] for k in w_per.keys()}
                
                norm = 0.0
                for name in nabala.keys():
                    norm += pow(nabala[name].norm(2), 2)
                    
                    noise = torch.normal(
                    0, 
                    self.args.sigma * self.args.C /np.sqrt(self.args.client_num_per_round),
                    nabala[name].shape,
                    device=self.device  # 确保噪声直接在GPU上生成
                )
                
                # 裁剪和添加噪声
                current_norm = torch.norm(nabala[name], 2)
                nabala[name] *= min(1, self.args.C/current_norm)
                nabala[name].add_(noise)

            total_norm = torch.sqrt(norm).cpu().numpy().reshape(1)
            norm_list.append(total_norm[0])

            # 更新本地模型
            w_per = {k: w_global[k] + nabala[k] for k in w_global.keys()}
            
            w_locals.append((client.get_sample_number(), w_per))
            nabala_w.append((client.get_sample_number(), nabala))
            
            self.stat_info["sum_training_flops"] += training_flops
            self.stat_info["sum_comm_params"] += num_comm_params
            loss_locals.append(metrics['train_loss'])
            acc_locals.append(metrics['train_correct']) 
            total_locals.append(metrics['train_total'])

        self.stat_info["local_norm"].append(norm_list)
        global_norm = sum(norm_list)/len(norm_list)
        self.stat_info["global_norm"].append(global_norm)

        self._train_on_sample_clients(loss_locals, acc_locals, total_locals, round_idx, len(client_indexes))
        
        # 聚合并更新全局模型
        nabala_w_global = self._aggregate(nabala_w)
        w_global = {k: last_w_global[k] + nabala_w_global[k] for k in w_global.keys()}

        # 测试
        self._test_on_all_clients(w_global, round_idx)

        # 定期保存和输出结果
        if round_idx % 50 == 0 or round_idx == self.args.comm_round - 1:
            self._save_and_print_results(round_idx, exper_index, w_global)
            
        # 定期清理GPU缓存
        if round_idx % 10 == 0:
            torch.cuda.empty_cache()

    def _aggregate(self, w_locals):
        training_num = sum(sample_num for sample_num, _ in w_locals)
        
        w_global = {}
        for k in w_locals[0][1].keys():
            w_global[k] = torch.zeros_like(w_locals[0][1][k], device=self.device)
            for sample_num, local_params in w_locals:
                w = sample_num / training_num
                w_global[k] += local_params[k] * w
        return w_global

    def _train_on_sample_clients(self, loss_locals, acc_locals, total_locals, round_idx, client_sample_number):
        self.logger.info("################global_train_on_all_clients : {}".format(round_idx))

        # 确保计算在同一设备上进行
        acc_locals = torch.tensor(acc_locals, device=self.device)
        total_locals = torch.tensor(total_locals, device=self.device)
        loss_locals = torch.tensor(loss_locals, device=self.device)

        g_train_acc = torch.mean(acc_locals/total_locals)
        g_train_loss = torch.mean(loss_locals/total_locals)

        print('The averaged global_train_acc:{}, global_train_loss:{}'.format(g_train_acc, g_train_loss))
        stats = {'The averaged global_train_acc': g_train_acc.item(), 'global_train_loss': g_train_loss.item()}
        self.stat_info["global_train_acc"].append(g_train_acc.item())
        self.stat_info["global_train_loss"].append(g_train_loss.item())
        self.logger.info(stats)

    def _test_on_all_clients(self, w_global, round_idx):
        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))
        
        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            client = self.client_list[client_idx]
            # 确保测试时模型参数在正确的设备上
            test_w_global = {k: v.to(self.device) for k,v in w_global.items()}
            g_test_local_metrics = client.local_test(test_w_global, True)
            
            g_test_metrics['num_samples'].append(g_test_local_metrics['test_total'])
            g_test_metrics['num_correct'].append(g_test_local_metrics['test_correct'])
            g_test_metrics['losses'].append(g_test_local_metrics['test_loss'])

            if self.args.ci == 1:
                break

        # 转换为tensor并移到GPU计算
        samples = torch.tensor(g_test_metrics['num_samples'], device=self.device)
        correct = torch.tensor(g_test_metrics['num_correct'], device=self.device)
        losses = torch.tensor(g_test_metrics['losses'], device=self.device)

        g_test_acc = torch.mean(correct/samples)
        g_test_loss = torch.mean(losses/samples)

        stats = {'global_test_acc': g_test_acc.item(), 'global_test_loss': g_test_loss.item()}
        self.stat_info["global_test_acc"].append(g_test_acc.item())
        self.stat_info["global_test_loss"].append(g_test_loss.item())
        self.logger.info(stats)

    def _save_and_print_results(self, round_idx, exper_index, w_global):
        print('global_train_loss={}'.format(self.stat_info["global_train_loss"]))
        print('global_train_acc={}'.format(self.stat_info["global_train_acc"]))
        print('global_test_loss={}'.format(self.stat_info["global_test_loss"]))
        print('global_test_acc={}'.format(self.stat_info["global_test_acc"]))
        
        self.logger.info("################Communication round : {}".format(round_idx))
        
        if round_idx % 200 == 0 or round_idx == self.args.comm_round-1:
            self.logger.info("################The final results, Experiment times: {}".format(exper_index))
            
            # 保存本地范数
            save_path = os.path.join("./LOG/cifar10/dumps", f"local_norm_dpfedsam_{self.args.p}_.dat")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.array(self.stat_info["local_norm"]).dump(save_path)
            
            # 保存模型
            if self.args.dataset == "cifar10":
                model = customized_resnet18(10)
                model.to(self.device)
                model.load_state_dict(w_global)
                
                save_dir = f"{os.getcwd()}/save_model"
                os.makedirs(save_dir, exist_ok=True)
                
                if self.args.spar_rand:
                        model_path = os.path.join(save_dir, 
                            f"dp-fedsam_threshold{self.args.C}_rho{self.args.rho}_spar_rand_p{self.args.p}.pth.tar")
                else:
                        model_path = os.path.join(save_dir,
                            f"dp-fedsam_threshold{self.args.C}_rho{self.args.rho}_spar_topk_p{self.args.p}.pth.tar")
                        
                torch.save(model, model_path)

            self.logger.info('local_norm = {}'.format(self.stat_info["local_norm"]))
            self.logger.info('global_norm = {}'.format(self.stat_info["global_norm"]))
            self.logger.info('global_train_loss={}'.format(self.stat_info["global_train_loss"]))
            self.logger.info('global_train_acc={}'.format(self.stat_info["global_train_acc"]))
            self.logger.info('global_test_loss={}'.format(self.stat_info["global_test_loss"]))
            self.logger.info('global_test_acc={}'.format(self.stat_info["global_test_acc"]))

    def init_stat_info(self):
        self.stat_info = {
            "sum_training_flops": 0,
            "sum_comm_params": 0,
            "global_train_loss": [],
            "global_train_acc": [],
            "global_test_loss": [],
            "global_test_acc": [],
            "local_norm": [],
            "global_norm": []
        }

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = list(range(client_num_in_total))
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    @torch.no_grad()
    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        # 确保w_global在正确的设备上
        w_global = {k: v.to(self.device) for k,v in w_global.items()}
        
        for client_idx in range(self.args.client_num_in_total):
            if mask_pers is None:
                inference_flops.append(self.model_trainer.count_inference_flops(w_global))
            else:
                w_per = {
                    name: w_global[name] * mask_pers[client_idx][name].to(self.device)
                    for name in mask_pers[client_idx]
                }
                inference_flops.append(self.model_trainer.count_inference_flops(w_per))
                
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        return avg_inference_flops

    def _to_device(self, data):
        """Helper function to move data to device"""
        if isinstance(data, (list, tuple)):
            return [self._to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)

    def _clean_gpu_memory(self):
        """Helper function to clean GPU memory"""
        torch.cuda.empty_cache()
        
    def get_stat_info(self):
        """Return training statistics"""
        return self.stat_info
