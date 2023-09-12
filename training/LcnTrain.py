import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score






# To be used as args parser from orchestrator
class Hyperparams:
    def __init__(self, depth, seed, drop_type, p, ensemble_n, shrinkage, back_n, net_type, hidden_dim, anneal,
                 optimizer, batch_size, epochs, lr, momentum, no_cuda, lr_step_size, gamma, task):
        self.depth = depth
        self.seed = seed
        self.drop_type = drop_type
        self.p = p
        self.ensemble_n = ensemble_n
        self.shrinkage = shrinkage
        self.back_n = back_n
        self.net_type = net_type
        self.hidden_dim = hidden_dim
        self.anneal = anneal
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.lr_step_size = lr_step_size
        self.gamma = gamma
        self.task = task
        self.input_dim = None
        self.output_dim = None

    def __bool__(self):
        return True

    def set_input_dims(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim



def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)


class LcnTrainingRoutine():
    def __init__(self, depth, seed, drop_type, p, ensemble_n, shrinkage, back_n, net_type, hidden_dim, anneal,
                 optimizer, batch_size, epochs, lr, momentum, no_cuda, lr_step_size, gamma, task, input_dim, output_dim):
        self.depth = depth
        self.seed = seed
        self.drop_type = drop_type
        self.p = p
        self.ensemble_n = ensemble_n
        self.shrinkage = shrinkage
        self.back_n = back_n
        self.net_type = net_type
        self.hidden_dim = hidden_dim
        self.anneal = anneal
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.total_epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.lr_step_size = lr_step_size
        self.gamma = gamma
        self.task = task
        self.optimizer_obj = None
        self.scheduler = None

        use_cuda = not no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.epoch = None



    def get_optimizer_scheduler(self, model):
        if self.optimizer == 'SGD':
            self.optimizer_obj = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True)
        elif self.optimizer == 'AMSGrad':
            self.optimizer_obj = torch.optim.Adam(model.parameters(), lr=self.lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_obj, step_size=self.lr_step_size, gamma=self.gamma)

    def scheduler_step(self, epoch):
        if self.optimizer is None:
            self.get_optimizer_scheduler()

        if self.scheduler is not None:
            self.scheulder.step(epoch)

        self.epoch = epoch


    def train(self, model, train_loader):
        alpha = get_alpha(self.epoch, self.epochs)
        model.train()
        dataset_len = 0
        train_loss = 0

        for (data, target) in train_loader:
            dataset_len += len(target)
            data, target = data.to(self.device), target.to(self.device)
            if self.task == 'classification':
                target = target.type(torch.cuda.LongTensor)

            self.optimizer.zero_grad()
            ###############
            data.requires_grad = True
            if model.net_type == 'locally_constant':
                if self.p != -1:
                    assert (self.p >= 0. and self.p < 1)
                    output, regularization = model(data, alpha=alpha, anneal=self.anneal, p=self.p, training=True)
                else:
                    output, regularization = model(data, alpha=alpha, anneal=self.anneal, p=1 - alpha, training=True)

            elif model.net_type == 'locally_linear':
                output, regularization = model.normal_forward(data)
            ###############

            self.optimizer.zero_grad()
            if self.task == 'classification':
                # Added: Bart
                target_one_dim = torch.argmax(target, dim=1)
                loss = F.cross_entropy(output, target_one_dim)
            elif self.task == 'regression':
                output = output.squeeze(-1)
                loss = ((output - target) ** 2).mean()

            loss.backward()
            self.optimizer.step()

            # NaN early stopping
            if torch.isnan(loss.item()):
                train_loss = None
                break

            if self.task == 'classification':
                train_loss += loss.item()
            elif self.task == 'regression':
                train_loss += loss.item() * len(target)
        else:
            train_loss /= dataset_len

            final_metric = {'train_loss': train_loss}
            return final_metric

        return train_loss


def train(args, model, device, train_loader, optimizer, task, anneal, alpha=1):
    model.train()
    dataset_len = 0
    train_loss = 0

    for (data, target) in train_loader:
        dataset_len += len(target)
        data, target = data.to(device), target.to(device)
        if args.task == 'classification':
            target = target.type(torch.cuda.LongTensor)

        optimizer.zero_grad()
        ###############
        data.requires_grad = True
        if model.net_type == 'locally_constant':
            if args.p != -1:
                assert (args.p >= 0. and args.p < 1)
                output, regularization = model(data, alpha=alpha, anneal=anneal, p=args.p, training=True)
            else:
                output, regularization = model(data, alpha=alpha, anneal=anneal, p=1 - alpha, training=True)

        elif model.net_type == 'locally_linear':
            output, regularization = model.normal_forward(data)
        ###############

        optimizer.zero_grad()
        if args.task == 'classification':
            # Added: Bart
            target_one_dim = torch.argmax(target, dim=1)
            loss = F.cross_entropy(output, target_one_dim)
        elif args.task == 'regression':
            output = output.squeeze(-1)
            loss = ((output - target) ** 2).mean()

        loss.backward()
        optimizer.step()

        # NaN early stopping
        if torch.isnan(loss.item()):
            train_loss = None
            break

        if args.task == 'classification':
            train_loss += loss.item()
        elif args.task == 'regression':
            train_loss += loss.item() * len(target)
    else:
        train_loss /= dataset_len

        final_metric = {'train_loss': train_loss}
        return final_metric

    return train_loss






# ================== Depreciated ============================
# Modified from the original paper
# def get_metrics(args, model, device, test_loader, metrics_func, test_set_name):
#     with torch.no_grad():
#         model.eval()
#
#
#         data, target = next(iter(test_loader))
#
#         data, target = data.to(device), target.to(device)
#         if args.task == 'classification':
#             target = target.type(torch.cuda.LongTensor)
#
#         ###############
#         data.requires_grad = True
#         if model.net_type == 'locally_constant':
#             output, relu_masks = model(data, p=0, training=False)
#         elif model.net_type == 'locally_linear':
#             output, relu_masks = model.normal_forward(data, p=0, training=False)
#         ###############
#
#         if args.task == 'classification':
#             output = torch.softmax(output, dim=-1)
#             metrics = metrics_func(target, output, True)
#         elif args.task == 'regression':
#             metrics = metrics_func(target, output, False)
#
#         return metrics



# def get_loss(args, model, device, test_loader, test_set_name):
#     with torch.no_grad():
#         model.eval()
#         test_loss = 0
#         correct = 0
#
#         score = []
#         label = []
#         dataset_len = 0
#
#         pattern_to_pred = dict()
#         tree_x = []
#         tree_pattern = []
#
#         for data, target in test_loader:
#             dataset_len += len(target)
#             label += list(target)
#             data, target = data.to(device), target.to(device)
#             if args.task == 'classification':
#                 target = target.type(torch.cuda.LongTensor)
#
#             ###############
#             data.requires_grad = True
#             if model.net_type == 'locally_constant':
#                 output, relu_masks = model(data, p=0, training=False)
#             elif model.net_type == 'locally_linear':
#                 output, relu_masks = model.normal_forward(data, p=0, training=False)
#             ###############
#
#             if args.task == 'classification':
#                 # Modified: Bart
#                 target_one_dim = torch.argmax(target, dim=1)
#                 test_loss += F.cross_entropy(output, target_one_dim, reduction='sum').item()
#                 # Removed: Bart
#                 # output = torch.softmax(output, dim=-1)
#                 # ...
#                 # output = output[:, 1]
#             elif args.task == 'regression':
#                 output = output.squeeze(-1)
#                 test_loss += ((output - target) ** 2).mean().item() * len(target)
#
#         test_loss /= dataset_len
#
#         # Removed: Bart
#         # if args.task == 'classification':
#         # ...
#
#         return test_loss
