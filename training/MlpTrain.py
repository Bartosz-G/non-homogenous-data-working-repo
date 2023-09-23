import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F








class MlpTrainingRoutine():
    def __init__(self, depth,
                 seed,
                 hidden_dim,
                 regularize,
                 embd_size,
                 optimizer,
                 batch_size,
                 epochs,
                 lr,
                 momentum,
                 no_cuda,
                 lr_step_size,
                 gamma,
                 task,
                 **kwargs):
        self.depth = depth
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.regularize = regularize
        self.embd_size = embd_size
        self.optimizer_str = optimizer
        self.batch_size = batch_size
        self.total_epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.lr_step_size = lr_step_size
        self.gamma = gamma
        self.task = task
        self.kwargs = kwargs
        self.optimizer = None
        self.scheduler = None

        use_cuda = not no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.epoch = 0



    def set_optimizer_scheduler(self, model):
        if self.optimizer_str == 'SGD':
            if self.momentum != 0:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif self.optimizer_str == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, amsgrad=True)
        if self.lr_step_size is None or self.lr_step_size == 0 :
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.gamma)

    def scheduler_step(self, epoch):
        if self.scheduler is not None:
            self.scheduler.step(epoch)

        self.epoch = epoch


    def train(self, model, train_loader):
        model.train()
        dataset_len = 0
        train_loss = 0

        for (data_cont, data_cat, target) in train_loader:
            dataset_len += len(target)
            data_cont, data_cat, target = data_cont.to(self.device), data_cat.to(self.device), target.to(self.device)
            if self.task == 'classification':
                target = target.type(torch.cuda.LongTensor)

            self.optimizer.zero_grad()
            ###############
            data_cont.requires_grad = True
            data_cat.requires_grad = True
            output = model(data_cont, data_cat)
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
            if torch.isnan(loss):
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
