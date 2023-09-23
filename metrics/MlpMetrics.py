import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torcheval.metrics import BinaryAUROC
from torcheval.metrics import BinaryAccuracy


class ConfusionMatrix:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

        if num_classes:
            self.matrix = torch.zeros((num_classes, num_classes), device='cuda')
        else:
            self.matrix = num_classes

    def update(self, outputs, targets, num_classes=2):
        if self.matrix is None:
            self.num_classes = num_classes
            self.matrix = torch.zeros((num_classes, num_classes), device='cuda')

        with torch.no_grad():
            for t, p in zip(targets.view(-1), outputs.view(-1)):
                self.matrix[t.long(), p.long()] += 1

    def compute(self):
        return self.matrix





class MlpMetricsClassification():
    def __init__(self):
        pass

    # @timer_decorator
    def get_loss(self, model, test_loader, hyperparameters, test_set_name=None):
        if test_set_name:
            assert isinstance(test_set_name, str), "test_set_name must be a string, such as train, val, test"


        no_cuda = hyperparameters['no_cuda']
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        with torch.no_grad():
            model.eval()
            test_loss = 0
            dataset_len = 0

            for (data_cont, data_cat, target) in test_loader:
                dataset_len += len(target)
                data_cont, data_cat, target = data_cont.to(device), data_cat.to(device), target.to(device)

                target = target.type(torch.cuda.LongTensor)


                ###############
                data_cont.requires_grad = True
                data_cat.requires_grad = True
                output = model(data_cont, data_cat)
                ###############

                target_one_dim = torch.argmax(target, dim=1)
                test_loss += F.cross_entropy(output, target_one_dim, reduction='sum').item()

            test_loss /= dataset_len

            if test_set_name:

                loss_name = f'{test_set_name}_loss'
                final_metrics = {loss_name: test_loss}
            else:
                final_metrics = {'loss': test_loss}

            return final_metrics

    def get_metrics(self, model, test_loader, hyperparameters, test_set_name=None):
        if test_set_name:
            assert isinstance(test_set_name, str), "test_set_name must be a string, such as train, val, test"

        no_cuda = hyperparameters['no_cuda']
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        with torch.no_grad():
            model.eval()
            # test_loss = 0
            # dataset_len = 0

            roc_auc_accumulator = BinaryAUROC(num_tasks=2)
            acc_accumulator = BinaryAUROC()
            confusion_accumulator = ConfusionMatrix()

            for (data_cont, data_cat, target) in test_loader:
                #dataset_len += len(target)
                data_cont, data_cat, target = data_cont.to(device), data_cat.to(device), target.to(device)

                target = target.type(torch.cuda.LongTensor)


                ###############
                data_cont.requires_grad = True
                data_cat.requires_grad = True
                output = model(data_cont, data_cat)
                ###############

                target_one_dim = torch.argmax(target, dim=1)
                # test_loss += F.cross_entropy(output, target_one_dim, reduction='sum').item()
                output = torch.softmax(output, dim=-1)

                preds = torch.argmax(output, dim=1).cpu()
                target_one_dim = target_one_dim.cpu()
                output = output.cpu()

                roc_auc_accumulator.update(output.t(), target.t())
                acc_accumulator.update(preds, target_one_dim)
                confusion_accumulator.update(preds, target_one_dim, num_classes=target.shape[1])

            roc_auc = roc_auc_accumulator.compute().mean().item()
            accuracy = acc_accumulator.compute().mean().item()
            confusion_matrix = confusion_accumulator.compute()

            confusion_matrix = [[int(i) for i in row] for row in confusion_matrix.cpu().numpy().tolist()]

            metrics = {
                'accuracy_score': accuracy,
                'roc_auc_score': roc_auc,
                'confusion_matrix': confusion_matrix}

            if test_set_name:
                assert isinstance(test_set_name, str), "test_set_name must be a string, such as 'train', 'val', 'test'"
                metrics_name = f'{test_set_name}_metrics'
                final_metrics = {metrics_name: metrics}
            else:
                final_metrics = {'metrics': metrics}

            return final_metrics

    #@timer_decorator
    def get_all(self, model, test_loader, hyperparameters, test_set_name=None):
        if test_set_name:
            assert isinstance(test_set_name, str), "test_set_name must be a string, such as train, val, test"

        no_cuda = hyperparameters['no_cuda']
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        with torch.no_grad():
            model.eval()
            test_loss = 0
            dataset_len = 0

            roc_auc_accumulator = BinaryAUROC(num_tasks=2)
            acc_accumulator = BinaryAUROC()
            confusion_accumulator = ConfusionMatrix()

            for (data_cont, data_cat, target) in test_loader:
                dataset_len += len(target)
                data_cont, data_cat, target = data_cont.to(device), data_cat.to(device), target.to(device)

                target = target.type(torch.cuda.LongTensor)


                ###############
                data_cont.requires_grad = True
                data_cat.requires_grad = True
                output = model(data_cont, data_cat)
                ###############

                target_one_dim = torch.argmax(target, dim=1)
                test_loss += F.cross_entropy(output, target_one_dim, reduction='sum').item()
                output = torch.softmax(output, dim=-1)

                preds = torch.argmax(output, dim=1).cpu()
                target_one_dim = target_one_dim.cpu()
                output = output.cpu()

                roc_auc_accumulator.update(output.t(), target.t())
                acc_accumulator.update(preds, target_one_dim)
                confusion_accumulator.update(preds, target_one_dim, num_classes=target.shape[1])

            roc_auc = roc_auc_accumulator.compute().mean().item()
            accuracy = acc_accumulator.compute().mean().item()
            confusion_matrix = confusion_accumulator.compute()

            confusion_matrix = [[int(i) for i in row] for row in confusion_matrix.cpu().numpy().tolist()]

            metrics = {
                'accuracy_score': accuracy,
                'roc_auc_score': roc_auc,
                'confusion_matrix': confusion_matrix}

            if test_set_name:
                assert isinstance(test_set_name, str), "test_set_name must be a string, such as 'train', 'val', 'test'"
                loss_name = f'{test_set_name}_loss'
                metrics_name = f'{test_set_name}_metrics'
                final_metrics = {loss_name: test_loss, metrics_name: metrics}
            else:
                final_metrics = {'loss': test_loss, 'metrics': metrics}

            return final_metrics








class MlpMetricsRegression():
    def __init__(self, quantiles=[0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]):
        self.quantiles = quantiles

    # @timer_decorator
    def get_all(self, model, test_loader, hyperparameters, test_set_name=None):
        if test_set_name:
            assert isinstance(test_set_name, str), "test_set_name must be a string, such as train, val, test"

        no_cuda = hyperparameters['no_cuda']
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        with torch.no_grad():
            model.eval()
            test_loss = 0
            dataset_len = 0
            standard_errors_list = []
            target_list = []


            for (data_cont, data_cat, target) in test_loader:
                dataset_len += len(target)
                data_cont, data_cat, target = data_cont.to(device), data_cat.to(device), target.to(device)

                target_list.append(target)

                ###############
                data_cont.requires_grad = True
                data_cat.requires_grad = True
                output = model(data_cont, data_cat)
                ###############

                standard_errors = ((output - target) ** 2)
                standard_errors_list.append(standard_errors)
                test_loss += ((output - target) ** 2).mean().item() * len(target)

            test_loss /= dataset_len

            standard_errors_tensor = torch.cat(standard_errors_list, dim=0)
            all_targets = torch.cat(target_list, dim=0)


            standard_errors_tensor = torch.squeeze(standard_errors_tensor)
            all_targets = torch.squeeze(all_targets)

            ss_res = torch.sum(standard_errors_tensor)
            target_mean = torch.mean(all_targets)
            ss_tot = torch.sum((all_targets - target_mean) ** 2)
            r2_score = 1 - (ss_res / ss_tot)

            RMSE = float(np.sqrt(test_loss))

            quantiles = torch.tensor(self.quantiles, device=device)
            quantile_values = torch.quantile(standard_errors_tensor, quantiles)

            # standard_errors_pd = pd.DataFrame(standard_errors_tensor.cpu().numpy())
            # quantile_dict = standard_errors_pd.quantile(self.quantiles).to_dict()

            quantile_dict = {q: float(v.item()) for q, v in zip(self.quantiles, quantile_values)}

            metrics = {
                'RMSE': RMSE,
                'r2_score': r2_score.cpu().item(),
                'se_quant': quantile_dict}

            if test_set_name:
                assert isinstance(test_set_name, str), "test_set_name must be a string, such as 'train', 'val', 'test'"
                loss_name = f'{test_set_name}_loss'
                metrics_name = f'{test_set_name}_metrics'
                final_metrics = {loss_name: test_loss, metrics_name: metrics}
            else:
                final_metrics = {'loss': test_loss, 'metrics': metrics}

            # return test_loss, test_score
            return final_metrics

    def get_metrics(self, model, test_loader, hyperparameters, test_set_name=None):
        if test_set_name:
            assert isinstance(test_set_name, str), "test_set_name must be a string, such as train, val, test"

        no_cuda = hyperparameters['no_cuda']
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        with torch.no_grad():
            model.eval()
            test_loss = 0
            dataset_len = 0
            standard_errors_list = []
            target_list = []

            for (data_cont, data_cat, target) in test_loader:
                dataset_len += len(target)
                data_cont, data_cat, target = data_cont.to(device), data_cat.to(device), target.to(device)

                target_list.append(target)

                ###############
                data_cont.requires_grad = True
                data_cat.requires_grad = True
                output = model(data_cont, data_cat)
                ###############

                standard_errors = ((output - target) ** 2)
                standard_errors_list.append(standard_errors)
                test_loss += ((output - target) ** 2).mean().item() * len(target)

            test_loss /= dataset_len

            standard_errors_tensor = torch.cat(standard_errors_list, dim=0)
            all_targets = torch.cat(target_list, dim=0)


            standard_errors_tensor = torch.squeeze(standard_errors_tensor)
            all_targets = torch.squeeze(all_targets)

            ss_res = torch.sum(standard_errors_tensor)
            target_mean = torch.mean(all_targets)
            ss_tot = torch.sum((all_targets - target_mean) ** 2)
            r2_score = 1 - (ss_res / ss_tot)

            RMSE = float(np.sqrt(test_loss))

            quantiles = torch.tensor(self.quantiles, device=device)
            quantile_values = torch.quantile(standard_errors_tensor, quantiles)

            # standard_errors_pd = pd.DataFrame(standard_errors_tensor.cpu().numpy())
            # quantile_dict = standard_errors_pd.quantile(self.quantiles).to_dict()

            quantile_dict = {q: float(v.item()) for q, v in zip(self.quantiles, quantile_values)}

            metrics = {
                'RMSE': RMSE,
                'r2_score': r2_score.cpu().item(),
                'se_quant': quantile_dict}

            if test_set_name:
                assert isinstance(test_set_name, str), "test_set_name must be a string, such as 'train', 'val', 'test'"
                metrics_name = f'{test_set_name}_metrics'
                final_metrics = {metrics_name: metrics}
            else:
                final_metrics = {'metrics': metrics}


            return final_metrics
