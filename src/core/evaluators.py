import torch
import numpy as np
from sklearn.metrics import r2_score

class Evaluator(object):

    def __init__(self, logger, device):
        self.logger = logger
        self.device = device
        self.reset()

    def reset(self):
        pass

    def update(self, y_pred, y_true):
        pass

    def compute(self):
        pass

    def print_log(self, epoch, num_steps):
        pass


class R2Evaluator(Evaluator):

    def reset(self):
        self.r2_score = 0.
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, y_pred, y_true, video_latent=False):
        if video_latent:
            y_pred = y_pred[0].mean.squeeze().mean(dim=0).mean(dim=0).detach().cpu().numpy()
        else:
            y_pred = y_pred[0].mean.squeeze().mean(dim=0).detach().cpu().numpy()
        self.y_pred = np.concatenate((self.y_pred, y_pred), axis=0) if self.y_pred.size else y_pred

        y_true = y_true.squeeze().detach().cpu().numpy()
        self.y_true = np.concatenate((self.y_true, y_true), axis=0) if self.y_true.size else y_true

    def compute(self):
        self.r2_score = r2_score(y_true=self.y_true, y_pred=self.y_pred)
        return self.r2_score

class MaeEvaluator(Evaluator):

    def reset(self):
        self.mae_score = 0.
        self.y_pred = torch.tensor([], device=self.device)
        self.y_true = torch.tensor([], device=self.device)

    def update(self, y_pred, y_true, video_latent=False):
        if video_latent:
            y_pred = y_pred[0].mean.squeeze().mean(dim=0).mean(dim=0)
        else:
            y_pred = y_pred[0].mean.squeeze().mean(dim=0)
        self.y_pred = torch.cat((self.y_pred, y_pred), dim=0) if self.y_pred.size else y_pred

        y_true = y_true.squeeze()
        self.y_true = torch.cat((self.y_true, y_true), dim=0) if self.y_true.size else y_true

    def compute(self):
        self.mae_scoe = torch.nn.functional.l1_loss(self.y_pred, self.y_true)
        return self.mae_scoe.item()



#class AccEvaluator(Evaluator):
#
#    def reset(self):
#        self.acc = 0
#        self.num_total = 0.
#        self.num_correct = 0.
#
#    def update(self, output_dict):
#        logits = output_dict['logits']
#        labels = output_dict['labels']
#        accuracy = topk_accuracy(logits, labels, topk=(1,))[0]
#
#        batch_size = logits.shape[0]
#        self.num_correct += accuracy * batch_size
#        self.num_total += batch_size
#
#    def compute(self):
#        self.acc = self.num_correct / float(self.num_total)
#        return self.acc
#
#
#
#def topk_accuracy(outputs, labels, topk=(1,)):
#    """Computes the accuracy for the top k predictions"""
#    with torch.no_grad():
#        maxk = max(topk)
#        batch_size = labels.size(0)
#
#        _, pred = torch.topk(outputs, k=maxk, dim=1, largest=True, sorted=True)
#        pred = pred.t()
#        correct = pred.eq(labels.view(1, -1).expand_as(pred))
#
#        topk_accuracies = []
#        for k in topk:
#            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
#            topk_accuracies.append(correct_k.mul_(1.0 / batch_size).item())
#        return topk_accuracies

