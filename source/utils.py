import torch
import os
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, last_iter=-1):
        self.warmup_iters = warmup_iters
        self.current_iter = 0 if last_iter == -1 else last_iter
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            return [
                base_lr * (self.current_iter + 1) / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]

    def step(self, it=None):
        if it is None:
            it = self.current_iter + 1
        self.current_iter = it
        super(LinearWarmupScheduler, self).step(it)
        
def compute_weighted_metrics(preds, gts):

    num_classes = len(torch.unique(gts))
    device = preds.device

    class_counts = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        class_counts[cls] = (gts == cls).sum()

    correct = (preds == gts).sum().item()
    acc = correct / gts.size(0)

    precision_list, recall_list, f1_list = [], [], []
    for cls in range(num_classes):
        tp = ((preds == cls) & (gts == cls)).sum().item()
        fp = ((preds == cls) & (gts != cls)).sum().item()
        fn = ((preds != cls) & (gts == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        weight = class_counts[cls] / class_counts.sum()
        precision_list.append(precision * weight)
        recall_list.append(recall * weight)
        f1_list.append(f1 * weight)

    pre = sum(precision_list).item()
    recall = sum(recall_list).item()
    f1 = sum(f1_list).item()

    return acc, pre, recall, f1
