import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_class_weights(dataset):
    class_counts = torch.zeros(4)  # 4 classes for burn severity
    total_pixels = 0

    if hasattr(dataset, 'stack_inputs') and dataset.stack_inputs:
        # single-branch dataset format: (stacked_img, mask, filename)
        for _, mask, _ in dataset:
            mask_np = mask.numpy()
            for i in range(4):
                class_counts[i] += (mask_np == i).sum()
            total_pixels += mask_np.size
    else:
        # dual-branch dataset format: (before_img, after_img, mask, filename)
        for _, _, mask, _ in dataset:
            mask_np = mask.numpy()
            for i in range(4):
                class_counts[i] += (mask_np == i).sum()
            total_pixels += mask_np.size

    class_weights = total_pixels / (4 * class_counts)
    return class_weights / class_weights.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            focal_loss = alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DeepSupervisionLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.75, 0.875, 1.0]):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weights = weights

    def forward(self, outputs, target):
        if isinstance(outputs, list):  # training with deep supervision
            loss = 0
            for output, weight in zip(outputs, self.weights):
                loss += weight * self.criterion(output, target)
            return loss / len(outputs)
        else:  # inference
            return self.criterion(outputs, target)