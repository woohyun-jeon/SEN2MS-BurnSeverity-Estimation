import numpy as np

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.ravel()
    targets = targets.ravel()

    cm = confusion_matrix(targets, preds)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    precision = precision_score(targets, preds, average=None, zero_division=0)
    recall = recall_score(targets, preds, average=None, zero_division=0)
    f1 = f1_score(targets, preds, average=None, zero_division=0)
    accuracy = accuracy_score(targets, preds)

    class_metrics = {}
    for i in range(4):
        class_metrics[f'class_{i}'] = {
            'iou': iou[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i]
        }

    return {
        'class_metrics': class_metrics,
        'mean_iou': np.mean(iou),
        'mean_precision': np.mean(precision),
        'mean_recall': np.mean(recall),
        'mean_f1_score': np.mean(f1),
        'accuracy': accuracy
    }