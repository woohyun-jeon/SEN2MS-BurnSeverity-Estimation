import os
from tqdm import tqdm
import numpy as np
import warnings
import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from datasets import get_siamese_datasets, get_semseg_datasets
from models import UNet, NestedUNet, SwinUNet, TransUNet, BiUNet, SNUNet, ChangeFormer, TransFireNet
from metrics import calculate_metrics
from utils import set_seed, load_config, load_dataset_ids, EarlyStopping
from loss import calculate_class_weights, FocalLoss, DeepSupervisionLoss


def train_siamese_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, patience, min_delta):
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_model_state = None
    best_val_metrics = None

    for epoch in tqdm(range(num_epochs), desc='Siamese training'):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for pre_fire, post_fire, labels, _ in dataloaders['train']:
            pre_fire, post_fire, labels = pre_fire.to(device), post_fire.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(pre_fire, post_fire)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * pre_fire.size(0)

            # SNUNet returns list during training
            if isinstance(outputs, list):
                # use the final output (outputs[-1]) for predictions
                train_preds.append(torch.argmax(outputs[-1], dim=1))
            else:
                train_preds.append(torch.argmax(outputs, dim=1))
            train_targets.append(labels)

        train_metrics = evaluate_siamese_model(model, dataloaders['train'], criterion, device)
        val_metrics = evaluate_siamese_model(model, dataloaders['val'], criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}: ")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['mean_iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['mean_iou']:.4f}")

        scheduler.step()

        if best_val_metrics is None or val_metrics['mean_iou'] > best_val_metrics['mean_iou']:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()

        if early_stopping(val_metrics['mean_iou']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_model_state, best_val_metrics


def train_semseg_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, patience, min_delta):
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_model_state = None
    best_val_metrics = None

    for epoch in tqdm(range(num_epochs), desc='Semantic segmentation training'):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for images, labels, _ in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            # for SNUNet, use the final output
            if isinstance(outputs, list):
                train_preds.append(torch.argmax(outputs[-1], dim=1))
            else:
                train_preds.append(torch.argmax(outputs, dim=1))
            train_targets.append(labels)

        train_metrics = evaluate_semseg_model(model, dataloaders['train'], criterion, device)
        val_metrics = evaluate_semseg_model(model, dataloaders['val'], criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}: ")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['mean_iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['mean_iou']:.4f}")

        scheduler.step()

        if best_val_metrics is None or val_metrics['mean_iou'] > best_val_metrics['mean_iou']:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()

        if early_stopping(val_metrics['mean_iou']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_model_state, best_val_metrics


def evaluate_siamese_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for pre_fire, post_fire, labels, _ in dataloader:
            pre_fire, post_fire, labels = pre_fire.to(device), post_fire.to(device), labels.to(device)
            outputs = model(pre_fire, post_fire)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * pre_fire.size(0)
            all_preds.append(torch.argmax(outputs, dim=1))
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader.dataset)

    return metrics


def evaluate_semseg_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # for SNUNet, use the final output
            if isinstance(outputs, list):
                all_preds.append(torch.argmax(outputs[-1], dim=1))
            else:
                all_preds.append(torch.argmax(outputs, dim=1))
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader.dataset)

    return metrics


def save_predictions(model, dataloader, device, save_dir, mode='siamese'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        if mode == 'siamese':
            for pre_fire, post_fire, _, filenames in dataloader:
                pre_fire, post_fire = pre_fire.to(device), post_fire.to(device)
                outputs = model(pre_fire, post_fire)

                # for SNUNet, use the final output (outputs[-1])
                if isinstance(outputs, list):
                    pred_output = outputs[-1]
                else:
                    pred_output = outputs

                predictions = torch.argmax(pred_output, dim=1).cpu().numpy()

                for pred, filename in zip(predictions, filenames):
                    save_path = os.path.join(save_dir, filename)
                    pred = pred.astype(np.uint8)
                    with rasterio.open(save_path, 'w', driver='GTiff', height=pred.shape[0], width=pred.shape[1], count=1, dtype='uint8') as dst:
                        dst.write(pred, 1)
        else:
            for images, _, filenames in dataloader:
                images = images.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                for pred, filename in zip(predictions, filenames):
                    save_path = os.path.join(save_dir, filename)
                    pred = pred.astype(np.uint8)
                    with rasterio.open(save_path, 'w', driver='GTiff', height=pred.shape[0], width=pred.shape[1], count=1, dtype='uint8') as dst:
                        dst.write(pred, 1)


def main():
    config_path = 'configs.yaml'
    cfgs = load_config(config_path)
    set_seed(seed=cfgs['params']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset IDs
    train_ids = load_dataset_ids(os.path.join(cfgs['path']['data_path'], 'train.txt'))
    val_ids = load_dataset_ids(os.path.join(cfgs['path']['data_path'], 'valid.txt'))
    test_ids = load_dataset_ids(os.path.join(cfgs['path']['data_path'], 'test.txt'))

    # define models
    models = {
        'stack_UNet': {'class': UNet, 'mode': 'semseg', 'in_channels': 6},
        'stack_NestedUNet': {'class': NestedUNet, 'mode': 'semseg', 'in_channels': 6},
        'stack_SwinUNet': {'class': SwinUNet, 'mode': 'semseg', 'in_channels': 6},
        'stack_TransUNet': {'class': TransUNet, 'mode': 'semseg', 'in_channels': 6},
        'siamese_BiUNet': {'class': BiUNet, 'mode': 'siamese', 'in_channels': 3},
        'siamese_SNUNet': {'class': SNUNet, 'mode': 'siamese', 'in_channels': 3},
        'siamese_ChangeFormer': {'class': ChangeFormer, 'mode': 'siamese', 'in_channels': 3},
        'siamese_TransFireNet': {'class': TransFireNet, 'mode': 'siamese', 'in_channels': 3}
    }

    results = {}

    for model_name, model_info in models.items():
        print(f"\nTraining and evaluating {model_name}")

        # get datasets
        if model_info['mode'] == 'siamese':
            train_dataset, val_dataset, test_dataset = get_siamese_datasets(
                cfgs['path']['data_path'], train_ids, val_ids, test_ids
            )
        else:
            train_dataset, val_dataset, test_dataset = get_semseg_datasets(
                cfgs['path']['data_path'], train_ids, val_ids, test_ids
            )

        # set dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfgs['params']['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=4)

        dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        # set model
        model = model_info['class'](
            in_channels=model_info['in_channels'],
            num_classes=4,
            pretrained=True
        ).to(device)

        # set loss
        class_weights = calculate_class_weights(train_loader.dataset)
        if isinstance(model, (SNUNet, NestedUNet)):
            criterion = DeepSupervisionLoss(weights=[0.5, 0.75, 0.875, 1.0])
        else:
            criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)

        optimizer = optim.Adam(model.parameters(), lr=cfgs['params']['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfgs['params']['num_epochs'])

        # train model
        train_func = train_siamese_model if model_info['mode'] == 'siamese' else train_semseg_model
        eval_func = evaluate_siamese_model if model_info['mode'] == 'siamese' else evaluate_semseg_model

        best_model_state, best_val_metrics = train_func(
            model, dataloaders, criterion, optimizer, scheduler, device,
            num_epochs=cfgs['params']['num_epochs'],
            patience=cfgs['params']['early_stopping_patience'],
            min_delta=cfgs['params']['early_stopping_min_delta']
        )

        # load best model and evaluate
        model.load_state_dict(best_model_state)
        train_metrics = eval_func(model, train_loader, criterion, device)
        val_metrics = best_val_metrics
        test_metrics = eval_func(model, test_loader, criterion, device)

        results[model_name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }

        # save results
        torch.save(model.state_dict(), os.path.join(cfgs['path']['out_path'], f'{model_name}.pth'))
        save_predictions(model, test_loader, device, os.path.join(cfgs['path']['out_path'], f'{model_name}_predictions'), mode=model_info['mode'])

    # save and print final results
    print("\nFinal Results Summary:")
    with open(os.path.join(cfgs['path']['out_path'], 'result_summary.txt'), 'w') as f:
        for model_name, result in results.items():
            print(f"{model_name}:")
            f.write(f"{model_name}:\n")
            for split in ['train', 'val', 'test']:
                metrics = result[split]
                print(f"  {split.capitalize()} Metrics:")
                f.write(f"  {split.capitalize()} Metrics:\n")
                for metric in ['mean_iou', 'mean_precision', 'mean_recall', 'mean_f1_score', 'accuracy', 'loss']:
                    print(f"    {metric}: {metrics[metric]:.4f}")
                    f.write(f"    {metric}: {metrics[metric]:.4f}\n")
                print()
                f.write("\n")


if __name__ == "__main__":
    main()