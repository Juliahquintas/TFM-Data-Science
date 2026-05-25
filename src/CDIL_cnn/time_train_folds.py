"""
time_train_folds.py — Motor de entrenamiento y evaluación
Implementación EXACTA de Marta Rey. No tocar.

Exporta: TrainModel · evaluate · plot_metrics
"""

import torch, numpy as np
import matplotlib.pyplot as plt
import pandas as pd, seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)


def plot_metrics(epochs, train_losses, train_accs, val_losses, val_accs, dir1, dir2):
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses,   label='Validation Loss')
    plt.title('Model Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(dir1, dpi=300); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(range(1, epochs+1), train_accs, label='Train')
    plt.plot(range(1, epochs+1), val_accs,   label='Validation')
    plt.title('Model Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout(); plt.savefig(dir2, dpi=300); plt.close()


def _quick_eval(tag, sep, loader, device, net, loss_fn, loginf):
    """Evaluación rápida durante el entrenamiento (sin métricas completas)."""
    total_loss, total_n, total_correct = 0, 0, 0
    t0 = datetime.now()
    for X, Y in tqdm(loader, total=len(loader)):
        X, Y   = X.float().to(device), Y.to(device)
        pred    = net(X)
        total_loss    += loss_fn(pred, Y).item()
        total_n       += len(Y)
        _, pred_labels = pred.max(1)
        total_correct += pred_labels.eq(Y).sum().item()
    loss_mean = total_loss / total_n
    acc       = total_correct / total_n * 100
    elapsed   = (datetime.now() - t0).total_seconds()
    loginf(f'{tag} num: {total_n} — {tag} loss: {loss_mean:.4f} — {tag} accuracy: {acc:.2f} — Time: {elapsed:.1f}s')
    loginf('_' * sep)
    return loss_mean, acc


def evaluate(net, device, loader, class_names, save_dir, fold_idx, loss_fn):
    """Evaluación completa sobre test: calcula todas las métricas y guarda figuras."""
    net.eval()
    total_loss, total_n, total_correct = 0, 0, 0
    y_pred, y_probs, y_true = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs         = net(inputs)
            total_loss     += loss_fn(outputs, labels).item()
            total_n        += len(labels)
            _, predicted    = outputs.max(1)
            total_correct  += predicted.eq(labels).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(outputs.cpu().numpy()[:, 1])   # prob clase Parkinson
            y_true.extend(labels.cpu().numpy())

    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, average='binary')
    recall      = recall_score(y_true,    y_pred, average='binary')
    f1          = f1_score(y_true,         y_pred, average='binary')
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score   = roc_auc_score(y_true,    y_pred)
    cm          = confusion_matrix(y_true,  y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f'[Fold {fold_idx}] Acc:{accuracy:.4f} Prec:{precision:.4f} '
          f'Rec:{recall:.4f} F1:{f1:.4f} AUC:{auc_score:.4f} Spec:{specificity:.4f}')

    pd.DataFrame([{'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
                   'F1 Score': f1, 'AUC': auc_score, 'Specificity': specificity}])\
      .to_csv(f'{save_dir}/test_metrics_fold{fold_idx}.csv', index=False)

    # ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='b', label=f'ROC (AUC={auc_score:.2f})')
    plt.plot([0,1],[0,1], color='black', lw=1, linestyle='--')
    plt.grid(); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_dir}/roc_curve_fold{fold_idx}.png', dpi=300); plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/cm_fold{fold_idx}.png', dpi=300); plt.close()

    return accuracy, precision, recall, f1, auc_score, specificity, fpr, tpr, thresholds, y_true, y_pred


def TrainModel(net, device, trainloader, valloader, testloader,
               n_epochs, optimizer, loss_fn, loginf, file_name):
    """
    Entrena durante n_epochs y guarda el checkpoint con mejor val_accuracy.
    Devuelve: train_losses, train_accs, val_losses, val_accs
    """
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    saving_best = 0

    for epoch in range(n_epochs):
        net.train()
        t_loss, t_correct, t_n = 0, 0, 0
        t0 = datetime.now()

        for X, Y in tqdm(trainloader, total=len(trainloader)):
            X, Y = X.float().to(device), Y.to(device)
            optimizer.zero_grad()
            pred       = net(X)
            batch_loss = loss_fn(pred, Y)
            batch_loss.backward()
            optimizer.step()
            t_loss    += batch_loss.item()
            t_correct += (pred.argmax(1) == Y).float().sum().item()
            t_n       += len(Y)

        tr_loss = t_loss / t_n
        tr_acc  = t_correct / t_n * 100
        train_losses.append(tr_loss); train_accs.append(tr_acc)
        loginf(f'Epoch {epoch} | Train loss:{tr_loss:.4f} acc:{tr_acc:.2f} | '
               f'Time:{(datetime.now()-t0).total_seconds():.1f}s')

        with torch.no_grad():
            net.eval()
            val_loss, val_acc = _quick_eval('Val',  80, valloader,  device, net, loss_fn, loginf)
            val_losses.append(val_loss); val_accs.append(val_acc)

            if val_acc >= saving_best:
                saving_best = val_acc
                torch.save(net.state_dict(), file_name)
                _, test_acc = _quick_eval('Test', 120, testloader, device, net, loss_fn, loginf)
                loginf(f'★ Best val:{val_acc:.2f}  test:{test_acc:.2f}  (epoch {epoch})')

    loginf('Training complete.'); loginf('_' * 200)
    return train_losses, train_accs, val_losses, val_accs
