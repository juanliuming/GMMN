import torch
import torch.nn as nn
import torch.nn.functional as F
from vertices_selection_pretrain import VerticesSelection
from GMMNBlock import GMMNBlock
from MLP2 import WeightedFusionModel
from dataset import dataloader
import timeit
import os
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CyclicLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 12
num_classes = 3
EPOCHS = 2
train_loader, val_loader = dataloader(batch_size)

class DMEclassification(nn.Module):
    def __init__(self,num_classes,device):
        super(DMEclassification, self).__init__()
        self.VerticesSelection = VerticesSelection(device)
        self.GCNBlock = GMMNBlock(device)
        self.MultiModalFusion = WeightedFusionModel(num_classes,device)
        self.to(device)

    def forward(self, images):
        v1, v2, v3 = self.VerticesSelection(images)
        v1_new, v2_new, v3_new = self.GCNBlock(v1, v2, v3)
        output = self.MultiModalFusion(v1_new, v2_new, v3_new)
        return output

model = DMEclassification(num_classes, device)
optimizer = torch.optim.AdamW([
    {'params': model.VerticesSelection.BlueLayerResNet.parameters(), 'lr': 1e-5, 'weight_decay': 0.001},
    {'params': model.VerticesSelection.RGBLayerResNet.parameters(), 'lr': 1e-5, 'weight_decay': 0.001},
    {'params': model.VerticesSelection.MixLayerResNet.parameters(), 'lr': 1e-5, 'weight_decay': 0.001},
    {'params': model.VerticesSelection.attention16.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.VerticesSelection.attention32.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.VerticesSelection.attention64.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
], lr=0.0001, weight_decay=0.01)

initial_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

base_lrs = [lr / 10 for lr in initial_lrs]
max_lrs = [lr * 10 for lr in initial_lrs]

scheduler = CyclicLR(optimizer,
                     base_lr=base_lrs,
                     max_lr=max_lrs,
                     step_size_up=5 * len(train_loader),
                     mode='triangular',
                     cycle_momentum=False)

num_samples = [176, 41, 195]
total_samples = sum(num_samples)
class_weights = torch.tensor([total_samples / num for num in num_samples], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
start = timeit.default_timer()
best_val_accuracy = 0.0
best_model_save_path = ""

for epoch in tqdm(range(EPOCHS), desc="Epochs", position=0, leave=True):
    model.train()
    train_running_loss = 0
    train_labels = []
    train_preds = []
    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", position=0, leave=True) as train_pbar:
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            predict = model(images)
            loss = criterion(predict, labels)
            predict_label = torch.argmax(predict, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            train_labels.extend(labels.cpu().detach().numpy())
            train_preds.extend(predict_label.cpu().detach().numpy())
            train_running_loss += loss.item()

            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            train_pbar.set_postfix({"batch_loss": loss.item(), "lr": current_lrs[0]})

    train_loss = train_running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0
    val_labels = []
    val_preds = []

    with tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{EPOCHS}", position=0, leave=True) as val_pbar:
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                predict = model(images)
                loss = criterion(predict, labels)
                predict_label = torch.argmax(predict, dim=1)

                val_labels.extend(labels.cpu().detach().numpy())
                val_preds.extend(predict_label.cpu().detach().numpy())
                val_running_loss += loss.item()

                val_pbar.set_postfix({"batch_loss": loss.item()})

    val_loss = val_running_loss / len(val_loader)
    train_accuracy = sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels)
    val_accuracy = sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)

    train_precision = precision_score(train_labels, train_preds, average='macro', zero_division=0)
    train_recall = recall_score(train_labels, train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)

    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)


    print("-" * 30)
    print(f"Train Loss Epoch {epoch+1} : {train_loss:.4f}")
    print(f"Val Loss Epoch {epoch+1} : {val_loss:.4f}")
    print(f"Train Accuracy Epoch {epoch+1} : {train_accuracy:.4f}")
    print(f"Val Accuracy Epoch {epoch+1} : {val_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
    print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
    print("-" * 30)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_save_path = os.path.join('best_models', f"DMEClassification_best_model_epoch_{epoch+1}_valacc_{best_val_accuracy:.4f}.pth")
        torch.save(model.state_dict(), best_model_save_path)

stop = timeit.default_timer()
print(f"Training Time: {stop - start:.2f}s")

save_dir = "saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

best_models_dir = "best_models"
if not os.path.exists(best_models_dir):
    os.makedirs(best_models_dir)

final_model_save_path = os.path.join(save_dir, f"DMEClassification_model_epoch_{EPOCHS}_valacc_{val_accuracy:.4f}_trainacc_{train_accuracy:.4f}.pth")
torch.save(model.state_dict(), final_model_save_path)

print(f"Best model saved at: {best_model_save_path}")
