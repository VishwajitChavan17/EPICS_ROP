import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import seaborn as sns

from dataset_v2 import get_dataloaders
from model_v2 import get_rop_model

# 🔥 CONFIGURATION
DATA_ROOT = "C:/Users/vishw/Downloads/EPICS/HVDROPDB_RetCam_Neo_Classification"
BATCH_SIZE = 8  # Keep low for B3 on 12GB VRAM
IMG_SIZE = 512
EPOCHS_PHASE1 = 5   # Just the head
EPOCHS_PHASE2 = 25  # Full fine-tuning
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
            
    avg_loss = running_loss / len(loader)
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, acc, f1, auc, all_labels, all_preds

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('training_curves.png')
    plt.show()

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    
    # 2. Initialize Model
    model = get_rop_model(num_classes=2).to(DEVICE)
    
    # 3. Loss (Weighted if classes are imbalanced)
    # Check dataset_v2 output for distribution and adjust if needed
    criterion = nn.CrossEntropyLoss()
    
    # --- PHASE 1: Train Head Only ---
    print("\n--- PHASE 1: Training Classifier Head ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)
    
    for epoch in range(EPOCHS_PHASE1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, val_auc, _, _ = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")

    # --- PHASE 2: Fine-tune Everything ---
    print("\n--- PHASE 2: Fine-tuning Full Model ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_auc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS_PHASE2):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc, v_f1, v_auc, y_true, y_pred = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        
        print(f"Epoch {epoch+1}: T-Loss: {t_loss:.4f}, T-Acc: {t_acc:.2f}% | V-Loss: {v_loss:.4f}, V-Acc: {v_acc:.2f}%, AUC: {v_auc:.4f}")
        
        scheduler.step(v_loss)
        
        if v_auc > best_auc:
            best_auc = v_auc
            torch.save(model.state_dict(), 'best_rop_model.pth')
            print("Model saved! ⭐")

    # 4. Final Evaluation
    print("\nFinal Evaluation:")
    _, _, _, _, y_true, y_pred = validate(model, val_loader, criterion, DEVICE)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'ROP']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'ROP'], yticklabels=['Normal', 'ROP'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    main()
