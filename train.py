
# ============================================
# GLAUCOMA DETECTION SYSTEM - COMPLETE CODE
# ============================================
# Save these files in your project directory
# Requirements: CUDA 11.8, 6GB VRAM

# ============================================
# FILE 1: train.py (Training Script)
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# CLAHE Preprocessing
# ============================================
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)

        # Convert to LAB color space for CLAHE
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_np = self.clahe.apply(img_np)

        return Image.fromarray(img_np)

# ============================================
# Custom Dataset
# ============================================
class GlaucomaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================
# Data Preparation
# ============================================
def prepare_data(data_dir, test_size=0.15, val_size=0.15):
    glaucoma_dir = os.path.join(data_dir, 'glaucoma')
    normal_dir = os.path.join(data_dir, 'normal')

    # Get all image paths
    glaucoma_images = [os.path.join(glaucoma_dir, f) for f in os.listdir(glaucoma_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Create labels (1 for glaucoma, 0 for normal)
    glaucoma_labels = [1] * len(glaucoma_images)
    normal_labels = [0] * len(normal_images)

    # Combine
    all_images = glaucoma_images + normal_images
    all_labels = glaucoma_labels + normal_labels

    print(f"Total Glaucoma images: {len(glaucoma_images)}")
    print(f"Total Normal images: {len(normal_images)}")
    print(f"Total dataset size: {len(all_images)}")

    # First split: separate test set (15%)
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=test_size, random_state=SEED, stratify=all_labels
    )

    # Second split: separate train and validation (15% of original = ~17.6% of train_val)
    val_ratio = val_size / (1 - test_size)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=val_ratio, random_state=SEED, stratify=train_val_labels
    )

    print(f"\nTrain: {len(train_images)} ({len(train_images)/len(all_images)*100:.1f}%)")
    print(f"Validation: {len(val_images)} ({len(val_images)/len(all_images)*100:.1f}%)")
    print(f"Test: {len(test_images)} ({len(test_images)/len(all_images)*100:.1f}%)")

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# ============================================
# Data Transforms with CLAHE
# ============================================
def get_transforms():
    # Training transforms with CLAHE and augmentation
    train_transform = transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms with CLAHE only
    val_transform = transforms.Compose([
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# ============================================
# DenseNet Model
# ============================================
def create_densenet_model(num_classes=2, pretrained=True):
    # Use DenseNet-121 (lighter for 6GB VRAM)
    model = models.densenet121(pretrained=pretrained)

    # Freeze early layers to save memory
    for param in list(model.parameters())[:-30]:  # Freeze all except last 30 layers
        param.requires_grad = False

    # Replace classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model

# ============================================
# Training Function with Early Stopping
# ============================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, 
                patience=7, device='cuda', save_dir='models'):

    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'Early stopping patience: {patience_counter}/{patience}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break

    return history

# ============================================
# Evaluation Function
# ============================================
def evaluate_model(model, test_loader, device='cuda', save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # ROC-AUC (for binary classification)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        roc_auc = 0.0

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Glaucoma'])

    print("\n" + "="*50)
    print("TEST SET PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics, cm, all_labels, all_preds, all_probs

# ============================================
# Plotting Functions
# ============================================
def plot_training_history(history, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision, Recall, F1
    axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2, marker='o')
    axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2, marker='s')
    axes[1, 0].plot(history['val_f1'], label='F1-Score', linewidth=2, marker='^')
    axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Remove empty subplot
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training history plot to {save_dir}/training_history.png")
    plt.close()

def plot_confusion_matrix(cm, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Glaucoma'],
                yticklabels=['Normal', 'Glaucoma'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_dir}/confusion_matrix.png")
    plt.close()

def plot_roc_curve(all_labels, all_probs, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve to {save_dir}/roc_curve.png")
    plt.close()

# ============================================
# Main Execution
# ============================================
def main():
    # Configuration
    DATA_DIR = r'C:\Users\Astha Paika\Desktop\glaucoma\data'
    BATCH_SIZE = 8  # Reduced for 6GB VRAM
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    PATIENCE = 7

    print("="*60)
    print("GLAUCOMA DETECTION - DENSENET TRAINING")
    print("="*60)

    # Prepare data
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = prepare_data(DATA_DIR)

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    train_dataset = GlaucomaDataset(train_images, train_labels, transform=train_transform)
    val_dataset = GlaucomaDataset(val_images, val_labels, transform=val_transform)
    test_dataset = GlaucomaDataset(test_images, test_labels, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating DenseNet-121 model...")
    model = create_densenet_model(num_classes=2, pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Train
    print("\nStarting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs=NUM_EPOCHS, patience=PATIENCE, device=device)

    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load('models/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    metrics, cm, all_labels, all_preds, all_probs = evaluate_model(model, test_loader, device=device)

    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(cm)
    plot_roc_curve(all_labels, all_probs)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: models/best_model.pth")
    print(f"Results saved to: results/")
    print(f"Metrics saved to: results/metrics.json")

if __name__ == '__main__':
    main()
