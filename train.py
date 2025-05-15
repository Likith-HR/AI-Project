import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cnn_classifier import WasteClassifier
import os
from tqdm import tqdm
import numpy as np
from collections import Counter
import shutil
import glob
from PIL import Image

def is_valid_image(file_path):
    """Check if file is a valid image without using file extension"""
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except Exception:
        return False

def clean_dataset_directory(directory):
    """Clean up the dataset directory by removing invalid images"""
    if not os.path.exists(directory):
        print(f"Directory {directory} doesn't exist.")
        return
    
    # Get all subdirectories (categories)
    category_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not category_dirs:
        print(f"No category subdirectories found in {directory}")
        return
    
    print(f"Found categories: {category_dirs}")
    
    # Clean image files in each category folder
    for category in category_dirs:
        category_path = os.path.join(directory, category)
        print(f"Checking {category} folder for valid images...")
        
        files = os.listdir(category_path)
        if not files:
            print(f"Warning: No files found in {category}")
        
        for file_name in files:
            file_path = os.path.join(category_path, file_name)
            if os.path.isfile(file_path) and not is_valid_image(file_path):
                print(f"Removing invalid image: {file_path}")
                os.remove(file_path)
    
    print(f"Dataset directory {directory} prepared successfully.")

def get_class_weights(dataset):
    """Calculate class weights to handle imbalanced datasets"""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total_samples = len(targets)
    
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  {dataset.classes[cls]}: {count} images ({count/total_samples*100:.1f}%)")
    
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    weights = [class_weights[target] for target in targets]
    return torch.DoubleTensor(weights)

def train_model(train_dir='C:/Users/Hp/Downloads/Finaldataset/train2',
               val_dir='C:/Users/Hp/Downloads/Finaldataset/val',
               model_save_path='waste_cnn.pt',
               num_epochs=25,
               batch_size=32,
               patience=5,
               num_workers=4,
               learning_rate=1e-3,
               weight_decay=1e-4):
    
    # Clean and prepare dataset directories
    print("Preparing training and validation datasets...")
    clean_dataset_directory(train_dir)
    clean_dataset_directory(val_dir)
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets to get number of classes
    train_dataset = datasets.ImageFolder(train_dir)
    val_dataset = datasets.ImageFolder(val_dir)
    
    # Verify class consistency between train and val
    assert train_dataset.classes == val_dataset.classes, "Train and validation classes must match!"
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    # Calculate class weights for imbalanced datasets
    class_weights = get_class_weights(train_dataset)
    sampler = WeightedRandomSampler(class_weights, len(class_weights))
    
    # Initialize model with correct number of classes
    model = WasteClassifier(num_classes=num_classes)
    model = model.to(device)
    
    # Unfreeze some layers for fine-tuning
    model.unfreeze_layers(10)
    
    # Data transforms
    train_transform = model.get_transform(train=True)
    val_transform = model.get_transform(train=False)
    
    # Recreate datasets with transforms
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )
    
    # Create data loaders with weighted sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Loss function with label smoothing and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop with early stopping
    best_acc = 0.0
    no_improve = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': running_loss/len(train_pbar),
                'acc': 100.*train_correct/train_total
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        # Class-specific accuracy tracking
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Calculate per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
                
                # Update progress bar
                val_pbar.set_postfix({
                    'acc': 100.*val_correct/val_total
                })
        
        val_acc = 100 * val_correct / val_total
        train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = running_loss / len(train_loader)
        
        # Print per-class accuracy
        print("\nPer-class validation accuracy:")
        for i in range(num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"  {train_dataset.classes[i]}: {class_acc:.2f}%")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_dataset.classes,
                'best_acc': best_acc,
                'history': history
            }, model_save_path)
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'No improvement for {patience} epochs. Stopping training.')
                break
    
    print(f'Training complete. Best validation accuracy: {best_acc:.2f}%')
    return model, train_dataset.classes, history

if __name__ == '__main__':
    train_model()
