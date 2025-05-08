"""
Training pipeline for Vision Transformer (ViT) with various positional encoding methods.
Supports MNIST and CIFAR-10 datasets with configurable architecture parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vit import VisionTransformer
import argparse
import os
from tqdm import tqdm
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_args():
    """
    Parse command line arguments for the training configuration.
    """
    parser = argparse.ArgumentParser(description='Vision Transformer Training')

    # Directories
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    # Positional encoding method
    parser.add_argument('--pos_encoding', type=str, default='absolute', 
                       choices=['none', 'absolute', 'relative', 'polynomial', 'rope-axial', 'rope-mixed'])
    # RoPE theta parameter for controlling frequency bands
    parser.add_argument('--rope_theta', type=float, default=100.0,
                       help='Theta parameter for RoPE variants (lower value = higher frequency)')
    # Polynomial-specific parameters
    parser.add_argument('--poly_degree', type=int, default=3,
                       help='Degree of polynomial for PolynomialRPE (default: 3)')
    parser.add_argument('--poly_shared_heads', action='store_true', default=True,
                       help='Share polynomial coefficients across attention heads')
    parser.add_argument('--no-poly_shared_heads', action='store_false', dest='poly_shared_heads',
                       help='Do not share polynomial coefficients across attention heads')
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # Model architecture parameters
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=6)
    return parser.parse_args()

def get_dataset(args):
    """
    Load and prepare dataset with appropriate transformations.
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_classes: Number of classes in the dataset
        in_chans: Number of input channels (1 for MNIST, 3 for CIFAR-10)
    """
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        in_chans = 1
    else:  # cifar10
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        in_chans = 3

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes, in_chans

def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
        average_loss: Average loss over all batches
        accuracy: Training accuracy percentage
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
    
    return total_loss/len(train_loader), 100.*correct/total

def test(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    
    Returns:
        average_loss: Average loss over all batches
        accuracy: Test accuracy percentage
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
    
    return total_loss/len(test_loader), 100.*correct/total

def main():
    """
    Main training function that handles:
    1. Setup of training configuration
    2. Model initialization
    3. Training loop execution
    4. Checkpoint and metric logging
    """
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{args.log_dir}/{args.dataset}_{args.pos_encoding}_{timestamp}.csv'

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'best_acc'])
    
    train_loader, test_loader, num_classes, in_chans = get_dataset(args)
    
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        pos_encoding=args.pos_encoding,
        rope_theta=args.rope_theta,
        poly_degree=args.poly_degree,
        poly_shared_heads=args.poly_shared_heads
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{args.ckpt_dir}/{args.dataset}_{args.pos_encoding}_best.pth')
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc, best_acc])
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Best Test Acc: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 