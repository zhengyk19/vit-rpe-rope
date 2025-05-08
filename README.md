# Vision Transformer Positional Encoding Comparison

This repository contains code for a Data Mining course project comparing different positional encoding methods in Vision Transformers (ViTs). The project evaluates Absolute Positional Encoding (APE) against various Relative Positional Encoding (RPE) methods on classification tasks using MNIST and CIFAR-10 datasets.

## Project Overview

This project implements and compares five positional encoding methods:
- No Positional Encoding (baseline)
- Absolute Positional Encoding (APE)
- Regular Relative Positional Encoding (RPE)
- Polynomial RPE (Poly-RPE)
- RoPE-Axial (Rotary Position Embedding with axial encoding)
- RoPE-Mixed (Rotary Position Embedding with learnable mixed frequencies)

## Setup

```bash
# Clone the repository
git clone https://github.com/zhengyk19/vit-rpe-rope.git
cd vit-rpe-rope

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision tqdm matplotlib seaborn
```

## Training Models

To train Vision Transformer models with different positional encoding methods:

```bash
# Train a model with Absolute Positional Encoding on MNIST
python train.py --dataset mnist --pos_encoding absolute --epochs 35 --batch_size 128

# Train a model with Polynomial RPE on CIFAR-10
python train.py --dataset cifar10 --pos_encoding polynomial --epochs 20 --batch_size 128

# Train a model with RoPE-Axial encoding
python train.py --dataset cifar10 --pos_encoding rope-axial --rope_theta 100.0 --epochs 20
```

### Training Arguments

- `--dataset`: Choose dataset - `mnist` or `cifar10`
- `--pos_encoding`: Positional encoding method - `none`, `absolute`, `relative`, `polynomial`, `rope-axial`, or `rope-mixed`
- `--rope_theta`: Theta parameter for RoPE variants (default: 100.0)
- `--poly_degree`: Degree of polynomial for Poly-RPE (default: 3)
- `--batch_size`: Batch size for training (default: 128)
- `--epochs`: Number of training epochs (default: 25)
- `--lr`: Learning rate (default: 0.001)
- `--img_size`: Input image size (default: 32)
- `--patch_size`: Patch size for ViT (default: 4)
- `--embed_dim`: Embedding dimension (default: 192)
- `--depth`: Number of transformer layers (default: 6)
- `--num_heads`: Number of attention heads (default: 6)

## Visualizing Positional Encodings

After training models, you can visualize the positional encoding patterns:

```bash
# Visualize a single model's positional encoding
python pe_similarity_visualizer.py --load_model --model_path checkpoints/cifar10_absolute_best.pth --model_config absolute

# Compare multiple positional encoding methods
python pe_similarity_visualizer.py --methods absolute relative polynomial rope-axial rope-mixed

# Visualize RoPE frequency representations
python rope_frequency_visualizer.py --methods rope-axial rope-mixed --pattern single
```

### Visualization Arguments

For `pe_similarity_visualizer.py`:
- `--load_model`: Flag to load a trained model
- `--model_path`: Path to the trained model checkpoint
- `--model_config`: Positional encoding method of the model
- `--methods`: List of positional encoding methods to visualize
- `--output_dir`: Directory to save visualizations (default: "visualizations")

For `rope_frequency_visualizer.py`:
- `--methods`: RoPE variants to visualize (rope-axial, rope-mixed)
- `--pattern`: Input pattern for analysis - `single`, `double` or `diagonal`


## Results

The project demonstrates that different positional encoding methods excel in different contexts:
- Polynomial RPE performs best on MNIST (99.23% accuracy)
- RoPE-Axial dominates on CIFAR-10 (66.93% accuracy)

Visualizations reveal how each method encodes spatial relationships differently, with RoPE-Axial concentrating frequency power along principal axes while RoPE-Mixed distributes it more evenly across directions.

## Acknowledgments

This project was completed as part of the Data Mining course. It builds on the Vision Transformer architecture introduced by Dosovitskiy et al. (2020) and incorporates Rotary Position Embedding methods from Heo et al. (2024). 