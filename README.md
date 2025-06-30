
# OptimizedAcquisition4vEM

**Data acquisition strategy of volume electron microscopy enables accurate image stack denoising using machine learning**

This repository provides the official implementation of our method for optimizing data acquisition in block-face volume electron microscopy (vEM). We demonstrate that deep learning-based denoising—under both supervised and self-supervised settings can accurately restore 3D structural details from noisy image stacks, allowing significantly higher imaging throughput without compromising downstream structural analysis.

## 📦 Installation

We recommend setting up a dedicated conda environment:

```bash
conda env create -f environment_optimized_vem.yml
conda activate optimized_vem
```

## 🧠 Features

- Deep learning-based denoising framework for vEM
- Supports both supervised and self-supervised training
- Enables throughput-oriented acquisition strategies
- Generalizable to SBEM, FIB-SEM, and other block-face EM systems
- Includes scripts for training, inference, and evaluation

## 🗂 Directory Structure

```
.
├── model/                           # Denoising network architectures
├── dataset/                         # Dataset loading and preprocessing
├── utils/                           # Auxiliary functions and tools
├── ckpt/                            # Checkpoints of trained models
├── configs/                         # Config files for training
├── log/                             # Training log and TensorBoard files
├── figs/                            # Figures used in README or paper
├── data/                            # Sample input stacks
├── environment_optimized_vem.yml    # Conda environment configuration
├── train_supervised_denoise.py      # Supervised training script
├── train_self_supervised_denoise.py # Self-supervised training script
├── volume_inference.py              # Denoising inference script
└── README.md
```

## 🚀 Getting Started

1. Place your own vEM `.tif` stack in `data/`.
2. Modify the configuration file in `configs/` to match your dataset and model.
3. Run supervised training:
   ```bash
   python train_supervised_denoise.py --config configs/supervised.yaml
   ```
4. Or run self-supervised training:
   ```bash
   python train_self_supervised_denoise.py --config configs/selfsup.yaml
   ```
5. Inference on a noisy stack:
   ```bash
   python volume_inference.py --input data/noisy_stack.tif --ckpt ckpt/best.pth
   ```

## 📄 License

This project will be released under an open-source license (TBD).
