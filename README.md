# LoRA Easy Training - Jupyter Widget Edition 🚀

> **A comprehensive, educational LoRA training system with advanced features**  
> Widget-based interface designed for both beginners and advanced users

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20VastAI-lightgrey.svg)

## ✨ Features

### 🎓 **Educational Interface**
- **Real-time explanations** for every training parameter
- **Live step calculator** with visual feedback (target 250-1000 steps)
- **Smart recommendations** (e.g., CAME optimizer → REX scheduler)
- **Visual warnings** for incompatible settings

### 🧪 **Advanced Training Options**
- **Hybrid optimizer support**: CAME, Prodigy Plus, StableAdamW, ADOPT
- **LyCORIS methods**: DoRA, LoKr, LoHa, (IA)³, BOFT, GLoRA  
- **Memory optimizations**: Fused Back Pass, gradient checkpointing
- **Advanced schedulers**: REX Annealing, Schedule-Free optimization

### 🛠️ **Professional Tools**
- **Two-notebook architecture**: Separate dataset prep and training workflows
- **Modular backend system**: Easy to extend and maintain
- **VastAI optimization**: Container detection and automatic optimizations
- **Comprehensive validation**: Environment, GPU, and dependency checking

### 📊 **Dataset Management**
- **WD14 v3 taggers** with ONNX runtime optimization
- **Advanced tag management**: blacklisting, removal, trigger words
- **Multiple captioning methods**: WD14 for anime, BLIP for photos
- **Dataset upload and extraction** from local files or HuggingFace

## 🚀 Quick Start

### Prerequisites
- **Linux environment** (tested on Ubuntu 20.04+, VastAI containers)
- **Python 3.10+**
- **NVIDIA GPU** with 6GB+ VRAM (8GB+ recommended)
- **Git** and **aria2c** for downloading

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Lora_Easy_Training_Jupyter.git
cd Lora_Easy_Training_Jupyter

# Make installer executable and run
chmod +x ./jupyter.sh
./jupyter.sh

# Alternative: Run installer directly
python ./installer.py
```

### 🧮 Quick Training Calculator

Not sure about your dataset size or settings? Use our personal calculator:

```bash
python3 personal_lora_calculator.py
```

This tool helps you:
- Calculate optimal repeats and epochs for your dataset size
- Get personalized learning rate recommendations
- Estimate total training steps
- Build confidence for small datasets (stop being a chicken!) 🐔➡️🦅

### Launch Jupyter

```bash
# Start Jupyter notebook
jupyter notebook

# Open the training notebooks:
# 1. Dataset_Maker_Widget.ipynb - for dataset preparation
# 2. Lora_Trainer_Widget.ipynb - for training configuration
```

## 📖 Usage Guide

### 1. Dataset Preparation (`Dataset_Maker_Widget.ipynb`)

```python
# Run this cell to start the dataset widget
from widgets.dataset_widget import DatasetWidget

dataset_widget = DatasetWidget()
dataset_widget.display()
```

**Features:**
- Upload and extract dataset ZIP files
- Tag images with WD14 v3 taggers or BLIP captioning
- Manage captions and trigger words
- Remove unwanted tags

### 2. Training Setup (`Lora_Trainer_Widget.ipynb`)

```python
# Environment setup
from widgets.setup_widget import SetupWidget
setup_widget = SetupWidget()
setup_widget.display()

# Training configuration  
from widgets.training_widget import TrainingWidget
training_widget = TrainingWidget()
training_widget.display()
```

**Key Settings (Following Holostrawberry's Guide):**
- **Learning Rate**: UNet `5e-4`, Text Encoder `1e-4`
- **LoRA Structure**: `8 dim / 4 alpha` (great for characters, ~50MB)
- **Scheduler**: Cosine with 3 restarts
- **Target Steps**: 250-1000 (calculated automatically)

### 3. Advanced Mode

Enable advanced features by checking **"🧪 Enable Advanced Training Options"**:

- **🚀 Advanced Optimizers**: CAME (memory efficient), Prodigy Plus (learning rate free)
- **💾 Memory Wizardry**: Fused Back Pass for VRAM optimization
- **🦄 LyCORIS Methods**: DoRA for higher quality, LoKr for efficiency
- **🔬 Experimental Lab**: Future features and research-grade techniques

## 🎯 Training Examples

### Character LoRA (Recommended Settings)
```
Dataset Size: 20 images
Repeats: 10, Epochs: 10, Batch Size: 4
Total Steps: 500 ✅

Network: 8 dim / 4 alpha
Optimizer: AdamW8bit or CAME (advanced)
Learning Rate: 5e-4 UNet, 1e-4 Text Encoder
Scheduler: Cosine with 3 restarts
```

### Style LoRA (Advanced Settings)
```
LoRA Type: LoCon
LyCORIS Method: (IA)³ - Implicit Attention
Lower learning rates for longer training
```

## 🌐 VastAI Deployment

The system automatically detects VastAI containers and applies optimizations for cloud GPU training.

### 🎯 **GPU Recommendations by Training Type**

#### **SDXL LoRA Training** (Most Common)
```bash
# Minimum Specs (8GB can work with optimizations):
# GPU: RTX 3070 (8GB) - batch size 1-2, 1024 max resolution
# RAM: 16GB+
# Storage: 50GB+
# Cost: ~$0.15-0.25/hour
# Note: Results won't be as good as higher VRAM cards

# Recommended (Comfortable Training):
# GPU: RTX 4060 Ti 16GB or RTX 3090 (24GB)
# RAM: 16GB+
# Storage: 50GB+
# Cost: ~$0.20-0.40/hour
```

#### **SD 1.5 LoRA Training** (Lightweight)
```bash
# Ultra Budget (8GB VRAM possible!):
# GPU: RTX 2070/3060/3070 (8GB)
# RAM: 16GB+ (important for VRAM overflow)
# Storage: 30GB+
# Cost: ~$0.10-0.20/hour
# Settings: Max 1024 resolution, careful batch sizes, system RAM overflow

# Recommended:
# GPU: RTX 3090 (24GB)
# RAM: 16GB+
# Storage: 50GB+
# Cost: ~$0.20-0.30/hour
```

#### **Advanced Features** (DoRA, Large Datasets)
```bash
# DoRA Training (Works on 24GB!):
# GPU: RTX 3090 (24GB) or RTX 4090 (24GB)
# RAM: 32GB+
# Storage: 50GB+
# Cost: ~$0.20-0.40/hour
# Note: DoRA is 2-3x slower but same VRAM as regular LoRA

# Large Datasets (1000+ images):
# GPU: RTX 4090 (24GB) or A6000 (48GB)
# RAM: 32GB+
# Storage: 100GB+
# Cost: ~$0.40-0.70/hour
```

#### **Flux LoRA Training** (Cutting Edge)
```bash
# Minimum for Flux:
# GPU: RTX 4060 Ti 16GB (confirmed working)
# RAM: 32GB+
# Storage: 100GB+
# Cost: ~$0.30-0.50/hour
# Note: Uses "split mode" to fit in 16GB (doubles training time)

# Recommended:
# GPU: RTX 4090 (24GB) or better
# Note: Flux support coming - check community guides for current methods
```

### 🚀 **VastAI Quick Deploy**
```bash
# Choose your GPU tier, then:
git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git && cd Lora_Easy_Training_Jupyter && ./installer.py && ./start_jupyter.sh
```

### 💡 **VastAI Budget Tips** 💰
- **Ultra budget**: RTX 2070/3060 8GB can work for SD 1.5 (~$0.10/hr!)
- **Start cheap**: RTX 3070 for SD 1.5 (~$0.15/hr)
- **Sweet spot**: RTX 3090 handles everything including DoRA (~$0.25/hr)
- **Spot instances**: Save 50-70% - perfect for overnight training
- **Off-peak hours**: Prices drop significantly during US nighttime

### 🔧 **8GB VRAM Training Tips** (Based on Community Success)
- **SDXL on 8GB**: Confirmed working with RTX 3070 (batch size 1-2, results limited)
- **SD 1.5 on 8GB**: Works well, good results possible
- **Resolution**: 1024x1024 max for SDXL, can go higher for SD 1.5
- **Batch size**: Start with 1, max 2 for SDXL
- **Optimizations**: fp16, xformers, no Half VAE, enable buckets
- **Reality check**: "Results won't be as good as higher VRAM cards" but still usable!

## 🔧 Architecture

### Core Components
- **`core/managers.py`**: SetupManager, ModelManager for environment setup
- **`core/dataset_manager.py`**: Dataset processing and image tagging
- **`core/training_manager.py`**: Hybrid training manager with advanced features
- **`core/utilities_manager.py`**: Post-training utilities and optimization

### Widget Interface
- **`widgets/setup_widget.py`**: Environment setup and model downloads
- **`widgets/dataset_widget.py`**: Dataset preparation interface
- **`widgets/training_widget.py`**: Training configuration with advanced mode
- **`widgets/utilities_widget.py`**: Post-training tools

### Advanced Features
- **Modular optimizer system**: Easy to add new optimizers
- **Smart validation**: Prevents incompatible configurations
- **Educational tooltips**: Real-time explanations
- **Future-proofed**: Ready for new techniques (HakuLatent, etc.)

## 📚 Learning Resources

This project is designed to be educational. Each setting includes explanations:

- **📊 Step Calculator**: Visual feedback on training length
- **🎯 Smart Recommendations**: Automatic optimal pairings
- **⚠️ Compatibility Warnings**: Prevents common mistakes
- **📖 Detailed Tooltips**: Learn what each parameter does

## 🐛 Troubleshooting

### Common Issues

**"Environment not ready"**
```bash
# Run validation first
python -c "from widgets.setup_widget import SetupWidget; w=SetupWidget(); w.run_validate_environment(None)"
```

**"Training fails immediately"**
- Check total steps (should be 250-1000)
- Verify model and dataset paths
- Ensure sufficient VRAM/storage

**"Advanced features not working"**
- Some features require specific optimizers
- Fused Back Pass needs batch size = 1
- Check compatibility warnings in advanced mode

### Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check tooltips and explanations in widgets
- **Community**: Share your LoRAs and experiences!

## 🏆 Credits & Acknowledgments

This project builds upon the incredible work of many contributors in the AI training community:

### 🎯 **Primary Inspirations**
- **[Holostrawberry](https://civitai.com/user/holostrawberry)** - Educational training guides and proven parameter recommendations
- **[Derrian Distro](https://github.com/derrian-distro)** - LoRA_Easy_Training_Scripts_Backend, CAME optimizer integration
- **[Kohya-ss](https://github.com/kohya-ss)** - sd-scripts foundation and training infrastructure
- **[Kohaku-BlueLeaf](https://github.com/KohakuBlueleaf)** - LyCORIS advanced adaptation methods and HakuLatent research

### 🚀 **Technical Foundations**
- **[OneTrainer](https://github.com/Nerogar/OneTrainer)** - Fused Back Pass implementation and modern architecture
- **[bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)** - GUI design patterns and user experience
- **[Jelosus2](https://github.com/Jelosus2)** - Colab notebook adaptations and community feedback

### 📚 **Educational Resources**
- **Holostrawberry's Training Guides** - Parameter selection and best practices
- **Arc en Ciel Community** - Training techniques and community wisdom
- **Civitai Training Articles** - Real-world applications and results

### 🧪 **Research & Innovation**
- **LyCORIS Team** - DoRA, LoKr, LoHa, and advanced adaptation research
- **OneTrainer Contributors** - Memory optimization and training efficiency
- **HakuLatent Project** - Future-focused latent space research
- **Stable Diffusion Community** - Continuous innovation and knowledge sharing

### 🎨 **Design Philosophy**
This project embodies the principle that **AI training should be accessible, educational, and empowering**. We believe in:
- **Learning through doing** with real-time explanations
- **Progressive disclosure** (basic → advanced features)
- **Community knowledge sharing** 
- **Neurodivergent-friendly** interfaces and clear documentation

### 💝 **Special Thanks**
- **The broader Stable Diffusion community** for open research and collaboration
- **VastAI users** who test and provide feedback on container deployments
- **GitHub contributors** who report issues and suggest improvements
- **Educational content creators** who make AI training accessible to everyone

---

**"Either gonna work or blow up!" - Built with curiosity, tested with courage! 😄**

## 📄 License

MIT License - Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Made with ❤️ by the community, for the community*