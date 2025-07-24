# LoRA Easy Training - Jupyter Widget Edition 🚀

**A comprehensive, educational LoRA training system with advanced features**

- Widget-based interface designed for both beginners and advanced users
- Please note this is STILL a work in progress.
- Testing was only done on a singular RTX 4090 on a Vast AI Docker Container with pre installed SD WEB UI FORGE.
- Results MAY vary, please feel free to report issues as you see fit.
- Also the training guide is a little wonky, and the steps calculator isn't perfect.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20VastAI-lightgrey.svg)[![Discord](https://img.shields.io/discord/1024442483750490222?logo=discord&style=for-the-badge&color=5865F2)](https://discord.gg/HhBSvM9gBY) [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a>


## ✨ Features

### 🎓 **Educational Interface**
- **Real-time explanations** for every training parameter
- **Live step calculator** with visual feedback (target 250-1000 steps)
- **Smart recommendations** (e.g., CAME optimizer → REX scheduler)
- **Visual warnings** for incompatible settings

### 🧪 **Advanced Training Options**
- **Hybrid optimizer support**: CAME, Prodigy Plus, StableAdamW, ADOPT
- **LyCORIS methods**: DoRA, LoKr, LoHa, (IA)³, BOFT, GLoRA
- **Memory optimizations**: gradient checkpointing
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
git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/
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

## 🐛 Troubleshooting

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

### 🧪 **Research & Innovation**
- **LyCORIS Team** - DoRA, LoKr, LoHa, and advanced adaptation research
- **OneTrainer Contributors** - Memory optimization and training efficiency
- **HakuLatent Project** - Future-focused latent space research
- **Stable Diffusion Community** - Continuous innovation and knowledge sharing
---

**"Either gonna work or blow up!" - Built with curiosity, tested with courage! 😄**

## 📄 License

MIT License - Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Made with ❤️ by the community, for the community*
