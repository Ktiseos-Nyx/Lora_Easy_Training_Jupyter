# LoRA Easy Training - Jupyter Widget Edition 🚀

**A comprehensive, easy to use guided LoRA training system with advanced features**

Welcome to Lora Easy Training Jupyter! We've distilled the brilliant, accessible LoRA training methods pioneered by AndroidXXL, Holostrawberry, Jelosus2, and Linaqruf in their Google Colab Notebooks, and brought them to a more open environment. While not as simple as a 'drag-and-drop to Google Drive' solution, this system runs directly on your machine or rented GPU platforms, offering unparalleled control and flexibility. Lora Easy Training Jupyter aims to democratize advanced LoRA training, making powerful GPU resources available for anything from an 8GB consumer card all the way up to H100s (if your budget allows!). It's designed to be as straightforward as possible, empowering you with direct access to your training environment!

| Python Version | License | Discord | Twitch | Support |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Discord](https://img.shields.io/badge/discord/1024442483750490222?logo=discord&style=for-the-badge&color=5865F2)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) |  <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a> |

## Table of Contents
- [LoRA Easy Training - Jupyter Widget Edition 🚀](#lora-easy-training---jupyter-widget-edition-)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [✨ Features](#-features)
    - [🎓 **Educational Interface**](#-educational-interface)
    - [🧪 **Advanced Training Options**](#-advanced-training-options)
    - [🛠️ **Professional Tools**](#️-professional-tools)
    - [📊 **Dataset Management**](#-dataset-management)
  - [🚀 Quick Start](#-quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [🧮 Quick Training Calculator](#-quick-training-calculator)
    - [Launch Jupyter](#launch-jupyter)
  - [📖 Usage Guide](#-usage-guide)
    - [1. Dataset Preparation (`Dataset_Maker_Widget.ipynb`)](#1-dataset-preparation-dataset_maker_widgetipynb)
    - [How to Get Model/VAE Links](#how-to-get-modelvae-links)
      - [From Civitai](#from-civitai)
      - [From Hugging Face](#from-hugging-face)
    - [2. Training Setup (`Lora_Trainer_Widget.ipynb`)](#2-training-setup-lora_trainer_widgetipynb)
    - [3. Advanced Mode](#3-advanced-mode)
  - [🔧 Architecture](#-architecture)
    - [Core Components](#core-components)
    - [Widget Interface](#widget-interface)
  - [🐛 Troubleshooting](#-troubleshooting)
    - [Support](#support)
  - [🏆 Credits \& Acknowledgments](#-credits--acknowledgments)
    - [🎯 **Primary Inspirations**](#-primary-inspirations)
    - [🚀 **Technical Foundations**](#-technical-foundations)
    - [🧪 **Research \& Innovation**](#-research--innovation)
  - [🔒 Security](#-security)
  - [📄 License](#-license)
  - [🤝 Contributing](#-contributing)

## About

- Widget-based interface designed for both beginners and advanced users
- Please note this is STILL a work in progress.
- Testing was only done on a singular RTX 4090 on a Vast AI Docker Container with pre installed SD WEB UI FORGE.
- Results MAY vary, please feel free to report issues as you see fit.
- The system has been recently streamlined with improved widget organization and calculator accuracy.

## ✨ Features

### 🎓 **Educational Interface**
- **Real-time explanations** for every training parameter
- **Live step calculator** with visual feedback (target 250-1000 steps)
- **Smart recommendations** (e.g., CAME optimizer → REX scheduler)
- **Visual warnings** for incompatible settings

### 🧪 **Advanced Training Options**
- **Advanced optimizers**: CAME (memory efficient), Prodigy Plus (adaptive learning)
- **LyCORIS methods**: DoRA, LoKr, LoHa, IA³, BOFT, GLoRA
- **Memory optimizations**: Gradient checkpointing and 8-bit variants
- **Advanced schedulers**: REX Annealing, cosine with restarts

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
- **Operating System**: Windows, macOS, or Linux.
- **Python**: Version 3.10 or newer.
- **GPU**: An NVIDIA GPU with at least 8GB of VRAM is strongly recommended for a smooth experience.
- **Git**: Required to download the repository.

### Installation

1.  **Install Git**

    If you don't have Git installed, you can get it here:
    -   **Windows**: Download and install from [git-scm.com](https://git-scm.com/download/win).
    -   **macOS**: Open your terminal and run `xcode-select --install`.
    -   **Linux**: Use your distribution's package manager (e.g., `sudo apt-get install git` for Debian/Ubuntu).

2.  **Clone the Repository**

    Open your terminal or command prompt, navigate to where you want to store the project, and run:
    ```bash
    git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
    cd Lora_Easy_Training_Jupyter
    ```

3.  **Run the Setup Script**

    This script will prepare the environment and install all necessary dependencies.
    -   **On macOS and Linux**:
        ```bash
        chmod +x ./jupyter.sh
        ./jupyter.sh
        ```
    -   **On Windows (or as an alternative for other platforms)**:
        ```bash
        python ./installer.py
        ```
    > **Note:** All other requirements are automatically installed by the setup script.


### 🧮 Quick Training Calculator

Not sure about your dataset size or settings? Use our personal calculator:

```bash
python3 personal_lora_calculator.py
```

This tool helps you:
- Calculate optimal repeats and epochs for your dataset size
- Get personalized learning rate recommendations
- Estimate total training steps
- Build confidence for any dataset size (no more guesswork!) 🎯

### Launch Jupyter

- Please note this is only if you're using this on a barebones rental OR a local machine, most rented setups have Jupyter already running.

```bash
# Start Jupyter notebook
jupyter notebook

# Open the training notebooks:
# 1. Dataset_Maker_Widget.ipynb - for dataset preparation
# 2. Lora_Trainer_Widget.ipynb - for training configuration
# 3. LoRA_Calculator_Widget.ipynb - for step calculations
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
- **Unified Dataset Setup**: Choose between URL/ZIP download or direct image upload
- **WD14 v3 Taggers**: Advanced anime tagging with ONNX optimization
- **BLIP Captioning**: Professional photo captioning support
- **Caption Management**: Add trigger words and remove unwanted tags
- **Dataset Cleanup**: Smart cleanup tools for re-tagging workflows

### How to Get Model/VAE Links

To use custom models or VAEs, you need to provide a direct download link. Here’s how to find them on popular platforms:

#### From Civitai

**Method 1: Using the Model Version ID**

This is the easiest method if a model has multiple versions.

1.  Navigate to the model or VAE page.
2.  Look at the URL in your browser's address bar. If it includes `?modelVersionId=XXXXXX`, you can copy the entire URL and paste it directly into the widget.
3.  If you don't see this ID, try switching to a different version of the model and then back to your desired version. The ID should then appear in the URL.

![How to get a link from Civitai using the version ID](./assets/model_url_civitai_1.png)

**Method 2: Copying the Download Link**

Use this method if the model has only one version or if a version has multiple files.

1.  On the model or VAE page, scroll down to the "Files" section.
2.  Right-click the **Download** button for the file you want.
3.  Select "Copy Link Address" (or similar text) from the context menu.

![How to get a link from Civitai by copying the download address](./assets/model_url_civitai_2.png)

#### From Hugging Face

**Method 1: Using the Repository URL**

1.  Go to the main page of the model or VAE repository you want to use.
2.  Copy the URL directly from your browser's address bar.

![How to get a link from Hugging Face using the repository URL](./assets/model_url_hf_1.png)

**Method 2: Copying the Direct File Link**

1.  Navigate to the "Files and versions" tab of the repository.
2.  Find the specific file you want to download.
3.  Click the **"..."** menu to the right of the file size, then right-click the "Download" link and copy the link address.

![How to get a link from Hugging Face by copying the direct file address](./assets/model_url_hf_2.png)

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
- **Scheduler**: Cosine (with optional restarts)
- **Target Steps**: 250-1000 (calculated automatically)

### 3. Advanced Mode

Enable advanced features by checking **"🧪 Enable Advanced Training Options"**:

- **🚀 Advanced Optimizers**: CAME (memory efficient), Prodigy Plus (learning rate free)
- **🦄 LyCORIS Methods**: DoRA, LoKr, LoHa, BOFT, GLoRA for advanced adaptation
- **💾 Memory Optimizations**: Gradient checkpointing and efficient training
- **🔬 Experimental Features**: REX scheduler, Huber loss, and more

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
- **[Holostrawberry](https://github.com/hollowstrawberry)** - Educational training guides and proven parameter recommendations
- **[Derrian Distro](https://github.com/derrian-distro)** - LoRA_Easy_Training_Scripts_Backend, CAME optimizer integration
- **[Kohya-ss](https://github.com/kohya-ss)** - sd-scripts foundation and training infrastructure
- **[Kohaku-BlueLeaf](https://github.com/KohakuBlueleaf)** - LyCORIS advanced adaptation methods and HakuLatent research

### 🚀 **Technical Foundations**
- **[kohya_ss](https://github.com/kohya-ss/sd-scripts/)** - Core training scripts and infrastructure
- **[OneTrainer](https://github.com/Nerogar/OneTrainer)** - Advanced training research and future integration
- **[Jelosus2](https://github.com/Jelosus2)** - Colab notebook adaptations
- **[AndroidXXL](https://github.com/AndroidXXL)** - Colab notebook adaptations
- **[ArcEnCiel](https://arcenciel.io/)** Continued Cabal & Peer Support
- **[Civitai](https://civitai.com/)** Scarily for the platform to exist on.

### 🧪 **Research & Innovation**
- **LyCORIS Team** - DoRA, LoKr, LoHa, BOFT, GLoRA and advanced adaptation research
- **Training Research Community** - Memory optimization and efficiency improvements
- **Stable Diffusion Community** - Continuous innovation and knowledge sharing
---

**"Either gonna work or blow up!" - Built with curiosity, tested with courage! 😄**

## 🔒 Security

Found a security issue? Check our [Security Policy](SECURITY.md) for responsible disclosure guidelines.

## 📄 License

MIT License - Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved. Feel free to open issues or submit pull requests on GitHub.

---

*Made with ❤️ by the community, for the community*
