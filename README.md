# LoRA Easy Training - Jupyter Widget Edition 🚀

**Train LoRAs with guided notebooks instead of confusing command lines**

This is a user-friendly LoRA training system based on proven methods from popular Colab notebooks. Instead of typing scary commands, you get helpful widgets that walk you through each step. Works on your own computer or rented GPU servers.

## 🙏 **Built on the Shoulders of Giants**

This project builds upon and integrates the excellent work of:

- **[Jelosus2's LoRA Easy Training Colab](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** - Original Colab notebook that inspired this adaptation
- **[Derrian-Distro's LoRA Easy Training Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** - Core training backend and scripts
- **[HoloStrawberry's Training Methods](https://github.com/holostrawberry)** - Community wisdom and proven training techniques
- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** - Foundational training scripts and infrastructure

*Special thanks to these creators for making LoRA training accessible to everyone!*

| Python Version | License | Discord | Twitch | Support |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Discord](https://img.shields.io/badge/Discord-Join%20Our%20Server-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) |  <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a> |

## Table of Contents
- [About](#about)
- [✨ What You Get](#-what-you-get)
- [🚀 Quick Start](#-quick-start)
- [📖 How to Use](#-how-to-use)
- [🧮 Quick Training Calculator](#-quick-training-calculator)
- [🔧 Architecture](#-architecture)
- [🐛 Troubleshooting](#-troubleshooting)
- [🏆 Credits](#-credits)
- [🔒 Security](#-security)
- [📄 License](#-license)
- [🤝 Contributing](#-contributing)

## About

- Widget-based interface designed for both beginners and advanced users
- Please note this is STILL a work in progress.
- Testing was only done on a singular RTX 4090 on a Vast AI Docker Container with pre installed SD WEB UI FORGE.
- Results MAY vary, please feel free to report issues as you see fit.
- The system has been recently streamlined with improved widget organization and calculator accuracy.

## ✨ What You Get

### 🎓 **Beginner-Friendly**
- Helpful explanations for every setting (no more guessing!)
- Step calculator shows you exactly how long training will take
- Warnings when settings don't work together

### 🧪 **Advanced Options** (If You Want Them)
- Memory-efficient optimizers (CAME, Prodigy Plus)
- Special LoRA types (DoRA, LoKr, LoHa, IA³, BOFT, GLoRA)
- Memory-saving options for smaller GPUs

### 🛠️ **Easy Setup**
- **Prerequisites**: This installation assumes you already have Jupyter Lab or Jupyter Notebook running
- Two simple notebooks: one for datasets, one for training
- Works with VastAI and other GPU rental services
- Checks your system automatically

### 📊 **Dataset Tools**
- Auto-tag your images (WD14 for anime, BLIP for photos)
- Add/remove tags easily
- Upload ZIP files or folders

## 🚀 Quick Start

### What You Need
- **Computer**: Windows, macOS, or Linux
- **Python**: Version 3.10 ⚠️ **IMPORTANT: Python 3.12+ will break dependencies!**
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (can work with less)
- **Git**: For downloading this project (explained below)

### ⚠️ Python Version Compatibility

**🔍 Check Your Python Version First:**
```bash
python --version
# Should show Python 3.10.x - if it shows 3.12+, you'll need to fix this
```

**✅ Compatible Environments:**
- **Local**: Python 3.10.x installations
- **Anaconda/Miniconda**: Create environment with Python 3.10
- **Docker/Containers**: PyTorch or CUDA containers with Python 3.10
- **VastAI**: PyTorch/CUDA templates (avoid base Ubuntu images)

**❌ Known Issues:**
- **Python 3.12+**: Breaks TensorFlow dependencies (common in newer systems)
- **Base Ubuntu 22.04+**: Often ships with Python 3.12
- **Latest Anaconda**: May default to Python 3.12

**🔧 How to Fix Python Version Issues:**

**Local Users (Windows/Mac/Linux):**
```bash
# Create Python 3.10 environment with conda/miniconda
conda create -n lora-training python=3.10
conda activate lora-training
```

**Docker Users:**
Use PyTorch or CUDA base images instead of plain Ubuntu:
- `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`
- `nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04`

**VastAI Users:** 
Select templates with "PyTorch" or "CUDA" rather than "Ubuntu" base images.

### Installation

1.  **Get Git** (if you don't have it)

    Git is a tool for downloading code projects. Don't worry - you just need to install it once and you're done!
    
    **Check if you already have Git:** Open your terminal/command prompt and type `git --version`. If you see a version number, you're good to go!
    
    **If you need to install Git:**
    -   **Windows**: Download from [git-scm.com](https://git-scm.com/download/win) and run the installer
    -   **Mac**: Open Terminal and type `xcode-select --install` 
    -   **Linux**: Type `sudo apt-get install git` (Ubuntu/Debian) or use your system's package manager

2.  **Download This Project**

    Open your terminal/command prompt and navigate to where you want the project folder. Then run:
    ```bash
    git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
    cd Lora_Easy_Training_Jupyter
    ```

3.  **Run Setup**

    This automatically installs everything you need:
    
    **Mac/Linux:**
    ```bash
    chmod +x ./jupyter.sh
    ./jupyter.sh
    ```
    
    **Windows (or if the above doesn't work):**
    ```bash
    python ./installer.py
    ```
    
    Just wait for it to finish - it downloads the training tools and sets everything up.

### Start Training

**If using VastAI or similar:** Jupyter is probably already running - just open the notebooks in your browser.

**If on your own computer:** Start Jupyter like this:
```bash
jupyter notebook
```

**Then open these notebooks:**
1. `Dataset_Maker_Widget.ipynb` - Prepare your images and captions
2. `Lora_Trainer_Widget.ipynb` - Set up and run training  
3. `LoRA_Calculator_Widget.ipynb` - Calculate training steps (optional)

## 📖 How to Use

### Step 1: Prepare Your Images

Open `Dataset_Maker_Widget.ipynb` and run the first cell:

```python
# This starts the dataset preparation tool
from widgets.dataset_widget import DatasetWidget
dataset_widget = DatasetWidget()
dataset_widget.display()
```

Upload your images (ZIP files work great!) and the system will auto-tag them for you.

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

### Step 2: Train Your LoRA

Open `Lora_Trainer_Widget.ipynb` and run the cells to start training:

```python
# First, set up your environment
from widgets.setup_widget import SetupWidget
setup_widget = SetupWidget()
setup_widget.display()

# Then configure training
from widgets.training_widget import TrainingWidget  
training_widget = TrainingWidget()
training_widget.display()
```

**Good Starting Settings:**
- Learning Rate: UNet `5e-4`, Text Encoder `1e-4`
- LoRA: `8 dim / 4 alpha` (works for most characters)
- Target: 250-1000 training steps (the calculator helps you figure this out)

## 3. Extras


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

### Known Issues & Compatibility

⚠️ **Flux/SD3.5 Training (EXPERIMENTAL)**
- The `Flux_SD3_Training/` folder contains **work-in-progress** Flux and SD3.5 LoRA training
- May not function correctly - still under active development
- Use at your own risk for testing purposes only

⚠️ **Triton/ONNX Compatibility Warnings**
- **Docker/VastAI users**: Triton compiler may fail with AdamW8bit optimizer
- **Symptoms**: "TRITON NOT FOUND" or "triton not compatible" errors
- **Solution**: System will auto-fallback to AdamW (uses more VRAM but stable)
- **ONNX Runtime**: Dependency conflicts possible between `onnxruntime-gpu` and `open-clip-torch`

⚠️ **Advanced LoRA Methods (EXPERIMENTAL)**
- **DoRA, GLoRA, BOFT (Butterfly)**: May not function correctly as of yet
- **Status**: Currently under testing and validation
- **Recommendation**: Use standard LoRA or LoCon for stable results
- **More testing**: Additional compatibility testing is ongoing

⚠️ **VastAI/Cloud Instance Compatibility Issues**
- **Reality**: VastAI instances vary wildly - some work perfectly, others are broken out of the box
- **Common Issues**: Triton, ONNX, bitsandbytes, xFormers, and WandB may fail depending on the specific instance
- **Symptoms**: 
  - Dependency conflicts, import errors, CUDA detection failures
  - WandB 403 errors (Cloudflare/network routing issues)
  - "Library not found" or version compatibility problems
- **Before reporting bugs**: Test basic functionality on your specific instance first
- **Solutions**:
  - **First step**: Try the environment setup - it may resolve dependency issues
  - **WandB fails**: Clear API key to use local TensorBoard logging instead
  - **Dependencies broken**: Try `/venv/main/bin/pip install --force-reinstall [package]`
  - **Everything fails**: Kill the instance and try a different VastAI server
- **Bottom line**: If your instance can't do basic operations (import libraries, zip files), it's fundamentally broken - get a new one

### Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check tooltips and explanations in widgets
- **Community**: Share your LoRAs and experiences!

## 🏆 Credits

This project is built on the work of many awesome people:

**Training Methods:**
- **[Holostrawberry](https://github.com/hollowstrawberry)** - Training guides and recommended settings
- **[Kohya-ss](https://github.com/kohya-ss)** - Core training scripts
- **[LyCORIS Team](https://github.com/67372a/LyCORIS)** - Advanced LoRA methods (DoRA, LoKr, etc.)
- **[Derrian Distro](https://github.com/derrian-distro)** - Custom optimizers

**Notebook Inspirations:**
- **AndroidXXL, Jelosus2, Linaqruf** - Original Colab notebooks that made LoRA training accessible

**Community:**
- **[ArcEnCiel](https://arcenciel.io/)** - Ongoing support and testing
- **[Civitai](https://civitai.com/)** - Platform for sharing LoRAs

---

*"Either gonna work or blow up!" - Made with curiosity! 😄*

## 🔒 Security

Found a security issue? Check our [Security Policy](SECURITY.md) for responsible disclosure guidelines.

## 📄 License

MIT License - Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved. Feel free to open issues or submit pull requests on GitHub.

---

*Made with ❤️ by the community, for the community*
