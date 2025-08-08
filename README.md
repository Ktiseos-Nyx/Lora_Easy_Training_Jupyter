# LoRA Easy Training - Jupyter Widget Edition 🚀

**Train LoRAs with guided notebooks instead of confusing command lines**

This is a user-friendly LoRA training system based on proven methods from popular Google Colab notebooks. Instead of typing scary commands, you get helpful widgets that walk you through each step. Works on your own computer or rented GPU servers.



| Python Version | License | Discord | Twitch | Support |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Discord](https://img.shields.io/badge/Discord-Join%20Our%20Server-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) |  <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a> |

## Table of Contents
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

## ✨ What You Get

- **🎓 Beginner-friendly**: Helpful explanations and step-by-step guidance
- **🧮 Training calculator**: Shows exactly how long training will take
- **🛠️ Easy setup**: Works with VastAI, RunPod, and local computers
- **📊 Dataset tools**: Auto-tag images, upload files, manage captions
- **🚀 Multiple options**: SDXL, SD 1.5, various optimizers and LoRA types

All in simple notebooks - no command line required!

## 🚀 Quick Start

### What You Need
- **GPU**: NVIDIA (8GB+ VRAM) OR AMD GPU (16GB+ VRAM recommended for RDNA2/3)
- **Python**: Version 3.10.6 (compatible with Kohya-ss training)
- **Platform**: Windows or Linux based Operationg Systems.
- **Device** Local GPU or Rented Cloud GPU spaces. (Not Google Colab)

### 🖥️ Supported Platforms

**✅ Recommended (Easy Setup):**
- **VastAI**: PyTorch containers with Python 3.10 (NVIDIA + select AMD GPUs)
- **RunPod**: CUDA development templates (NVIDIA GPUs)
- **Local NVIDIA**: Anaconda/Miniconda with Python 3.10.6 + CUDA
- **Local AMD (Linux)**: Anaconda/Miniconda with Python 3.10.6 + ROCm 6.2+

**🧪 Experimental AMD Support:**
- **Local AMD (Windows)**: ZLUDA or DirectML acceleration
- **Cloud AMD**: Limited availability on popular GPU rental platforms.
- ⚠️ **NO SUPPORT FOR LOCAL MACINTOSH ARM/M1-M4 MACHINES** Currently RESEARCHING how to do this on mac machines intel or otherwise.

### 🐍 Python Setup

**Check your Python version first:**
```bash
python --version
# Need: Python 3.10.6 (other versions may break dependencies)
```

**If you don't have Python 3.10.6:**
```bash
# Create conda environment (recommended)
conda create -n lora-training python=3.10.6 -y
conda activate lora-training

# Or install Python 3.10.6 directly from python.org
```

**Always activate your environment before installation:**
```bash
conda activate lora-training  # If using conda
```

### 📥 Installation

**Prerequisites:** Git (for downloading) and Python 3.10.6

**Quick Git Check:**
```bash
git --version  # If this fails, install Git first
```

**Install Git if needed:**
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Mac**: `xcode-select --install` in Terminal
- **Linux**: `sudo apt install git` (Ubuntu/Debian)

**Download and Setup:**
```bash
# 1. Clone the repository
git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
cd Lora_Easy_Training_Jupyter

# 2. Run the installer (downloads ~10-15GB)
python ./installer.py

# Alternative for Mac/Linux:
chmod +x ./jupyter.sh && ./jupyter.sh
```

### 🚀 Start Training

1. **Open Jupyter** (if not already running):
   ```bash
   jupyter notebook
   # Or: jupyter lab
   ```

2. **Use the notebooks in order:**
   - `Dataset_Maker_Widget.ipynb` - Prepare images and captions
   - `Lora_Trainer_Widget.ipynb` - Configure and run training
   - `LoRA_Calculator_Widget.ipynb` - Calculate optimal steps (optional)

## 📖 How to Use

### Step 1: Prepare Your Images

Open `Dataset_Maker_Widget.ipynb` and run the cells in order:

```python
# Cell 1: Environment setup (if needed)
from shared_managers import create_widget
setup_widget = create_widget('setup')
setup_widget.display()

# Cell 2: Dataset preparation
dataset_widget = create_widget('dataset')
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

### AMD GPU Support

**🔥 AMD GPU Training is now supported through multiple acceleration methods:**

#### **ROCm (Linux Only) - Recommended**
- **Requirements**: Linux, AMD RDNA2/3 GPU, ROCm 6.1+ drivers
- **Installation**: Automatic via setup widget "Diagnose & Fix" button
- **Performance**: Native AMD acceleration, best compatibility
- **Setup Command**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0`

#### **ZLUDA (Experimental) - Windows & Linux**
- **Requirements**: AMD RDNA2+ GPU, ZLUDA runtime libraries
- **Installation**: Manual - download from [ZLUDA GitHub](https://github.com/vosen/ZLUDA)
- **Performance**: CUDA-to-AMD translation layer, experimental but promising
- **Status**: Some limitations with matrix operations, actively developed

#### **DirectML (Windows Fallback)**
- **Requirements**: Windows, any DirectX 12 compatible AMD GPU
- **Installation**: `pip install torch-directml`
- **Performance**: Lower performance but broader compatibility
- **Limitations**: Limited LoRA training support

#### **AMD GPU Memory Requirements**
- **RDNA2/3**: 16GB+ VRAM recommended (RX 6800 XT, RX 7900 XTX)
- **Older Cards**: May work with reduced settings
- **Memory Optimization**: Enable gradient checkpointing for large models

#### **AMD Training Tips**
- **Batch Size**: Start with 1, increase gradually
- **Resolution**: 768x768 recommended vs 1024x1024 for NVIDIA
- **Optimizer**: CAME optimizer saves significant VRAM
- **Mixed Precision**: fp16 may have compatibility issues, try bf16

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

⚠️ **AMD ZLUDA/ROCM**
- **Support for non NVIDIA CARDS** Currently untested and in development.
- **Symptoms** Untested on cards under 24gb of Video Ram.
- **Solution** Will gather users who could test this.
- **Support** WILL NOT WORK ON IMAC INTEL OR MAC METAL MACHINES.

⚠️ **Advanced LoRA Methods (EXPERIMENTAL)**
- **DoRA, GLoRA, BOFT (Butterfly)**: May not function correctly as of yet
- **Status**: Currently under testing and validation
- **Recommendation**: Use standard LoRA or LoCon for stable results
- **More testing**: Additional compatibility testing is ongoing

### Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check tooltips and explanations in widgets
- **Community**: Share your LoRAs and experiences!

## 🏆 Credits

🙏 **Built on the Shoulders of Giants**

This project builds upon and integrates the excellent work of:

- **[Jelosus2's LoRA Easy Training Colab](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** - Original Colab notebook that inspired this adaptation
- **[Derrian-Distro's LoRA Easy Training Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** - Core training backend and scripts
- **[HoloStrawberry's Training Methods](https://github.com/holostrawberry)** - Community wisdom and proven training techniques
- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** - Foundational training scripts and infrastructure
- **[Linaqruf](https://github.com/Linaqruf)** - Pioneer in accessible LoRA training, creator of influential Colab notebooks and training methods that inspired much of this work
- **AndroidXXL, Jelosus2** - Additional Colab notebook contributions that made LoRA training accessible
- **[ArcEnCiel](https://arcenciel.io/)** - Ongoing support and testing
- **[Civitai](https://civitai.com/)** - Platform for sharing LoRAs
- **[LyCORIS Team](https://github.com/67372a/LyCORIS)** - Advanced LoRA methods (DoRA, LoKr, etc.)

Special thanks to these creators for making LoRA training accessible to everyone!

---

## 🔒 Security

Found a security issue? Check our [Security Policy](SECURITY.md) for responsible disclosure guidelines.

## 📄 License

MIT License - Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved. Feel free to open issues or submit pull requests on GitHub.

---

*Made with ❤️ by the community, for the community*
