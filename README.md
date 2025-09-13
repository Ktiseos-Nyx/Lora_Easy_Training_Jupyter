# LoRA Easy Training - Jupyter Widget Edition 🚀

A user-friendly LoRA training system powered by Kohya's proven training backend and interactive Jupyter widgets. Features unified model detection and streamlined workflows for VastAI, RunPod, and local environments.

| Python Version | License | Discord | Twitch | Support |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Discord](https://img.shields.io/badge/Discord-Join%20Our%20Server-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) |  <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a> |

## 🧪 Development Branch Notice

> **⚠️ You are on the `submodule-updates` branch - This is our current testing branch!**
>
> This branch includes experimental features that are available in the Kohya backend but may not be fully tested in our setup:
> - 🔬 **FLUX training** - Available in Kohya, integration status unknown
> - 🧬 **SD3/SD3.5 training** - Available in Kohya, integration status unknown  
> - 🌟 **Lumina2 training** - Available in Kohya, integration status unknown
> - 🔧 **Latest bug fixes** and performance improvements
> - ⚡ **Enhanced upload widgets** (fixed cache issues)
>
> **Note**: These experimental features exist in the underlying Kohya scripts but haven't been thoroughly tested with our widget system. Use at your own risk and expect possible issues.

## 🌟 Overview & Key Features

A LoRA training system built on Kohya's proven training backend with interactive Jupyter widget interfaces. Supports local development and cloud deployment on VastAI, RunPod, and similar platforms.

**Key Features:**
- Widget-based configuration interface
- Automatic model type detection (SDXL, SD1.5, FLUX, SD3)
- Integrated dataset preparation and tagging tools
- Training parameter calculator and optimization
- Multiple LoRA variants and optimizers
- Cross-platform compatibility

<details><summary>What You Need</summary>

- **GPU**: Nvidia (For built-in CUDA support) or AMD Cards for ROCm. (Future Support for ARC and otherwise coming)
- **Python**: Version 3.10+ required
- **Platform**: Windows or Linux based Operating Systems.

**Windows Users:** If you encounter Rust compilation errors during safetensors installation, this is not related to our notebook setup. It's a common Python packaging issue on Windows. Feel free to reach out on our [Discord](https://discord.gg/HhBSM9gBY) for assistance - we're happy to help guide you through the solution!
</details>

## 🚀 Quick Start (Installation & Setup)

Our trainer directly handles the task of installing major requirements depending on your environment.

- **Note for Cloud Users**: If you are on platforms like Vast.AI or RunPod, Jupyter is often launched automatically after your instance starts. You can usually proceed directly to the "Usage Guide" once your environment is ready.

<details><summary>📋 Installation Prerequisites</summary>

You will need Git and Python 3.10+.

**Check your Python version first:**

```bash
python --version
# Need: Python 3.10+ required
```

You can install Python 3.10+ from Python's [main website here](https://www.python.org/downloads/).

**Quick Git Check:**

```bash
git --version  # If this fails, install Git first
```

**Install Git if needed:**
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Mac**: `xcode-select --install` in Terminal
- **Linux**: `sudo apt install git` (Ubuntu/Debian)

</details>

**Main Installation Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
cd Lora_Easy_Training_Jupyter

# 2. Switch to the testing branch (recommended)
git checkout submodule-updates

# 3. Run the installer (downloads ~10-15GB)
python ./installer.py

# Alternative for Mac/Linux:
chmod +x ./jupyter.sh && ./jupyter.sh
```

## 📖 Usage Guide

### How to Launch Jupyter

(If Jupyter is NOT running)

```bash
jupyter notebook
# Or: jupyter lab
```

### Notebook Workflow

The system uses three specialized notebooks:

- **`Dataset_Maker_Widget.ipynb`** - Prepare images and captions for training
- **`Unified_LoRA_Trainer.ipynb`** - Configure and execute LoRA training
- **`Utilities_Notebooks.ipynb`** - Calculate parameters and resize trained models

For detailed workflow instructions, see our [Quick Start Guide](docs/quickstart.md) and [Notebook Workflow Guide](docs/guides/notebook-workflow.md).

## 🛠️ Troubleshooting & Support

**Known Issues & Compatibility**:

- ⚠️ **Triton/ONNX Compatibility Warnings**: Docker/VastAI users may encounter issues with AdamW8bit optimizer.
- ⚠️ **NO SUPPORT FOR LOCAL MACINTOSH ARM/M1-M4 MACHINES**
- 🐛 **FileUpload Widget Issues**: In some container environments, the file upload widget may not respond to file selection. **Workaround**: Use the manual upload buttons or direct file copying to dataset directories.
- 🔧 **CAME Optimizer Path Issues**: Due to container environment differences, you may need to manually edit the generated TOML config file. If training fails with "module 'LoraEasyCustomOptimizer' has no attribute 'CAME'", change `optimizer_type = "LoraEasyCustomOptimizer.CAME"` to `optimizer_type = "LoraEasyCustomOptimizer.came.CAME"` in your training config files.

**Getting Help**:
    - If you encounter issues or have questions, please:
        - Check our GitHub [Issues](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/issues) or [Discussions](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/discussions).
        - Join our [Discord Server](https://discord.gg/HhBSM9gBY) for community support.
    - We maintain a running tab of common issues and solutions in our [docs/](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/tree/main/docs) folder.

## 🙏 Credits & Acknowledgements

- **Built on the Shoulders of Giants**
This project builds upon and integrates the excellent work of:
- **[Jelosus2's LoRA Easy Training Colab](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** - Original Colab notebook that inspired this adaptation
- **[Derrian-Distro's LoRA Easy Training Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** - Core training backend and scripts as well as the forked Lycoris Repository and CAME/REX optimization strategies.
- **[HoloStrawberry's Training Methods](https://github.com/holostrawberry)** - Community wisdom and proven training techniques as well as foundational Google Colab notebooks.
- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** - Foundational training scripts and infrastructure
- **[Linaqruf](https://github.com/Linaqruf)** - Pioneer in accessible LoRA training, creator of influential Colab notebooks and training methods that inspired much of this work
- **AndroidXXL, Jelosus2** - Additional Colab notebook contributions that made LoRA training accessible
- **[ArcEnCiel](https://arcenciel.io/)** - Ongoing support and testing as well as Open Source AI Generative Models.
- **[Civitai](https://civitai.com/)** - Platform for Open Source AI Content
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

Made with ❤️ by the community, for the community.
