# LoRA Easy Training - Jupyter Widget Edition 🚀

A user-friendly LoRA training system powered by Kohya's proven training backend and interactive Jupyter widgets. Features unified model detection and streamlined workflows for VastAI, RunPod, and local environments.

| Python Version | License | Discord | Twitch | Support |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Discord](https://img.shields.io/badge/Discord-Join%20Our%20Server-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) |  <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a> |

## 🧪 Experimental Branch Notice

> **⚠️ You are on the `submodule-updates` branch - This is our experimental development branch!**
>
> This branch includes cutting-edge features and model support that may be unstable:
> - 🔬 **FLUX training** (transformer-based diffusion)
> - 🧬 **SD3/SD3.5 training** (advanced diffusion architecture)
> - 🌟 **Lumina2 training** (NextDiT + Gemma2 text encoder)
> - 🔧 **Latest bug fixes** and performance improvements
> - ⚡ **Enhanced upload widgets** (fixed cache issues)
>
> **Setup for experimental features**:
> ```bash
> git checkout submodule-updates
> # Make installer executable and run
> chmod +x ./jupyter.sh
> ./jupyter.sh
> ```

## 🌟 Overview & Key Features

- **What is this project?** A user-friendly LoRA training system based on KohyaSS, powered by interactive Jupyter widgets. Instead of typing lengthy Python commands, you get helpful widgets that walk you through each step. Works on your own local computer or rented GPU servers.
- **Why use it?**
    - **🎓 Beginner-friendly**: Helpful explanations and step-by-step guidance.
    - **🧮 Training calculator**: Shows roughly how long training could take.
    - **🛠️ Easy setup**: Works with VastAI, RunPod, and local computers.
    - **📊 Dataset tools**: Auto-tag images, upload files, manage captions.
    - **🚀 Multiple architectures**: SDXL, SD 1.5 (production-ready), plus experimental FLUX, SD3, and Lumina2 support with various optimizers and LoRA types.

<details><summary>What You Need</summary>

- **GPU**: Nvidia (For built-in CUDA support) or AMD Cards for ROCm. (Future Support for ARC and otherwise coming)
- **Python**: Version 3.10+ (3.10.6 recommended for maximum compatibility)
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
# Need: Python 3.10+ (3.10.6 recommended for maximum compatibility)
```

You can install Python 3.10.6 directly from Python's [main website here](https://www.python.org/downloads/release/python-3106/).

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

# 1a. If you're running the Unified Branch (For Testing) Please continue with the following commands:
git branch
git checkout unified
git fetch origin unified

# 2. Run the installer (downloads ~10-15GB)
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

Your main workflow:
- `Dataset_Maker_Widget.ipynb` - Prepare images and captions
- `Unified_LoRA_Trainer.ipynb` - Train SDXL, SD 1.5, or experimental Flux/SD3 models
- `Utilities_Notebooks.ipynb` - Calculate optimal training parameters, Resize Lora & More.

<details><summary>📊 Data Ingestion Options</summary>

Options for getting data into the system:
- **URL/ZIP Download**: Download and extract datasets from URLs (e.g., Hugging Face, Civitai) or local ZIP files.
- **Direct Image Upload**: Upload individual images directly into your dataset folder.
- **Gallery-DL Scraper**: Utilize the advanced `gallery-dl` integration to scrape images and their tags from over 300 supported websites.

</details>

<details><summary>How to Get Model/VAE Links</summary>

To use custom models or VAEs, you need to provide a direct download link. Here’s how to find them on popular platforms:

#### From Civitai

**Method 1: Using the Model Version ID**

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
</details>

<details><summary>🛠️ Advanced Features</summary>

### Image Utilities
- **Image Resizing**: Easily resize images in your dataset to a target resolution, with options for quality.

### Tag Curation
- **FiftyOne Integration**: Visually inspect and edit image tags using the FiftyOne interface. After making changes in FiftyOne, click 'Apply Curation Changes' to save them to your local caption files.

</details>

<details><summary>🏗️ New Unified Architecture</summary>

Our system now features:
- **Automatic Model Detection**: Unified trainer automatically detects SDXL vs SD 1.5 models
- **Kohya Backend Integration**: Leverages battle-tested Kohya training strategies
- **Environment-Agnostic**: Works across conda, venv, and system Python installations
- **Memory Optimization**: Automatic VRAM detection and profile selection
- **Cross-Platform**: Proper subprocess handling for Windows/Linux/macOS development

</details>

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
