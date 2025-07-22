# LoRA Easy Training - Jupyter Widget Edition
## Project Structure Overview

### 📁 **Main Project Files**
- `README.md` - Comprehensive user documentation
- `LICENSE` - MIT license and acknowledgments
- `requirements.txt` - Core Python dependencies
- `installer.py` - Comprehensive installation script
- `jupyter.sh` - Simple launch wrapper

### 📚 **Training Notebooks** (User Interface)
- `Dataset_Maker_Widget.ipynb` - Dataset preparation workflow
- `Lora_Trainer_Widget.ipynb` - Training configuration and execution

### 🧩 **Core System** (`core/`)
- `managers.py` - SetupManager, ModelManager (environment & downloads)
- `dataset_manager.py` - Dataset processing and image tagging
- `training_manager.py` - **HybridTrainingManager** with advanced features
- `utilities_manager.py` - Post-training utilities and optimization

### 🎛️ **Widget Interface** (`widgets/`)
- `setup_widget.py` - Environment setup and model downloads
- `dataset_widget.py` - Dataset upload, tagging, caption management
- `training_widget.py` - **Advanced training configuration with educational mode**
- `utilities_widget.py` - Post-training tools and optimization

### 🔧 **Custom Components** (`custom/`)
- `tag_images_by_wd14_tagger.py` - Enhanced WD14 v3 tagger with ONNX support

### 📖 **Documentation** (`docs/`)
- `PROJECT_OVERVIEW.md` - This file (project structure)
- `development/` - Development documentation and design notes

### 🖼️ **Assets** (`assets/`)
- Documentation images and examples
- Civitai/HuggingFace URL reference images

### 📓 **Sample Notebooks** (`sample_notebooks/`)
- Historical notebook versions and references
- Kept for educational/reference purposes

---

## 🏗️ **Architecture Highlights**

### **Two-Notebook Design**
- **Separation of concerns**: Dataset prep vs training
- **Clear workflow**: Step-by-step process
- **Neurodivergent-friendly**: Organized, predictable structure

### **Educational Philosophy**
- **Progressive disclosure**: Basic → Advanced mode
- **Real-time explanations**: Learn while doing
- **Smart recommendations**: Auto-optimal pairings
- **Visual feedback**: Color-coded warnings and tips

### **Hybrid Backend System**
- **Multiple optimizer support**: Kohya, Derrian, OneTrainer techniques
- **LyCORIS integration**: DoRA, LoKr, LoHa, (IA)³, BOFT, GLoRA
- **Memory optimizations**: Fused Back Pass, gradient checkpointing
- **Future-proofed**: Ready for HakuLatent and new research

### **Professional Features**
- **Environment detection**: VastAI, container, local optimization
- **Comprehensive validation**: GPU, memory, storage, network checks
- **API integrations**: Civitai and HuggingFace model downloads
- **Error handling**: Smart validation and user-friendly warnings

---

## 🎯 **Design Goals Achieved**

✅ **Accessible**: No complex server setup required  
✅ **Educational**: Learn training concepts through use  
✅ **Powerful**: Advanced features for experienced users  
✅ **Reliable**: Built on proven training foundations  
✅ **Extensible**: Modular system for future enhancements  
✅ **Community-Focused**: Proper credits and open source

---

*"Either gonna work or blow up!" - Built with curiosity, tested with courage! 😄*