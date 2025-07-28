# Getting Started: Your First LoRA

Welcome to LoRA Easy Training! This guide will walk you through the entire process of training your first LoRA using our widget-based notebooks. We'll go from installation to a finished, usable LoRA in just a few steps.

## Prerequisites

**Important**: This installation assumes you already have Jupyter Lab or Jupyter Notebook running on your system.

## 1. Installation

First, you need to get the project onto your machine and install the necessary dependencies.

1. **Clone the Repository**: Open a terminal or command prompt and run:
   ```bash
   git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
   ```

2. **Run the Installer**: Navigate into the newly created directory and run the installer script:
   ```bash
   cd Lora_Easy_Training_Jupyter
   python installer.py
   ```
   This step might take a while, as it needs to download several gigabytes of data.

3. **Launch Jupyter**: Start your Jupyter environment:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

## 2. The Three-Notebook System

The project uses three specialized notebooks for different purposes:

### 📊 **LoRA_Calculator_Widget.ipynb**
- **Purpose**: Calculate optimal training steps
- **When to use**: Before training to plan your parameters
- **What it does**: Shows you how long training will take and helps avoid over/under-training

### 📁 **Dataset_Maker_Widget.ipynb** 
- **Purpose**: Prepare your training dataset
- **What it handles**: Upload images, auto-tag with WD14/BLIP, edit captions, add trigger words
- **Required**: Must complete this before training

### 🚀 **Lora_Trainer_Widget.ipynb**
- **Purpose**: Configure and run training
- **What it does**: Environment setup, training configuration, live monitoring, post-training utilities

## 3. Your First LoRA: Step-by-Step

### Step 1: Plan Your Training (Calculator)

1. **Open**: `LoRA_Calculator_Widget.ipynb`
2. **Run Cell 1**: Display the step calculator
3. **Input your data**:
   - Number of images you plan to use (15-50 for characters)
   - Batch size (start with 2)
   - Repeats per epoch (5-10 for characters)
   - Max epochs (start with 10)
4. **Check the results**: Aim for reasonable training time (30min-2hours for first LoRA)

### Step 2: Prepare Your Dataset

1. **Open**: `Dataset_Maker_Widget.ipynb`
2. **Run Cell 1A** (if needed): Environment setup (skip if already done)
3. **Run Cell 2**: Main dataset widget
4. **Prepare your images**:
   - Gather 15-30 high-quality images of your character/style
   - Create a ZIP file of your images
5. **Upload and tag**:
   - Upload your ZIP file using the widget
   - Choose tagging method (WD14 for anime, BLIP for photos)  
   - Set threshold (0.35 for characters)
   - Run auto-tagging
6. **Add trigger word**:
   - Choose a unique trigger word (e.g., "saria_zelda", "mystyle_art")
   - Add it to all captions using the bulk edit tools
7. **Review and refine**: Check a few captions manually and edit if needed

### Step 3: Train Your LoRA

1. **Open**: `Lora_Trainer_Widget.ipynb`
2. **Run Cell 1**: Environment setup (downloads training backend ~10-15GB)
3. **Run Cell 2**: Training configuration widget
4. **Configure basic settings**:
   - **Model Name**: Give your LoRA a unique name (e.g., "my_first_character")
   - **Base Model**: Choose SDXL or SD1.5 model
   - **Dataset Path**: Point to your prepared dataset folder
   - **Trigger Word**: Same one you used in dataset prep
   - **Network Settings**: Start with 8 dim / 4 alpha
   - **Learning Rate**: 5e-4 for UNet, 1e-4 for Text Encoder
5. **Run Cell 3**: Training monitor and start training
6. **Monitor progress**: Watch the live training logs and progress bars

### Step 4: Use Your LoRA

1. **Find your LoRA**: After training completes, your `.safetensors` file will be in the output folder
2. **Install in your SD UI**: Copy the file to your Stable Diffusion web UI's `models/lora` directory
3. **Test generation**: Use your trigger word in prompts to activate the LoRA

## 4. Quick Tips for Success

### Dataset Quality
- **Image quality matters**: Use high-resolution, clear images
- **Variety is key**: Different poses, expressions, angles
- **Consistent style**: For style LoRAs, maintain visual consistency

### Training Settings
- **Start simple**: Use default settings for your first LoRA
- **Monitor loss**: Watch for steady decrease in training loss
- **Don't overtrain**: Stop if loss plateaus or starts increasing

### Common Issues
- **CUDA out of memory**: Reduce batch size to 1
- **Training too slow**: Check your target step count with the calculator
- **Poor results**: Review dataset quality and caption accuracy

## 5. Next Steps

Once you've successfully trained your first LoRA:

1. **Experiment with settings**: Try different optimizers (CAME, Prodigy)
2. **Advanced techniques**: Explore DoRA, LoKr, and other network types
3. **Larger datasets**: Scale up to 50-200 images for style LoRAs
4. **Share your work**: Upload to HuggingFace or Civitai using the utilities widget

**Congratulations on your first LoRA!** 🎉

---

*Need help? Check out our troubleshooting guide or join the Discord community.*