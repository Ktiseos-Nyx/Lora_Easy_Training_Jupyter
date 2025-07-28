# Widget Reference Guide

This comprehensive reference covers all widgets and their parameters in the LoRA Easy Training system.

## Dataset Widget (Dataset_Maker_Widget.ipynb)

### Upload Section
| Parameter | Type | Description |
|-----------|------|-------------|
| **ZIP File Upload** | File Browser | Upload ZIP files containing training images |
| **Individual Files** | File Browser | Add individual image files to dataset |
| **HuggingFace URL** | Text Field | Direct URL to HuggingFace dataset repository |
| **HF Token** | Text Field | Authentication token for private HF datasets |
| **Extract Directory** | Text Field | Target directory for extracted images |

### Auto-Tagging Section
| Parameter | Type | Description |
|-----------|------|-------------|
| **Tagging Method** | Dropdown | WD14 (anime/art) or BLIP (photos) |
| **WD14 Model** | Dropdown | wd14-vit-v2, wd14-convnext-v2, wd14-swinv2-v2, wd14-convnext-v3 |
| **Threshold** | Slider | Tag confidence threshold (0.1-0.9, default 0.35) |
| **Blacklist Tags** | Text Field | Comma-separated list of tags to exclude |
| **Caption Extension** | Text Field | File extension for caption files (default .txt) |

### Caption Management
| Parameter | Type | Description |
|-----------|------|-------------|
| **Trigger Word** | Text Field | Unique word to identify your LoRA concept |
| **Find Text** | Text Field | Text to search for in captions |
| **Replace With** | Text Field | Replacement text (empty to remove) |
| **Tags to Remove** | Text Field | Comma-separated tags to remove from all captions |
| **Sort Alphabetically** | Checkbox | Sort tags in alphabetical order |
| **Remove Duplicates** | Checkbox | Remove duplicate tags within captions |

### Dataset Scraper (Gelbooru)
| Parameter | Type | Description |
|-----------|------|-------------|
| **Tags** | Text Field | Gelbooru search tags (space-separated) |
| **Limit** | Number Field | Maximum number of images to download |
| **Output Folder** | Text Field | Destination folder name |
| **Confirmation** | Checkbox | Require confirmation before download |

## Training Widget (Lora_Trainer_Widget.ipynb)

### Basic Configuration
| Parameter | Type | Description |
|-----------|------|-------------|
| **Project Name** | Text Field | Name for your LoRA project |
| **Base Model** | Dropdown | Pre-downloaded base model to train on |
| **Dataset Path** | Text Field | Path to prepared training dataset |
| **Output Name** | Text Field | Filename for the trained LoRA |
| **Trigger Word** | Text Field | Unique activation word for your LoRA |

### Network Settings
| Parameter | Type | Description |
|-----------|------|-------------|
| **Network Type** | Dropdown | LoRA, DoRA, LoKr, LoHa, (IA)³, BOFT, GLoRA |
| **Network Dimension** | Number Field | LoRA rank/dimension (4-128, typical: 8-16) |
| **Network Alpha** | Number Field | Scaling factor (typical: half of dimension) |
| **Dropout** | Number Field | Regularization dropout rate (0.0-0.5) |

### Training Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| **Learning Rate (UNet)** | Number Field | Learning rate for UNet (1e-5 to 1e-3, typical: 1e-4) |
| **Learning Rate (Text Encoder)** | Number Field | Learning rate for text encoder (typically 10x lower) |
| **Optimizer** | Dropdown | AdamW, CAME, Prodigy, StableAdamW, ADOPT |
| **Scheduler** | Dropdown | constant, linear, cosine, cosine_with_restarts, REX |
| **Max Epochs** | Number Field | Maximum training epochs |
| **Batch Size** | Number Field | Images per batch (1-8, lower for less VRAM) |
| **Gradient Accumulation** | Number Field | Steps to accumulate gradients |

### Advanced Options (🧪 Enable Advanced Training Options)
| Parameter | Type | Description |
|-----------|------|-------------|
| **Mixed Precision** | Dropdown | fp16, bf16, no (fp16 saves memory) |
| **Gradient Checkpointing** | Checkbox | Trade compute for memory savings |
| **Clip Skip** | Number Field | Skip final CLIP layers (1-2) |
| **Min SNR Gamma** | Number Field | Training stability (5 recommended) |
| **V-Prediction** | Checkbox | Enable for v-prediction models |
| **Sample Prompts** | Text Area | Prompts for sample generation during training |
| **Sample Every N Epochs** | Number Field | How often to generate samples |

### Dataset Configuration
| Parameter | Type | Description |
|-----------|------|-------------|
| **Image Resolution** | Dropdown | Training resolution (512, 768, 1024) |
| **Bucket Resolution** | Dropdown | Aspect ratio bucketing settings |
| **Bucket No Upscale** | Checkbox | Don't upscale smaller images |
| **Dataset Repeats** | Number Field | How many times to repeat dataset per epoch |
| **Shuffle Caption** | Checkbox | Randomize caption tag order |
| **Keep Tokens** | Number Field | Number of tokens to keep at start of caption |

### Regularization (Optional)
| Parameter | Type | Description |
|-----------|------|-------------|
| **Regularization Dataset** | Text Field | Path to regularization images |
| **Reg Dataset Repeats** | Number Field | Repeats for regularization dataset |

## Setup Widget

### Environment Setup
| Parameter | Type | Description |
|-----------|------|-------------|
| **Install Backend** | Button | Download and install training backend |
| **Check System** | Button | Validate GPU, VRAM, and storage |
| **Setup Status** | Display | Shows current setup progress |

### Model Downloads
| Parameter | Type | Description |
|-----------|------|-------------|
| **Model URL** | Text Field | HuggingFace or Civitai model URL |
| **Model Type** | Dropdown | SDXL, SD1.5, or auto-detect |
| **Download Location** | Display | Shows where model will be saved |
| **HF Token** | Text Field | Token for private HuggingFace models |

### VAE Downloads
| Parameter | Type | Description |
|-----------|------|-------------|
| **VAE URL** | Text Field | URL to VAE model for improved quality |
| **Auto-detect VAE** | Checkbox | Automatically choose appropriate VAE |

## Training Monitor Widget

### Training Control
| Parameter | Type | Description |
|-----------|------|-------------|
| **Start Training** | Button | Begin LoRA training with current settings |
| **Stop Training** | Button | Halt training process |
| **Training Status** | Display | Current training phase and progress |

### Progress Monitoring
| Parameter | Type | Description |
|-----------|------|-------------|
| **Progress Bar** | Visual | Overall training progress |
| **Current Epoch** | Display | Current epoch number |
| **Current Step** | Display | Current step within epoch |
| **Loss Values** | Display | Training loss metrics |
| **Time Remaining** | Display | Estimated time to completion |
| **GPU Usage** | Display | VRAM and GPU utilization |

### Training Logs
| Parameter | Type | Description |
|-----------|------|-------------|
| **Live Log Output** | Text Area | Real-time training logs |
| **Error Detection** | Visual | Highlights errors and warnings |
| **Log Filter** | Dropdown | Filter log messages by type |

## Utilities Widget

### LoRA Management
| Parameter | Type | Description |
|-----------|------|-------------|
| **Source LoRA** | File Browser | Select LoRA file to modify |
| **Target Dimension** | Number Field | New dimension for resizing |
| **Target Alpha** | Number Field | New alpha value |
| **Output Name** | Text Field | Name for modified LoRA |

### Conversion Tools
| Parameter | Type | Description |
|-----------|------|-------------|
| **Input Format** | Dropdown | Source LoRA format |
| **Output Format** | Dropdown | Target LoRA format |
| **Conversion Type** | Dropdown | Type of conversion to perform |

### Upload Tools
| Parameter | Type | Description |
|-----------|------|-------------|
| **HuggingFace Repo** | Text Field | Repository name for upload |
| **Model Card** | Text Area | Description and documentation |
| **Tags** | Text Field | Comma-separated tags for discovery |
| **HF Token** | Text Field | Authentication token for upload |

### File Management
| Parameter | Type | Description |
|-----------|------|-------------|
| **Cleanup Options** | Checkboxes | Select temporary files to remove |
| **Backup Location** | Text Field | Directory for project backups |
| **Archive Format** | Dropdown | Format for backup archives |

## Calculator Widget (LoRA_Calculator_Widget.ipynb)

### Input Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| **Number of Images** | Number Field | Total images in your dataset |
| **Batch Size** | Number Field | Images processed per batch |
| **Dataset Repeats** | Number Field | How many times to repeat dataset per epoch |
| **Max Epochs** | Number Field | Maximum number of training epochs |
| **Gradient Accumulation** | Number Field | Steps to accumulate gradients |

### Output Calculations
| Parameter | Type | Description |
|-----------|------|-------------|
| **Total Steps** | Display | Calculated total training steps |
| **Steps per Epoch** | Display | Steps in each epoch |
| **Estimated Time** | Display | Approximate training duration |
| **Memory Estimate** | Display | Estimated VRAM requirements |

### Recommendations
| Parameter | Type | Description |
|-----------|------|-------------|
| **Step Count Assessment** | Display | Whether step count is appropriate |
| **Training Time Warning** | Display | Alerts for very long/short training |
| **Memory Warnings** | Display | VRAM requirement alerts |
| **Optimization Suggestions** | Display | Parameter adjustment recommendations |

## Parameter Interaction Guide

### Critical Parameter Relationships

**Learning Rate ↔ Optimizer:**
- AdamW: 1e-4 to 5e-4 typical
- CAME: Often works with higher rates
- Prodigy: Auto-adjusts, start with 1.0

**Network Dimension ↔ Alpha:**
- Common ratios: alpha = dim/2 (8→4, 16→8)
- Higher alpha = stronger effect
- dim/alpha = 1 for experimental approaches

**Batch Size ↔ VRAM:**
- 1024px + batch_size=1: ~6-8GB VRAM
- 768px + batch_size=2: ~6-8GB VRAM  
- 512px + batch_size=4: ~4-6GB VRAM

**Resolution ↔ Training Speed:**
- 512px: Fastest, good for testing
- 768px: Balanced quality/speed
- 1024px: Best quality, slowest

### Compatibility Matrix

| Network Type | Best For | Complexity | File Size |
|--------------|----------|------------|-----------|
| **LoRA** | General use, characters | Low | Small |
| **DoRA** | High quality, characters | Medium | Medium |
| **LoKr** | Efficient styles | Medium | Small |
| **LoHa** | Alternative approach | Medium | Medium |
| **(IA)³** | Minimal intervention | Low | Very Small |
| **BOFT** | Experimental | High | Variable |
| **GLoRA** | Generalized approach | High | Large |

---

*Use this reference to understand each parameter's purpose and optimal settings for your specific use case.*