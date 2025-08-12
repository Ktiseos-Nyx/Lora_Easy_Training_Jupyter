# 📖 Dataset Notebook Guide

## Comprehensive Guide

This comprehensive widget handles all dataset preparation tasks:

### 📁 Dataset Upload & Management
- **Local upload**: Drag & drop ZIP files or select folders
- **HuggingFace import**: Direct download from HF datasets
- **Automatic extraction**: Handles ZIP, RAR, 7z archives
- **Folder organization**: Creates proper training structure

### 🏷️ Auto-Tagging Systems

#### WD14 v3 Taggers (Recommended for Anime/Art)
- **wd14-vit-v2**: General purpose, balanced accuracy
- **wd14-convnext-v2**: Higher accuracy, slower
- **wd14-swinv2-v2**: Best for complex scenes
- **ONNX optimization**: 2-3x faster inference
- **Threshold control**: Adjust tag sensitivity

#### BLIP Captioning (Best for Photos/Realistic)
- **Natural language**: Generates descriptive sentences
- **Scene understanding**: Captures context and relationships
- **Perfect for**: Real photos, portraits, landscapes

### ✏️ Caption Management
- **Bulk editing**: Apply changes to all captions
- **Find & replace**: Update specific terms across dataset
- **Tag frequency analysis**: See most common tags
- **Manual editing**: Individual caption refinement

### 🎯 Trigger Word System
- **Automatic injection**: Add trigger words to all captions
- **Position control**: Beginning, end, or random placement
- **Consistency checking**: Ensure all images have triggers
- **Preview mode**: See changes before applying

### 🚫 Tag Filtering & Blacklists
- **Quality filters**: Remove low-confidence tags
- **Content filters**: Block unwanted content types
- **Custom blacklists**: Your own forbidden tags
- **Whitelist mode**: Only allow specific tags

### 📈 Quality Analysis
- **Image statistics**: Resolution, format, size analysis
- **Tag distribution**: Most/least common tags
- **Caption length**: Optimal length recommendations
- **Duplicate detection**: Find similar images

### 💡 Best Practices Tips

**Dataset Size:**
- **Characters**: 15-50 images work well
- **Styles**: 50-200 images for consistency
- **Concepts**: 20-100 images depending on complexity

**Image Quality:**
- **Resolution**: 768x768 minimum, 1024x1024 recommended
- **Variety**: Different poses, angles, expressions
- **Consistency**: Similar art style or character design

### 🛠️ Recommended Image Preparation Tools

**For Bulk Resizing & Cropping:**
- **[birme.net](https://www.birme.net/)** - **Highly Recommended!**
  - Bulk resize multiple images to exact dimensions (512x512, 1024x1024)
  - Smart cropping with auto focal point detection
  - Privacy-focused - images never leave your computer
  - Perfect for dataset standardization

**For Advanced Editing:**
- **[photopea.com](https://www.photopea.com/)** - Free Photoshop alternative
  - Browser-based, no downloads needed
  - Full PSD support and professional tools
  - Great for background removal, cleanup, advanced edits
  - Perfect for preparing images before adding to dataset

**Pro Tip:** Use birme.net first for bulk resizing, then photopea.com for any individual images that need special attention!

**Caption Quality:**
- **Accuracy**: Verify auto-generated tags
- **Completeness**: Include important details
- **Trigger words**: Use unique, memorable terms
- **Consistency**: Same terms for same concepts
