# Troubleshooting Guide

This guide provides solutions to the most common errors and issues you might encounter while using the LoRA Easy Training system.

## Installation Issues

### "Module not found" errors
- **Cause**: Python dependencies not properly installed
- **Solution**: 
  1. Re-run the installer: `python installer.py`
  2. Ensure you're using the correct Python version (3.10+)
  3. Check that you're in the project directory

### "Jupyter not found" error  
- **Cause**: Jupyter Lab/Notebook not installed on system
- **Solution**: 
  1. Install Jupyter: `pip install jupyterlab` or `pip install notebook`
  2. Alternatively, use Anaconda which includes Jupyter

### Installer fails with download errors
- **Cause**: Network connection or storage issues
- **Solution**:
  1. Check internet connection
  2. Ensure sufficient disk space (15-20GB free)
  3. Try running installer again (it will resume downloads)

## Training Errors

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`
- **Cause**: Training process trying to use more VRAM than available
- **Solutions** (try in order):
  1. **Reduce Batch Size**: Set to 1 in Training Widget
  2. **Lower Resolution**: Use 768x768 instead of 1024x1024
  3. **Reduce Network Dimension**: Try 4 or 6 instead of 8
  4. **Enable Advanced Options**: Use gradient checkpointing
  5. **Try CAME Optimizer**: Uses less memory than AdamW

### Loss becomes NaN (Not a Number)

**Error**: Training loss shows `NaN` values
- **Cause**: Learning rate too high, causing training to collapse
- **Solutions**:
  1. **Lower Learning Rate**: Cut by factor of 10 (e.g., 5e-4 → 5e-5)
  2. **Use Adaptive Optimizers**: Switch to Prodigy or CAME
  3. **Check Data**: Ensure no corrupted images in dataset
  4. **Reduce Network Dimension**: Try smaller LoRA rank

### Black/Gray/Washed-out Output Images

**Symptoms**: Generated images are monochrome or severely desaturated
- **Causes and Solutions**:
  1. **V-Prediction Models**: Enable `v_prediction` checkbox if using anime/v-pred models
  2. **Learning Rate Too High**: Lower learning rates significantly
  3. **Wrong Base Model**: Ensure model architecture matches training settings
  4. **Overtraining**: Stop training earlier, monitor loss curves

### Training Extremely Slow

**Symptoms**: Training taking much longer than expected
- **Causes and Solutions**:
  1. **Large Images**: Reduce resolution to 512x512 or 768x768
  2. **Large Batch Size**: Start with batch size of 1-2
  3. **CPU Training**: Ensure CUDA is properly installed and detected
  4. **Background Processes**: Close other GPU-intensive applications

### "No module named 'torch'" or similar

**Error**: PyTorch/training backend not found
- **Solutions**:
  1. **Run Environment Setup**: Use Cell 1 in `Lora_Trainer_Widget.ipynb`
  2. **Check Installation**: Re-run the installer script
  3. **Manual Install**: Install PyTorch manually for your system

## Dataset Issues

### "No images found" in dataset

**Error**: Widget reports empty dataset
- **Causes and Solutions**:
  1. **Wrong Path**: Check dataset path is correct
  2. **Unsupported Formats**: Only JPG, PNG, WebP supported
  3. **ZIP Structure**: Images should be in root folder of ZIP or single subfolder
  4. **File Permissions**: Ensure files are readable

### Auto-tagging Fails

**Error**: WD14 or BLIP tagging doesn't work
- **Causes and Solutions**:
  1. **Network Issues**: Check internet connection for model downloads
  2. **Disk Space**: Need 2-3GB free for tagger models
  3. **Memory Issues**: Reduce image resolution or batch size
  4. **Model Access**: Some models require HuggingFace login

### Captions Too Long/Short

**Issue**: Generated captions have too many or too few tags
- **Solutions**:
  1. **Adjust Threshold**: Lower for more tags (0.3), higher for fewer (0.5)
  2. **Filter Tags**: Use blacklist to remove unwanted tags
  3. **Manual Editing**: Edit important captions by hand
  4. **Different Model**: Try another WD14 variant

### Missing Trigger Words

**Issue**: Some images don't have trigger words in captions
- **Solutions**:
  1. **Bulk Add**: Use "Add Trigger Word" tool in dataset widget
  2. **Check Filters**: Ensure trigger word isn't in blacklist
  3. **Manual Check**: Verify a few files manually
  4. **Re-apply**: Run trigger word addition again

## Widget/Interface Issues

### Widgets Not Displaying

**Issue**: Empty cells or widgets don't appear
- **Solutions**:
  1. **Restart Kernel**: Kernel → Restart in Jupyter
  2. **Clear Output**: Cell → All Output → Clear
  3. **Re-run Cell**: Run the widget cell again
  4. **Check Dependencies**: Ensure ipywidgets is installed

### "Shared managers not found" error

**Error**: Cannot import shared_managers
- **Solutions**:
  1. **Check Working Directory**: Ensure you're in the project root
  2. **File Paths**: Verify all project files are present
  3. **Restart Kernel**: Fresh start often fixes import issues

### Progress Bars Stuck

**Issue**: Training progress bars don't update
- **Solutions**:
  1. **Refresh Browser**: Sometimes helps with display issues
  2. **Check Logs**: Look at text output for actual progress
  3. **System Resources**: Ensure system isn't overloaded

## Performance Issues

### Training Much Slower Than Expected

**Optimizations to try**:
1. **Use CAME Optimizer**: Often faster than AdamW
2. **Optimal Batch Size**: Find sweet spot for your GPU (usually 2-4)
3. **Resolution**: Train at 768x768 instead of 1024x1024
4. **Close Other Apps**: Free up system resources
5. **Check GPU Usage**: Monitor with `nvidia-smi`

### High Memory Usage

**Memory optimization strategies**:
1. **Gradient Checkpointing**: Enable in advanced options
2. **Mixed Precision**: Use fp16 or bf16 if supported
3. **Smaller Networks**: Reduce LoRA dimension
4. **CAME Optimizer**: Uses significantly less memory
5. **Close Jupyter Tabs**: Reduce browser memory usage

## Model Quality Issues

### LoRA Doesn't Work as Expected

**Common issues and fixes**:
1. **Trigger Word**: Ensure you're using the exact trigger word from training
2. **Model Compatibility**: Check base model matches what you trained on
3. **LoRA Strength**: Try different strength values (0.7-1.2)
4. **Undertrained**: May need more training steps
5. **Overtrained**: Try earlier checkpoints if available

### Results Look Like Base Model

**Issue**: LoRA has minimal effect
- **Solutions**:
  1. **Increase LoRA Strength**: Try 1.0-1.5 in your inference UI
  2. **Check Training**: Ensure loss was decreasing during training
  3. **Trigger Word**: Make sure you're using it in prompts
  4. **Network Settings**: Try higher network dimension (16 instead of 8)

### Inconsistent Quality

**Issue**: Some generations good, others poor
- **Solutions**:
  1. **Dataset Quality**: Review training images for consistency
  2. **Caption Quality**: Improve tag accuracy and consistency
  3. **Training Length**: May need more or fewer steps
  4. **Base Model**: Try different base models

## System Requirements Issues

### Insufficient VRAM

**For GPUs with < 8GB VRAM**:
1. **Batch Size**: Always use 1
2. **Resolution**: Train at 512x512 maximum
3. **Network Dimension**: Use 4-6 instead of 8
4. **CAME Optimizer**: Significantly reduces memory usage
5. **Gradient Checkpointing**: Enable in advanced settings

### Slow Training on Older Hardware

**Optimizations for older systems**:
1. **Lower Resolution**: 512x512 trains much faster
2. **Smaller Datasets**: 15-20 images instead of 50+
3. **Efficient Settings**: Use proven parameter combinations
4. **Close Background Apps**: Maximize available resources

## Getting Help

### Where to Get Support

1. **Check Documentation**: Review relevant guides first
2. **Discord Community**: Join for real-time help
3. **GitHub Issues**: Report bugs and feature requests
4. **Community Forums**: Civitai and Reddit communities

### Information to Include When Asking for Help

1. **System Specs**: GPU model, VRAM, OS
2. **Error Messages**: Full error text and traceback
3. **Settings Used**: Training parameters and dataset info
4. **Steps to Reproduce**: What you were doing when error occurred
5. **Screenshots**: Visual issues benefit from images

---

*"Either gonna work or blow up!" - Don't worry, most issues have simple solutions!*