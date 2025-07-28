# Terminal Commands for LoRA Training Troubleshooting

*Essential commands to fix common issues without nuking your entire installation*

---

Before you delete everything and start over, try these terminal commands to diagnose and fix common LoRA training issues. This guide covers Windows (Command Prompt/PowerShell), Linux, and macOS.

## 🚨 Before You Start

**Golden Rule**: Always backup your trained LoRAs before running cleanup commands!

**Safety First**: 
- Copy your `output/` folder somewhere safe
- Note down your working training configurations
- Take screenshots of error messages

## 📁 Navigation Basics

### Windows (Command Prompt/PowerShell)
```cmd
# Navigate to project
cd C:\path\to\Lora_Easy_Training_Jupyter

# List files
dir

# Check current location  
cd

# Go up one directory
cd ..

# Create directory
mkdir backup

# Copy files (PowerShell)
Copy-Item "output\*.safetensors" "backup\"
```

### Linux/macOS
```bash
# Navigate to project
cd /path/to/Lora_Easy_Training_Jupyter

# List files (detailed)
ls -la

# Check current location
pwd

# Go up one directory  
cd ..

# Create directory
mkdir backup

# Copy files
cp output/*.safetensors backup/
```

## 🔍 Diagnostic Commands

### Check Python and Package Status

**All Platforms:**
```bash
# Check Python version
python --version
python3 --version

# Check if packages are installed
pip list | grep torch
pip list | grep transformers
pip list | grep diffusers

# Check specific problem packages
python -c "import torch; print(torch.__version__)"
python -c "import pytorch_optimizer; print('pytorch_optimizer OK')"
```

### Check GPU and CUDA Status

**Windows:**
```cmd
# Check NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

**Linux/macOS:**
```bash
# Check NVIDIA GPU (Linux)
nvidia-smi
lspci | grep -i nvidia

# Check CUDA
nvcc --version
which nvcc

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Check Disk Space

**Windows:**
```cmd
# Check disk space
dir /-c
fsutil volume diskfree c:
```

**Linux/macOS:**
```bash
# Check disk space
df -h
du -sh * | sort -hr
```

## 🔧 Common Fix Commands

### Fix Import Errors

**Missing pytorch_optimizer:**
```bash
# Install missing package
pip install pytorch_optimizer>=3.1.0

# If that fails, try with user flag
pip install --user pytorch_optimizer>=3.1.0

# Force reinstall
pip install --force-reinstall pytorch_optimizer>=3.1.0
```

**General import failures:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Clear pip cache
pip cache purge

# Upgrade pip first
python -m pip install --upgrade pip
```

### Fix CUDA/PyTorch Issues

**CUDA version mismatch:**
```bash
# Check what CUDA PyTorch was built with
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch for your CUDA version (example for CUDA 11.8)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**No CUDA detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# Restart CUDA service (Windows - run as admin)
net stop nvdisplay.exe
net start nvdisplay.exe

# Linux - restart display manager
sudo systemctl restart gdm  # or lightdm, sddm depending on your system
```

### Fix Permission Issues

**Windows (run as Administrator):**
```cmd
# Take ownership of files
takeown /f "C:\path\to\Lora_Easy_Training_Jupyter" /r

# Reset permissions
icacls "C:\path\to\Lora_Easy_Training_Jupyter" /reset /t
```

**Linux/macOS:**
```bash
# Fix ownership
sudo chown -R $USER:$USER /path/to/Lora_Easy_Training_Jupyter

# Fix permissions
chmod -R 755 /path/to/Lora_Easy_Training_Jupyter
chmod +x installer.py
```

### Clean Temporary Files

**All Platforms:**
```bash
# Remove Python cache
find . -type d -name "__pycache__" -delete  # Linux/macOS
# Windows: for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

# Remove .pyc files
find . -name "*.pyc" -delete  # Linux/macOS
# Windows: del /s *.pyc

# Clean pip cache
pip cache purge
```

**Project-specific cleanup:**
```bash
# Remove temporary training files
rm -rf trainer/sd_scripts/venv/  # Linux/macOS
rmdir /s trainer\sd_scripts\venv  # Windows

# Clean Jupyter notebook checkpoints
rm -rf .ipynb_checkpoints/  # Linux/macOS
rmdir /s .ipynb_checkpoints  # Windows
```

## 🚑 Emergency Fixes

### "CUDA Out of Memory" Recovery

```bash
# Clear GPU memory (requires nvidia-ml-py)
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"

# Kill any stuck Python processes
# Windows:
taskkill /im python.exe /f

# Linux/macOS:
pkill -f python
```

### Broken Virtual Environment

**Recreate venv (if using one):**
```bash
# Remove old environment
rm -rf venv/  # Linux/macOS
rmdir /s venv  # Windows

# Create new environment
python -m venv venv

# Activate and reinstall
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Jupyter Kernel Issues

```bash
# Restart Jupyter kernel
jupyter kernelspec list
jupyter kernelspec remove python3  # if problematic
python -m ipykernel install --user --name=python3

# Clear Jupyter cache
jupyter --paths
# Manually delete cache directories shown
```

### Git Repository Issues

```bash
# Reset to last good commit (CAREFUL - this loses changes)
git status
git stash  # save current changes
git reset --hard HEAD

# Pull latest fixes
git pull origin main

# Restore your changes (if needed)
git stash pop
```

## 🔄 Reinstall Strategies (Last Resort)

### Selective Reinstallation

**Just the training backend:**
```bash
# Remove trainer directory
rm -rf trainer/  # Linux/macOS
rmdir /s trainer  # Windows

# Re-run setup
python installer.py
```

**Just Python packages:**
```bash
# Create fresh requirements install
pip freeze > current_packages.txt
pip uninstall -r current_packages.txt -y
pip install -r requirements.txt
```

### Nuclear Option (Complete Reinstall)

**Backup first:**
```bash
# Backup your important files
mkdir ../lora_backup
cp -r output/ ../lora_backup/
cp -r *.ipynb ../lora_backup/
cp -r docs/ ../lora_backup/  # if you modified docs

# Then delete and reclone
cd ..
rm -rf Lora_Easy_Training_Jupyter
git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
cd Lora_Easy_Training_Jupyter

# Restore your files
cp -r ../lora_backup/output/ .
cp -r ../lora_backup/*.ipynb .
```

## 📊 System Monitoring Commands

### Resource Usage

**Windows:**
```cmd
# CPU and memory usage
tasklist /fo csv | findstr python
wmic process where name="python.exe" get processid,pagefileusage

# GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv --loop=1
```

**Linux/macOS:**
```bash
# CPU and memory
top -p $(pgrep python)
htop -p $(pgrep python)

# GPU usage (Linux)
watch -n 1 nvidia-smi

# Disk I/O
iotop -p $(pgrep python)
```

### Network Issues

```bash
# Test internet connectivity
ping google.com
curl -I https://huggingface.co

# Test specific model downloads
curl -I https://huggingface.co/runwayml/stable-diffusion-v1-5

# Check DNS
nslookup huggingface.co
```

## 🔍 Log File Analysis

### Find Error Messages

**Windows:**
```cmd
# Search for errors in output
findstr /i "error" *.log
findstr /i "failed" *.log
findstr /i "cuda" *.log
```

**Linux/macOS:**
```bash
# Search for errors
grep -i "error" *.log
grep -i "failed" *.log  
grep -i "cuda" *.log

# Search in all files recursively
grep -r "CUDA out of memory" .
grep -r "ModuleNotFoundError" .
```

### Jupyter Log Files

```bash
# Find Jupyter runtime directory
jupyter --runtime-dir

# Check kernel logs
ls $(jupyter --runtime-dir)
cat $(jupyter --runtime-dir)/kernel-*.json
```

## 🛠️ Advanced Debugging

### Python Environment Debugging

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check site-packages location
python -m site

# Verify package locations
python -c "import torch; print(torch.__file__)"
python -c "import transformers; print(transformers.__file__)"
```

### Memory Debugging

```bash
# Check system memory
# Windows:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory

# Linux/macOS:
free -h  # Linux
vm_stat | head -10  # macOS

# Check swap usage
swapon -s  # Linux
swap_usage  # macOS
```

## 📚 Quick Reference Cards

### Windows Quick Fixes
```cmd
REM Basic diagnostics
python --version && pip --version
nvidia-smi
dir output

REM Common fixes
pip install --force-reinstall pytorch_optimizer
taskkill /im python.exe /f
rmdir /s __pycache__
```

### Linux/macOS Quick Fixes
```bash
# Basic diagnostics
python3 --version && pip --version
nvidia-smi
ls -la output/

# Common fixes
pip install --force-reinstall pytorch_optimizer
pkill -f python
find . -name "__pycache__" -delete
```

## 🎯 Prevention Tips

### Regular Maintenance

**Weekly:**
```bash
# Update packages
pip install --upgrade pip
pip list --outdated

# Clean cache
pip cache purge
```

**Before Big Training:**
```bash
# Check disk space
df -h  # Linux/macOS
fsutil volume diskfree c:  # Windows

# Check GPU memory
nvidia-smi

# Backup current LoRAs
cp output/*.safetensors backup/
```

### Environment Management

```bash
# Document working configuration
pip freeze > working_requirements.txt
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" >> system_info.txt

# Test environment health
python -c "
import torch, transformers, diffusers
print('✅ All major packages imported successfully')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
"
```

---

## 💡 Pro Tips

1. **Document everything**: Keep notes of what commands fixed what issues
2. **Test small first**: Before big training runs, do quick tests
3. **Monitor resources**: Keep an eye on disk space and memory usage
4. **Backup regularly**: Your trained LoRAs are valuable!
5. **Update gradually**: Don't update everything at once

## 🆘 When All Else Fails

1. **Discord/Community**: Ask for help with specific error messages
2. **GitHub Issues**: Report bugs with system info and error logs  
3. **Fresh VM/Container**: Sometimes a clean environment is fastest
4. **Different Hardware**: Test on another machine if available

---

*"The terminal is your friend - it tells you exactly what's wrong instead of just saying 'something broke'!"*

**Remember**: These commands are your toolkit for avoiding the nuclear option of complete reinstallation. Most issues can be fixed with targeted commands rather than starting from scratch! 🔧