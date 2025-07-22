#!/usr/bin/env python3
"""
LoRA Easy Training - Jupyter Widget Edition Installer
Comprehensive installation script for all environments
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

class LoRATrainingInstaller:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.python_cmd = self._get_python_command()
        self.is_vastai = self._detect_vastai()
        self.is_container = self._detect_container()
        
    def _get_python_command(self):
        """Detect the best Python command to use"""
        for cmd in ['python3.11', 'python3.10', 'python3', 'python']:
            if shutil.which(cmd):
                try:
                    result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        print(f"🐍 Found Python: {version}")
                        return cmd
                except:
                    continue
        raise RuntimeError("❌ Python 3.10+ not found. Please install Python first.")
    
    def _detect_vastai(self):
        """Detect if running on VastAI"""
        return bool(os.environ.get('VAST_CONTAINERLABEL') or '/workspace' in str(self.project_root))
    
    def _detect_container(self):
        """Detect if running in any container"""
        return bool(
            os.path.exists('/.dockerenv') or 
            os.environ.get('CONTAINER') == 'docker' or
            self.is_vastai
        )
    
    def print_banner(self):
        """Print installation banner"""
        print("=" * 60)
        print("🚀 LoRA Easy Training - Jupyter Widget Edition")
        print("   Comprehensive LoRA Training System")
        print("=" * 60)
        print()
        
        if self.is_vastai:
            print("☁️  VastAI Container Detected - Optimized installation")
        elif self.is_container:
            print("📦 Container Environment Detected")
        else:
            print("💻 Local Environment Detected")
        
        print(f"🐍 Python: {self.python_cmd}")
        print(f"📁 Install Path: {self.project_root}")
        print()
    
    def check_prerequisites(self):
        """Check and install prerequisites"""
        print("🔍 Checking prerequisites...")
        
        # Check for required commands
        required_commands = {
            'git': 'Git version control',
            'curl': 'HTTP client for downloads',
        }
        
        missing = []
        for cmd, desc in required_commands.items():
            if shutil.which(cmd):
                print(f"   ✅ {cmd} - {desc}")
            else:
                print(f"   ❌ {cmd} - {desc} (missing)")
                missing.append(cmd)
        
        # Try to install missing commands on supported systems
        if missing and (self.is_container or platform.system() == 'Linux'):
            print(f"📦 Installing missing dependencies: {', '.join(missing)}")
            try:
                # Try apt-get first (Ubuntu/Debian)
                subprocess.run(['apt-get', 'update'], check=True, capture_output=True)
                for cmd in missing:
                    if cmd == 'git':
                        subprocess.run(['apt-get', 'install', '-y', 'git'], check=True)
                    elif cmd == 'curl':
                        subprocess.run(['apt-get', 'install', '-y', 'curl'], check=True)
                print("   ✅ Dependencies installed successfully")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Try yum (CentOS/RHEL)
                try:
                    for cmd in missing:
                        subprocess.run(['yum', 'install', '-y', cmd], check=True)
                    print("   ✅ Dependencies installed successfully")
                except:
                    print("   ⚠️  Could not auto-install dependencies")
                    print("   📝 Please install manually:", ', '.join(missing))
    
    def check_gpu(self):
        """Check GPU availability"""
        print("🎮 Checking GPU support...")
        
        if os.path.exists('/proc/driver/nvidia/version'):
            print("   ✅ NVIDIA driver detected")
            
            # Try nvidia-smi for detailed info
            if shutil.which('nvidia-smi'):
                try:
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=name,memory.total', 
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\\n'):
                            if line.strip():
                                parts = line.split(', ')
                                if len(parts) >= 2:
                                    name, memory = parts[0], parts[1]
                                    memory_gb = int(memory) / 1024
                                    print(f"   🎮 GPU: {name.strip()} ({memory_gb:.1f} GB VRAM)")
                except Exception as e:
                    print(f"   ⚠️  nvidia-smi error: {e}")
            else:
                print("   ⚠️  nvidia-smi not available")
        else:
            print("   ❌ No NVIDIA driver detected")
            print("   📝 LoRA training requires NVIDIA GPU with 6GB+ VRAM")
    
    def install_jupyter(self):
        """Install Jupyter and required packages"""
        print("📚 Installing Jupyter and core packages...")
        
        # Core packages for the widget system
        packages = [
            'jupyter',
            'ipywidgets',
            'notebook>=6.4.12',
            'ipython',
            'toml',
            'requests',
            'tqdm',
            'Pillow',
        ]
        
        cmd = [self.python_cmd, '-m', 'pip', 'install', '--upgrade'] + packages
        
        try:
            subprocess.run(cmd, check=True)
            print("   ✅ Jupyter and core packages installed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install packages: {e}")
            sys.exit(1)
    
    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Setting up directories...")
        
        directories = [
            'pretrained_model',
            'vae', 
            'tagger_models',
            'output',
            'logs',
            'trainer',
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ Created: {directory}/")
    
    def install_aria2c(self):
        """Install aria2c for fast downloads"""
        print("📥 Installing aria2c for fast downloads...")
        
        if shutil.which('aria2c'):
            print("   ✅ aria2c already installed")
            return
        
        try:
            if platform.system() == 'Linux' or self.is_container:
                # Try apt-get first
                try:
                    subprocess.run(['apt-get', 'install', '-y', 'aria2'], check=True, capture_output=True)
                    print("   ✅ aria2c installed via apt-get")
                    return
                except:
                    pass
                
                # Try yum
                try:
                    subprocess.run(['yum', 'install', '-y', 'aria2'], check=True, capture_output=True)
                    print("   ✅ aria2c installed via yum")
                    return
                except:
                    pass
            
            print("   ⚠️  Could not auto-install aria2c")
            print("   📝 Please install aria2c manually for faster downloads")
            
        except Exception as e:
            print(f"   ⚠️  aria2c installation error: {e}")
    
    def create_launch_script(self):
        """Create convenient launch script"""
        print("🚀 Creating launch script...")
        
        launch_script = self.project_root / 'start_jupyter.sh'
        launch_content = f'''#!/bin/bash
# LoRA Easy Training - Jupyter Launch Script

echo "🚀 Starting LoRA Easy Training - Jupyter Widget Edition"
echo ""

# Navigate to project directory
cd "{self.project_root}"

# Check if Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "❌ Jupyter not found. Please run installer.py first"
    exit 1
fi

# Enable widget extensions
echo "🔧 Enabling Jupyter widgets..."
jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Start Jupyter notebook
echo "📚 Starting Jupyter notebook server..."
echo ""
echo "📝 Open these notebooks to get started:"
echo "   1. Dataset_Maker_Widget.ipynb - for dataset preparation"
echo "   2. Lora_Trainer_Widget.ipynb - for training"
echo ""

# Launch with appropriate settings
if [[ -n "$VAST_CONTAINERLABEL" ]] || [[ "$PWD" == *"/workspace"* ]]; then
    echo "☁️  VastAI detected - using container-optimized settings"
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""
else
    echo "💻 Local environment - opening browser"
    jupyter notebook
fi
'''
        
        with open(launch_script, 'w') as f:
            f.write(launch_content)
        
        # Make executable
        os.chmod(launch_script, 0o755)
        print(f"   ✅ Created launch script: {launch_script}")
    
    def show_completion_message(self):
        """Show installation completion message"""
        print("\n" + "=" * 60)
        print("✅ INSTALLATION COMPLETE!")
        print("=" * 60)
        print()
        print("🚀 To start training:")
        print("   ./start_jupyter.sh")
        print()
        print("📚 Or manually:")
        print("   jupyter notebook")
        print()
        print("📝 Open these notebooks:")
        print("   1. Dataset_Maker_Widget.ipynb - Dataset preparation")
        print("   2. Lora_Trainer_Widget.ipynb - Training & setup")
        print()
        
        if self.is_vastai:
            print("☁️  VastAI Tips:")
            print("   - Port 8888 should be exposed automatically")
            print("   - Use the provided URL to access Jupyter")
            print("   - Popular models are pre-selected for you")
        
        print("🎯 Next steps:")
        print("   1. Run environment setup in the training notebook")
        print("   2. Download a base model (Illustrious, PonyDiff, etc.)")
        print("   3. Prepare your dataset")
        print("   4. Start training!")
        print()
        print("💝 Happy training! Remember: 'Either gonna work or blow up!' 😄")
        print()
    
    def run_installation(self):
        """Run the complete installation process"""
        try:
            self.print_banner()
            self.check_prerequisites()
            self.check_gpu()
            self.install_jupyter()
            self.setup_directories()
            self.install_aria2c()
            self.create_launch_script()
            self.show_completion_message()
            
        except KeyboardInterrupt:
            print("\\n❌ Installation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"\\n❌ Installation failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    installer = LoRATrainingInstaller()
    installer.run_installation()