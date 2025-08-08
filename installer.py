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
    
    def _check_jupyter_running(self):
        """Check if any form of Jupyter is already running"""
        try:
            # Check for running processes
            jupyter_processes = [
                'jupyter notebook',
                'jupyter lab', 
                'jupyter server',
                'jupyter-notebook',
                'jupyter-lab'
            ]
            
            for process in jupyter_processes:
                result = subprocess.run(['pgrep', '-f', process], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return True, process
            
            # Check if port 8888 is in use (common Jupyter port)
            port_checks = [
                ['netstat', '-tlnp'],
                ['lsof', '-ti:8888']
            ]
            
            for cmd in port_checks:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        if cmd[0] == 'netstat' and ':8888' in result.stdout:
                            return True, "port 8888 (netstat)"
                        elif cmd[0] == 'lsof' and result.stdout.strip():
                            return True, "port 8888 (lsof)"
                except:
                    continue
                    
            return False, None
            
        except Exception:
            # If we can't check, assume not running
            return False, None
    
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
            'ffmpeg': 'FFmpeg for video/audio processing (optional, but recommended for some datasets)',
            'git-lfs': 'Git Large File Storage (optional, but recommended for large model downloads)',
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
                    elif cmd == 'ffmpeg':
                        subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
                    elif cmd == 'git-lfs':
                        subprocess.run(['apt-get', 'install', '-y', 'git-lfs'], check=True)
                print("   ✅ Dependencies installed successfully")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Try yum (CentOS/RHEL)
                try:
                    for cmd in missing:
                        if cmd == 'git-lfs':
                            subprocess.run(['yum', 'install', '-y', 'git-lfs'], check=True)
                        else:
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

    def check_pytorch_gpu_support(self):
        """Check PyTorch GPU support after installation (NVIDIA CUDA or AMD ROCm)"""
        print("⚡ Checking PyTorch GPU support...")
        try:
            import torch
            
            # Check for AMD ROCm support
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                print("   🔥 AMD ROCm PyTorch detected!")
                if torch.cuda.is_available():  # ROCm uses CUDA interface
                    device_count = torch.cuda.device_count()
                    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                    print(f"   ✅ ROCm GPU support working: {device_count} device(s) - {device_name}")
                    print(f"   📊 HIP version: {torch.version.hip}")
                else:
                    print("   ⚠️  ROCm PyTorch installed but GPU not accessible")
                    print("      Check ROCm drivers and HSA_OVERRIDE_GFX_VERSION environment variable")
                return
            
            # Check for NVIDIA CUDA support
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count() 
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                print(f"   ✅ NVIDIA CUDA support working: {device_count} device(s) - {device_name}")
                print(f"   📊 CUDA version: {torch.version.cuda}")
                
                # Check CuDNN
                if torch.backends.cudnn.is_available():
                    print("   ✅ CuDNN is available and working")
                else:
                    print("   ⚠️  CuDNN is NOT available - may affect training performance")
                    print("      Consider reinstalling PyTorch with proper CuDNN support")
            else:
                print("   ❌ No GPU acceleration detected in PyTorch")
                print("      Training will use CPU (very slow) - consider installing GPU-enabled PyTorch")
                
        except ImportError:
            print("   ⚠️  PyTorch not found - will be installed during backend setup")
        except Exception as e:
            print(f"   ❌ Error during GPU support check: {e}")

    def install_pillow_smart(self):
        """Smart Pillow installation with fallbacks for different environments"""
        print("🖼️  Installing Pillow with smart fallbacks...")
        
        # Check if already working
        try:
            import PIL
            from PIL import Image
            print(f"   ✅ Pillow {PIL.__version__} already working")
            return True
        except ImportError:
            pass
        
        # Installation methods in order of preference
        methods = [
            ("pre-compiled wheels", [self.python_cmd, '-m', 'pip', 'install', '--only-binary=all', 'Pillow<10.0.0']),
            ("standard install", [self.python_cmd, '-m', 'pip', 'install', 'Pillow<10.0.0']),
            ("no cache", [self.python_cmd, '-m', 'pip', 'install', '--no-cache-dir', 'Pillow<10.0.0']),
            ("specific version 9.4.0", [self.python_cmd, '-m', 'pip', 'install', 'Pillow==9.4.0']),
            ("specific version 9.3.0", [self.python_cmd, '-m', 'pip', 'install', 'Pillow==9.3.0'])
        ]
        
        # If in container, try installing system deps first
        if self.is_container:
            print("   🐳 Container detected - attempting system dependencies...")
            try:
                # Try apt-get
                subprocess.run(['apt-get', 'update'], check=True, capture_output=True)
                deps = ['libjpeg-dev', 'zlib1g-dev', 'libtiff-dev', 'libfreetype6-dev', 'liblcms2-dev', 'libwebp-dev']
                subprocess.run(['apt-get', 'install', '-y'] + deps, check=True, capture_output=True)
                print("   ✅ System dependencies installed")
                # Add force reinstall method after system deps
                methods.insert(1, ("force reinstall", [self.python_cmd, '-m', 'pip', 'install', '--force-reinstall', 'Pillow<10.0.0']))
            except:
                print("   ⚠️  Could not install system dependencies, trying pre-compiled only")
        
        # Try each method
        for method_name, cmd in methods:
            print(f"   🔄 Trying: {method_name}")
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # Verify it works
                import importlib
                if 'PIL' in sys.modules:
                    del sys.modules['PIL']
                import PIL
                from PIL import Image
                print(f"   ✅ Pillow {PIL.__version__} installed successfully using {method_name}")
                return True
            except Exception as e:
                print(f"   ❌ {method_name} failed")
                continue
        
        # All methods failed
        print("   ❌ All Pillow installation methods failed!")
        print("   📝 Manual steps to try:")
        if self.is_container:
            print("      - Run: apt-get update && apt-get install -y libjpeg-dev zlib1g-dev")
        print("      - Try: pip install --only-binary=all Pillow")
        print("      - Consider using a base image with pre-installed dependencies")
        return False

    def check_and_fix_requirements(self):
        """Check requirements and automatically fix any issues"""
        print("🔍 Checking existing requirements...")
        
        # Define required packages and their fixes
        required_packages = {
            'PIL': {'name': 'Pillow', 'fixer': self.install_pillow_smart},
            'ipywidgets': {'name': 'ipywidgets', 'fixer': None},
            'IPython': {'name': 'IPython', 'fixer': None},
            'toml': {'name': 'toml', 'fixer': None},
            'requests': {'name': 'requests', 'fixer': None},
            'tqdm': {'name': 'tqdm', 'fixer': None},
            'hf-transfer': {'name': 'hf-transfer', 'fixer': None},
            'torchvision': {'name': 'torchvision', 'fixer': None},
        }
        
        missing_packages = []
        working_packages = []
        
        for import_name, config in required_packages.items():
            try:
                if import_name == 'PIL':
                    # Special test for Pillow functionality
                    from PIL import Image, ImageDraw
                    img = Image.new('RGB', (10, 10), color='red')
                    working_packages.append(config['name'])
                    print(f"   ✅ {config['name']} working")
                else:
                    __import__(import_name)
                    working_packages.append(config['name'])
                    print(f"   ✅ {config['name']} working")
            except ImportError:
                missing_packages.append((import_name, config))
                print(f"   ❌ {config['name']} missing")
            except Exception as e:
                missing_packages.append((import_name, config))
                print(f"   ⚠️  {config['name']} installed but not working: {e}")
        
        # Auto-fix missing packages
        if missing_packages:
            print(f"\n🔧 Auto-fixing {len(missing_packages)} package issues...")
            
            for import_name, config in missing_packages:
                if config['fixer']:
                    print(f"   🔄 Fixing {config['name']}...")
                    if config['fixer']():
                        print(f"   ✅ {config['name']} fixed!")
                    else:
                        print(f"   ❌ Could not fix {config['name']}")
                else:
                    print(f"   📦 Will install {config['name']} with other packages")
            
            # Collect packages that need pip install
            pip_packages = [config['name'] for _, config in missing_packages if not config['fixer']]
            if pip_packages:
                print(f"   📦 Installing remaining packages: {', '.join(pip_packages)}")
                return pip_packages
        else:
            print("   🎉 All requirements already satisfied!")
        
        return []

    def install_jupyter(self):
        """Install Jupyter extensions and required packages (container-friendly)"""
        if self.is_container:
            print("📚 Installing Jupyter extensions and core packages (container mode)...")
            print("   🐳 Container detected - assuming Jupyter is already available")
        else:
            print("📚 Installing Jupyter and core packages...")
        
        # First, check what's already working and fix what's broken
        missing_packages = self.check_and_fix_requirements()
        
        # Install any remaining packages (but not jupyter itself in containers)
        if missing_packages:
            # In containers, assume jupyter is already installed, don't try to install it
            if self.is_container:
                missing_packages = [pkg for pkg in missing_packages if pkg != 'jupyter']
                if not missing_packages:
                    print("   ✅ All required packages already available in container")
                    return
            
            cmd = [self.python_cmd, '-m', 'pip', 'install', '--upgrade'] + missing_packages
            
            try:
                subprocess.run(cmd, check=True)
                print("   ✅ Remaining packages installed")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install packages: {e}")
                sys.exit(1)
        
        print("   🎉 All core packages ready!")
    
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
        """Create convenient launch script with smart Jupyter detection"""
        print("🚀 Creating launch script...")
        
        launch_script = self.project_root / 'start_jupyter.sh'
        
        # Universal script that works for both containers and local
        launch_content = f'''#!/bin/bash
# LoRA Easy Training - Smart Jupyter Launch Script

echo "🚀 LoRA Easy Training - Jupyter Widget Edition"
echo ""

# Navigate to project directory
cd "{self.project_root}"

# Function to check if Jupyter is running
check_jupyter_running() {{
    # Check for various Jupyter processes
    if pgrep -f "jupyter notebook" > /dev/null 2>&1 || \
       pgrep -f "jupyter lab" > /dev/null 2>&1 || \
       pgrep -f "jupyter server" > /dev/null 2>&1 || \
       pgrep -f "jupyter-notebook" > /dev/null 2>&1 || \
       pgrep -f "jupyter-lab" > /dev/null 2>&1; then
        return 0
    fi
    
    # Check if port 8888 is in use
    if command -v netstat > /dev/null 2>&1; then
        if netstat -tlnp 2>/dev/null | grep -q ":8888 "; then
            return 0
        fi
    fi
    
    if command -v lsof > /dev/null 2>&1; then
        if lsof -ti:8888 > /dev/null 2>&1; then
            return 0
        fi
    fi
    
    return 1
}}

# Enable widget extensions (safe to run multiple times)
echo "🔧 Enabling Jupyter widgets extensions..."
if command -v jupyter &> /dev/null; then
    jupyter nbextension enable --py --sys-prefix widgetsnbextension 2>/dev/null || echo "   ℹ️  Extensions already enabled or not needed"
else
    echo "   ⚠️  Jupyter command not found - extensions may need manual setup"
fi

echo ""
echo "📝 Your notebooks are ready:"
echo "   1. Dataset_Maker_Widget.ipynb - Dataset preparation"
echo "   2. Lora_Trainer_Widget.ipynb - SDXL Training"
echo "   3. Lora_Trainer_Widget_SD15.ipynb - SD 1.5 Training"
echo ""

# Smart environment detection and Jupyter handling
if [[ -n "$VAST_CONTAINERLABEL" ]] || [[ "$PWD" == *"/workspace"* ]] || [[ -n "$RUNPOD_POD_ID" ]] || [[ -f "/.dockerenv" ]]; then
    echo "🐳 Container environment detected"
    
    if check_jupyter_running; then
        echo "✅ Jupyter is already running!"
        echo "   • Check your container provider for the access URL"
        echo "   • VastAI: Look for 'Jupyter' in your instance page"
        echo "   • RunPod: Check the 'Connect' tab for Jupyter link"
        echo "   • Usually accessible on port 8888"
    else
        echo "❓ Jupyter doesn't appear to be running"
        echo "   • This might be a bare container"
        echo "   • You can start Jupyter manually with:"
        echo "     jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    fi
    
else
    echo "💻 Local environment detected"
    
    if check_jupyter_running; then
        echo "✅ Jupyter is already running!"
        echo "   • Check http://localhost:8888 in your browser"
        echo "   • Or look for the terminal where you started Jupyter"
    else
        if command -v jupyter &> /dev/null; then
            echo "🚀 Starting Jupyter notebook (will open browser)..."
            jupyter notebook
        else
            echo "❌ Jupyter not found. Please install it first:"
            echo "   pip install jupyter"
        fi
    fi
fi

echo ""
echo "🎯 Ready to start training! Open the notebooks above in your browser."
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
        
        if self.is_container:
            print("🐳 Container Mode Setup Complete!")
            print("   ./start_jupyter.sh  # Enable extensions and show instructions")
            print()
            print("📝 Your notebooks are ready:")
            print("   1. Dataset_Maker_Widget.ipynb - Dataset preparation")
            print("   2. Lora_Trainer_Widget.ipynb - SDXL Training")
            print("   3. Lora_Trainer_Widget_SD15.ipynb - SD 1.5 Training")
            print()
            
            if self.is_vastai:
                print("☁️  VastAI Container:")
                print("   • Jupyter should already be running on port 8888")
                print("   • Check your VastAI instance page for the Jupyter URL")
                print("   • No need to start Jupyter manually!")
            else:
                print("🐳 Container Environment:")
                print("   • Jupyter is likely already running (check port 8888)")
                print("   • Check your container provider for access instructions")
                print("   • RunPod: Look for 'Connect' tab → Jupyter")
            
        else:
            print("💻 Local Mode Setup Complete!")
            print("   ./start_jupyter.sh  # Start Jupyter and open browser")
            print()
            print("📚 Or manually:")
            print("   jupyter notebook")
            print()
            print("📝 Open these notebooks:")
            print("   1. Dataset_Maker_Widget.ipynb - Dataset preparation")
            print("   2. Lora_Trainer_Widget.ipynb - SDXL Training") 
            print("   3. Lora_Trainer_Widget_SD15.ipynb - SD 1.5 Training")
        
        print()
        print("🎯 Next steps:")
        print("   1. Open the training notebook and run the Setup widget")
        print("   2. Follow the notebook instructions step by step")
        print("   3. Happy training!")
        print()
        print("💝 Everything you need is in the notebooks! Remember: 'Either gonna work or blow up!' 😄")
        print()
    
    def run_installation(self):
        """Run the complete installation process"""
        try:
            self.print_banner()
            self.check_prerequisites()
            self.check_gpu()                    # Check hardware first
            self.install_jupyter()              # Install PyTorch and dependencies 
            self.setup_directories()            # Set up project structure
            self.install_aria2c()               # Install download tools
            self.check_pytorch_gpu_support()    # NOW check PyTorch GPU support (after PyTorch is installed!)
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