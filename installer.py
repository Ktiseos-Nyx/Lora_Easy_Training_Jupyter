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
        """Install Jupyter and required packages"""
        print("📚 Installing Jupyter and core packages...")
        
        # First, check what's already working and fix what's broken
        missing_packages = self.check_and_fix_requirements()
        
        # Install any remaining packages
        if missing_packages:
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
        print("🔧 If you encounter issues later:")
        print("   python check_requirements.py  # Re-validate your setup")
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