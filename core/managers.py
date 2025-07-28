# core/managers.py
import subprocess
import os
import sys
import re
import zipfile
import toml
import shutil
from huggingface_hub import HfApi, login

def get_venv_python_path(base_dir):
    """Get cross-platform virtual environment Python path"""
    if sys.platform == "win32":
        # Windows: venv/Scripts/python.exe
        return os.path.join(base_dir, "venv", "Scripts", "python.exe")
    else:
        # Unix/Linux/Mac: venv/bin/python
        return os.path.join(base_dir, "venv", "bin", "python")

class SetupManager:
    def __init__(self):
        # Use current working directory instead of hardcoded paths
        self.project_root = os.getcwd()  # This will be wherever the notebook is running
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        
        # Use Derrian's backend which includes sd_scripts and lycoris as submodules
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")
        
        # Submodule URLs - Only need derrian_backend which includes sd_scripts and lycoris
        self.submodules = {
            'derrian_backend': {
                'url': 'https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend.git',
                'path': self.derrian_dir,
                'description': 'Derrian\'s backend with Kohya scripts, LyCORIS, and custom optimizers'
            }
        }

    def _check_and_install_packages(self):
        packages = ["git", "aria2c"]
        for pkg in packages:
            if not shutil.which(pkg):
                print(f"{pkg} not found. Attempting to install...")
                try:
                    subprocess.run(["apt-get", "update"], check=True)
                    subprocess.run(["apt-get", "install", "-y", pkg], check=True)
                    print(f"{pkg} installed successfully.")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Failed to install {pkg}. Please install it manually. Error: {e}")
                    return False
        return True

    def _setup_submodule(self, name, config):
        """Clone and setup a single submodule"""
        path = config['path']
        url = config['url']
        
        if os.path.exists(path) and os.listdir(path):
            print(f"✅ {config['description']} already exists")
            return True
            
        print(f"📥 Cloning {config['description']}...")
        print(f"   URL: {url}")
        
        try:
            # Remove if exists but empty
            if os.path.exists(path):
                shutil.rmtree(path)
                
            # Clone the repository
            subprocess.run(["git", "clone", url, path], check=True)
            
            # Pin to specific commit if specified
            if 'commit' in config:
                print(f"📌 Pinning to commit: {config['commit'][:8]}...")
                subprocess.run(["git", "checkout", config['commit']], cwd=path, check=True)
            
            # Special handling for derrian_backend - initialize its nested submodules
            if name == 'derrian_backend':
                print("🔗 Initializing nested submodules (sd_scripts, lycoris)...")
                try:
                    # Initialize and update nested submodules
                    subprocess.run(["git", "submodule", "init"], cwd=path, check=True)
                    subprocess.run(["git", "submodule", "update"], cwd=path, check=True)
                    print("✅ Nested submodules initialized")
                    
                    # Run Derrian's installer to set up dependencies (non-interactive mode)
                    print("📦 Running Derrian's installer for dependencies...")
                    installer_path = os.path.join(path, "installer.py")
                    if os.path.exists(installer_path):
                        try:
                            # Use 'local' argument to skip interactive prompts
                            subprocess.run([sys.executable, installer_path, "local"], cwd=path, check=True)
                            print("✅ Derrian's dependencies installed")
                        except subprocess.CalledProcessError as e:
                            print(f"⚠️ Derrian's installer failed: {e}")
                            print("💡 Will continue with manual dependency installation")
                    else:
                        print("⚠️ Derrian's installer not found, skipping dependency setup")
                    
                    # Install custom scheduler optimizers (CAME, REX, etc.)
                    print("📦 Installing custom scheduler optimizers...")
                    custom_scheduler_path = os.path.join(path, "custom_scheduler")
                    setup_py_path = os.path.join(custom_scheduler_path, "setup.py")
                    if os.path.exists(setup_py_path):
                        try:
                            # Proper setup.py install command
                            subprocess.run([sys.executable, "setup.py", "install"], cwd=custom_scheduler_path, check=True)
                            print("✅ Custom scheduler optimizers (LoraEasyCustomOptimizer) installed")
                        except subprocess.CalledProcessError as e:
                            print(f"⚠️ Custom scheduler installation failed: {e}")
                            print("💡 Optimizers exist but may need manual installation")
                    else:
                        print("⚠️ Custom scheduler setup.py not found")
                    
                    # Run PyTorch fix for Windows users (just in case)
                    print("🔧 Running PyTorch fix (Windows compatibility)...")
                    fix_torch_path = os.path.join(path, "fix_torch.py")
                    if os.path.exists(fix_torch_path):
                        try:
                            subprocess.run([sys.executable, fix_torch_path], cwd=path, check=True)
                            print("✅ PyTorch Windows compatibility fix applied")
                        except subprocess.CalledProcessError:
                            print("⚠️ PyTorch fix failed (likely not needed on this system)")
                    else:
                        print("⚠️ PyTorch fix script not found")
                        
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Nested submodule setup failed: {e}")
                    print("💡 You may need to run setup again or manually initialize submodules")
                
            print(f"✅ {config['description']} setup complete")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup {name}: {e}")
            return False
            
    def _setup_all_submodules(self):
        """Setup all required submodules"""
        print("🏗️ Setting up training backend submodules...")
        
        # Ensure trainer directory exists
        os.makedirs(self.trainer_dir, exist_ok=True)
        
        success_count = 0
        for name, config in self.submodules.items():
            if self._setup_submodule(name, config):
                success_count += 1
                
        if success_count == len(self.submodules):
            print(f"🎉 All {len(self.submodules)} submodules setup successfully!")
            return True
        else:
            print(f"⚠️ Only {success_count}/{len(self.submodules)} submodules setup successfully")
            return False
            
    def _setup_custom_optimizers(self):
        """Setup Derrian's custom optimizers with proper attribution"""
        print("🧪 Setting up custom optimizers (CAME, REX)...")
        
        custom_scheduler_dir = os.path.join(self.derrian_dir, "custom_scheduler")
        
        if not os.path.exists(custom_scheduler_dir):
            print("❌ Derrian's custom_scheduler directory not found")
            return False
            
        # Add to Python path for imports
        import sys
        if custom_scheduler_dir not in sys.path:
            sys.path.insert(0, custom_scheduler_dir)
            print(f"📂 Added to Python path: {custom_scheduler_dir}")
        
        # Find and install the custom optimizer package
        setup_py = None
        for root, dirs, files in os.walk(custom_scheduler_dir):
            if 'setup.py' in files:
                setup_py = root
                break
                
        if setup_py:
            print("📦 Installing Derrian's custom optimizers...")
            try:
                # Install in editable mode
                subprocess.run(["pip", "install", "-e", "."], check=True, cwd=setup_py)
                print("✅ Custom optimizers installed successfully")
                
                # Test imports
                try:
                    import LoraEasyCustomOptimizer.came
                    import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
                    print("✅ Custom optimizers verified working")
                    return True
                except ImportError as e:
                    print(f"⚠️ Custom optimizers installed but import failed: {e}")
                    return False
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install custom optimizers: {e}")
                return False
        else:
            print("❌ setup.py not found in custom_scheduler")
            return False


    def _get_python_version(self):
        """Get the current Python version (major.minor)"""
        try:
            import sys
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            return version
        except Exception as e:
            print(f"⚠️ Could not detect Python version: {e}")
            return "3.12"  # Default fallback

    def _is_environment_setup(self):
        """Checks if the core components of the training environment are already in place."""
        # Check 1: Is the backend repo cloned and looks valid?
        if not os.path.exists(self.trainer_dir):
            return False
        try:
            contents = os.listdir(self.trainer_dir)
            junk_files = ['.ipynb_checkpoints', '__pycache__', '.DS_Store', '.git']
            real_files = [f for f in contents if f not in junk_files and not f.startswith('.')]
            expected_files = ['custom_scheduler', 'main.py', 'requirements.txt', 'sd_scripts']
            has_derrian_files = any(f in real_files for f in expected_files)
            if not (len(real_files) > 0 and has_derrian_files):
                return False
        except:
            return False

        # Check 2: Do the actual training scripts exist?
        possible_sd_locations = [
            os.path.join(self.trainer_dir, "sd_scripts"),
            os.path.join(self.trainer_dir, "sd-scripts"),
            self.trainer_dir
        ]
        sd_scripts_found = False
        for location in possible_sd_locations:
            if os.path.exists(location):
                try:
                    train_scripts = [f for f in os.listdir(location) if 'train_network.py' in f]
                    if train_scripts:
                        sd_scripts_found = True
                        break
                except:
                    continue # Ignore errors from listing non-directories
        
        return sd_scripts_found

    def setup_environment(self):
        """Clean submodule-based setup: Official Kohya + LyCORIS + Derrian's optimizers"""
        print("🏗️ Setting up training environment with clean submodule architecture...")
        
        # Check for system dependencies
        if not self._check_and_install_packages():
            return False
            
        # Setup all submodules
        if not self._setup_all_submodules():
            print("❌ Failed to setup required submodules")
            return False
            
        # Setup custom optimizers from Derrian's backend
        if not self._setup_custom_optimizers():
            print("⚠️ Custom optimizers not available, continuing with standard optimizers")
            
        # Install requirements for SD scripts
        self._install_sd_scripts_requirements()
        
        # Final verification
        print("\n🔍 Verifying installation...")
        self._verify_installation()
        
        print("🎉 Clean submodule setup complete!")
        return True
        
    def _install_sd_scripts_requirements(self):
        """Install requirements for SD scripts and additional optimizers"""
        requirements_file = os.path.join(self.sd_scripts_dir, "requirements.txt")
        
        if os.path.exists(requirements_file):
            print("📦 Installing SD scripts requirements...")
            try:
                # First try the normal install
                subprocess.run(['pip', 'install', '-r', requirements_file], check=True)
                print("✅ SD scripts requirements installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Requirements install failed: {e}")
                print("🔧 Attempting to install requirements individually (skipping problematic ones)...")
                
                # Try to install line by line, skipping problematic ones
                try:
                    with open(requirements_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('file://'):
                            try:
                                subprocess.run(['pip', 'install', line], check=True, capture_output=True)
                                print(f"  ✅ {line}")
                            except subprocess.CalledProcessError:
                                print(f"  ⚠️ Skipped: {line} (problematic)")
                    
                    print("✅ Requirements installed (with some skipped)")
                except Exception as read_error:
                    print(f"⚠️ Could not process requirements file: {read_error}")
        
        # Install additional optimizers and dependencies that might be missing
        print("📦 Installing additional optimizers and dependencies...")
        additional_packages = [
            "pytorch_optimizer>=3.1.0",  # For CAME, Prodigy, etc.
            "schedulefree",  # For schedule-free optimizers
            "prodigy-plus-schedule-free",  # For Prodigy Plus
            "onnx>=1.14.0",  # For ONNX model format support
            "onnxruntime>=1.17.0"  # For ONNX runtime (required for WD14 taggers)
        ]
        
        for package in additional_packages:
            try:
                subprocess.run(['pip', 'install', package], check=True)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {package}: {e}")
        
        if not os.path.exists(requirements_file):
            print("⚠️ SD scripts requirements.txt not found")
            
    def _verify_installation(self):
        """Verify all components are working"""
        
        # Check SD scripts
        train_script = os.path.join(self.sd_scripts_dir, "train_network.py")
        if os.path.exists(train_script):
            print("   ✅ Kohya SD training scripts")
        else:
            print("   ❌ Kohya SD training scripts missing")
            
        # Check LyCORIS
        try:
            import sys
            sys.path.insert(0, self.lycoris_dir)
            import lycoris
            print("   ✅ LyCORIS (DoRA, LoHa, LoKr, etc.)")
        except ImportError as e:
            print(f"   ❌ LyCORIS import failed: {e}")
            
        # Check pytorch_optimizer
        try:
            import pytorch_optimizer
            print("   ✅ PyTorch Optimizer (CAME, Prodigy, etc.)")
        except ImportError as e:
            print(f"   ❌ PyTorch Optimizer: {e}")
            
        # Check custom optimizers
        try:
            import LoraEasyCustomOptimizer.came
            import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
            print("   ✅ Derrian's custom optimizers (CAME, REX)")
        except ImportError as e:
            print(f"   ❌ Custom optimizers: {e}")
            
        # Check Derrian's utility modules
        derrian_utils_dir = os.path.join(self.derrian_dir, "utils")
        if os.path.exists(derrian_utils_dir):
            try:
                # Add derrian_backend to path for utility imports
                # NOTE: Using os.path.join for cross-platform compatibility (Windows/Mac/Linux)
                if self.derrian_dir not in sys.path:
                    sys.path.insert(0, self.derrian_dir)
                
                from utils import validation, process, resize_lora
                print("   ✅ Derrian's utilities (validation, process, resize_lora)")
            except ImportError as e:
                print(f"   ❌ Derrian's utilities: {e}")
        else:
            print("   ❌ Derrian's utilities directory missing")
            
        print("   🏁 Setup verification complete!")

class ModelManager:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.pretrained_model_dir = os.path.join(self.project_root, "pretrained_model")
        self.vae_dir = os.path.join(self.project_root, "vae")
        os.makedirs(self.pretrained_model_dir, exist_ok=True)
        os.makedirs(self.vae_dir, exist_ok=True)

    def _validate_url(self, url):
        if re.search(r"https:\/\/huggingface\.co\/.*(?:resolve|blob).*", url):
            return url.replace("blob", "resolve")
        elif m := re.search(r"https:\/\/civitai\.com\/models\/(\d+)", url):
            if model_version_id := re.search(r"modelVersionId=(\d+)", url):
                return f"https://civitai.com/api/download/models/{model_version_id.group(1)}"
        return url # Return original url if no match

    def download_file(self, url, dest_dir, api_token=""):
        validated_url = self._validate_url(url)
        if not validated_url:
            print("Invalid URL provided.")
            return None

        header = ""
        if "civitai.com" in validated_url and api_token and not "hf" in api_token:
            validated_url = f"{validated_url}?token={api_token}"
        elif "huggingface.co" in validated_url and api_token:
            header = f"Authorization: Bearer {api_token}"

        filename = os.path.basename(validated_url.split('?')[0])
        destination_path = os.path.join(dest_dir, filename)

        print(f"Downloading from {validated_url}...")
        command = [
            "aria2c", validated_url,
            "--console-log-level=warn",
            "-c", "-s", "16", "-x", "16", "-k", "10M",
            "-d", dest_dir,
            "-o", filename
        ]
        if header:
            command.extend(["--header", header])

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

            print(f"\nDownload complete: {destination_path}")
            return destination_path

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while downloading the file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

    def download_model(self, model_url, model_name=None, api_token=""):
        """Download a model to the pretrained_model directory"""
        filename = model_name if model_name else os.path.basename(model_url.split('?')[0])
        if not filename.endswith(('.safetensors', '.ckpt')):
            filename += '.safetensors'
        
        return self.download_file(model_url, self.pretrained_model_dir, api_token)
    
    def download_vae(self, vae_url, vae_name=None, api_token=""):
        """Download a VAE to the vae directory"""
        filename = vae_name if vae_name else os.path.basename(vae_url.split('?')[0])
        if not filename.endswith(('.safetensors', '.ckpt')):
            filename += '.safetensors'
            
        return self.download_file(vae_url, self.vae_dir, api_token)


