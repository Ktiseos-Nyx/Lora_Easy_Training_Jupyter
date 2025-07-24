# core/managers.py
import subprocess
import os
import re
import zipfile
import toml
import shutil
from huggingface_hub import HfApi, login

class SetupManager:
    def __init__(self):
        # Use current working directory instead of hardcoded paths
        self.project_root = os.getcwd()  # This will be wherever the notebook is running
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.repo_url = "https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend.git"

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
        """Downloads Derrian's backend and SD scripts, sets up training environment."""
        if self._is_environment_setup():
            print("✅ Environment already set up. Skipping installation.")
            return True

        print("🔧 Environment not fully set up. Starting installation process...")
        if not self._check_and_install_packages():
            return False

        # Clone Derrian's backend repository (for custom optimizers)
        # The original is_valid_backend() helper is now part of _is_environment_setup()
        # So we directly proceed with cloning if not already valid.
        if os.path.exists(self.trainer_dir):
            print(f"🔄 Trainer directory exists but appears empty/invalid - re-downloading...")
            import shutil
            shutil.rmtree(self.trainer_dir)
        else:
            print(f"📥 Downloading Derrian's backend from {self.repo_url}...")
            
        try:
            subprocess.run(["git", "clone", self.repo_url, self.trainer_dir], check=True)
            print("✅ Derrian's backend downloaded successfully.")
            print("ℹ️ Submodules will be handled by Derrian's installer")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading Derrian's backend: {e}")
            return False # Changed from 'return' to 'return False' for explicit failure

        # Install Derrian's custom optimizers from custom_scheduler directory
        custom_scheduler_dir = os.path.join(self.trainer_dir, "custom_scheduler")
        
        # Check for setup.py in multiple possible locations
        possible_setup_locations = [
            os.path.join(custom_scheduler_dir, "setup.py"),  # Direct in custom_scheduler
            os.path.join(custom_scheduler_dir, "LoraEasyCustomOptimizer", "setup.py"),  # In subdirectory
        ]
        
        print(f"🔍 Checking for custom scheduler at: {custom_scheduler_dir}")
        
        setup_py = None
        install_dir = None
        
        for location in possible_setup_locations:
            if os.path.exists(location):
                setup_py = location
                install_dir = os.path.dirname(location)
                print(f"✅ Found setup.py at: {location}")
                break
        
        if setup_py:
            print("📦 Installing Derrian's custom optimizers (CAME, REX)...")
            try:
                # Try pip install in editable mode first (more reliable)
                subprocess.run(["pip", "install", "-e", "."], check=True, cwd=custom_scheduler_dir)
                print("✅ Custom optimizers installed successfully via pip!")
                
                # Verify installation
                try:
                    import LoraEasyCustomOptimizer.came
                    import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
                    print("✅ Custom optimizers verified - CAME and REX ready!")
                except ImportError as verify_e:
                    print(f"⚠️ Installation succeeded but import failed: {verify_e}")
                    print("🔧 This might be a Python path issue...")
                    
            except subprocess.CalledProcessError as e:
                print(f"⚠️ pip install failed, trying setup.py: {e}")
                try:
                    subprocess.run(["python", "setup.py", "install"], check=True, cwd=custom_scheduler_dir)
                    print("✅ Custom optimizers installed via setup.py!")
                except subprocess.CalledProcessError as e2:
                    print(f"⚠️ Custom optimizer installation failed: {e2}")
                    print("🔧 Continuing without custom optimizers...")
        else:
            print("ℹ️ No setup.py found in custom_scheduler - checking directory contents...")
            if os.path.exists(custom_scheduler_dir):
                try:
                    contents = os.listdir(custom_scheduler_dir)
                    print(f"📁 custom_scheduler contents: {contents}")
                except Exception as e:
                    print(f"❌ Could not list custom_scheduler directory: {e}")
            else:
                print("❌ custom_scheduler directory does not exist")

        # Check the actual structure of Derrian's backend
        print("🔍 Analyzing Derrian's backend structure...")
        if os.path.exists(self.trainer_dir):
            try:
                contents = os.listdir(self.trainer_dir)
                print(f"📁 Trainer directory contents: {contents}")
                
                # Look for SD scripts in various locations
                possible_sd_locations = [
                    os.path.join(self.trainer_dir, "sd_scripts"),
                    os.path.join(self.trainer_dir, "sd-scripts"), 
                    os.path.join(self.trainer_dir, "scripts"),
                    self.trainer_dir  # Scripts might be in root
                ]
                
                sd_scripts_found = False
                for location in possible_sd_locations:
                    if os.path.exists(location):
                        # Check for training scripts
                        train_scripts = [f for f in os.listdir(location) if f.endswith('train_network.py')]
                        if train_scripts:
                            print(f"✅ Training scripts found at: {location}")
                            print(f"   Scripts: {train_scripts}")
                            sd_scripts_found = True
                            break
                
                if not sd_scripts_found:
                    print("ℹ️ No training scripts found - training manager will handle download when needed")
                    
            except Exception as e:
                print(f"⚠️ Error analyzing directory: {e}")
        else:
            print("❌ Trainer directory not found")

        # Check for and run SD scripts installation
        print("🔧 Setting up SD scripts...")
        
        # Detect Python version and choose the right Derrian installer
        python_version = self._get_python_version()
        print(f"🐍 Detected Python version: {python_version}")
        
        # Choose installer based on Python version
        version_to_installer = {
            "3.10": "install_310.sh",
            "3.11": "install_311.sh", 
            "3.12": "install_312.sh"
        }
        
        preferred_installer = version_to_installer.get(python_version, "install_312.sh")  # Default to 3.12
        
        # Look for installers in order of preference
        install_scripts = [preferred_installer] + ["install_312.sh", "install_311.sh", "install_310.sh", "install.sh", "installer.py"]
        # Remove duplicates while preserving order
        install_scripts = list(dict.fromkeys(install_scripts))
        
        found_installer = None
        for script in install_scripts:
            script_path = os.path.join(self.trainer_dir, script)
            if os.path.exists(script_path):
                found_installer = script_path
                if script == preferred_installer:
                    print(f"✅ Found matching Derrian's installer for Python {python_version}: {script}")
                else:
                    print(f"✅ Found Derrian's installer: {script} (fallback)")
                break
        
        if found_installer and found_installer.endswith('.sh'):
            print(f"🔧 Running Derrian's installer: {found_installer}...")
            try:
                subprocess.run(['chmod', '+x', found_installer], check=True)
                # The .sh script calls installer.py internally, but we need to ensure non-interactive mode
                # Let's run the installer.py directly with 'local' argument for non-interactive install
                print("🔧 Running non-interactive installation...")
                subprocess.run(['python', 'installer.py', 'local'], check=True, cwd=self.trainer_dir)
                print("✅ Derrian's backend installation completed!")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Derrian's installer failed: {e}")
                print("🔧 Continuing with manual requirements installation...")
        elif found_installer and found_installer.endswith('.py'):
            print(f"🔧 Running Python installer: {found_installer}...")
            try:
                # Use 'local' argument for non-interactive installation
                subprocess.run(['python', found_installer, 'local'], check=True, cwd=self.trainer_dir)
                print("✅ Derrian's backend installation completed!")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Python installer failed: {e}")
                print("🔧 Continuing with manual requirements installation...")
        
        # Install critical missing packages
        print("📦 Installing critical packages (bitsandbytes, triton, onnx)...")
        # Determine the correct venv python path
        venv_python = os.path.join(self.trainer_dir, "sd_scripts", "venv", "bin", "python")
        if not os.path.exists(venv_python):
            # Fallback if venv not found (e.g., if Derrian's installer didn't create it as expected)
            print("⚠️ Virtual environment Python not found. Falling back to system Python for package install.")
            venv_python = "python"

        bnb_wheel_url = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"
        try:
            print(f"Attempting to install bitsandbytes from: {bnb_wheel_url}")
            subprocess.run([venv_python, "-m", "pip", "install", "--force-reinstall", bnb_wheel_url], check=True)
            print("✅ bitsandbytes installed successfully from wheel.")
            
            # Also try to install triton and onnx, as they're common dependency issues
            print("Attempting to install triton...")
            subprocess.run([venv_python, "-m", "pip", "install", "triton"], check=True)
            print("✅ Triton installed successfully.")
            
            print("Attempting to install onnx and onnxruntime...")
            subprocess.run([venv_python, "-m", "pip", "install", "onnx", "onnxruntime"], check=True)
            print("✅ ONNX packages installed successfully.")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install packages from wheel: {e}")
            print("🔧 Attempting generic pip install for all packages...")
            try:
                subprocess.run([venv_python, "-m", "pip", "install", "bitsandbytes", "triton", "onnx", "onnxruntime"], check=True)
                print("✅ All packages installed successfully via generic pip.")
            except subprocess.CalledProcessError as e2:
                print(f"❌ Failed to install some packages even with generic pip: {e2}")
                print("💡 You may need to manually install missing packages for your specific setup.")
        
        # Install requirements with smart filtering
        requirements_file = os.path.join(self.trainer_dir, "requirements.txt")
        if os.path.exists(requirements_file):
            print("📦 Installing SD scripts requirements (filtering local packages)...")
            
            # Create a filtered requirements file
            filtered_requirements = os.path.join(self.trainer_dir, "requirements_filtered.txt")
            
            try:
                with open(requirements_file, 'r') as f:
                    lines = f.readlines()
                
                # Filter out local package references and problematic lines
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and 
                        not line.startswith('#') and 
                        not line.startswith('-e .') and
                        not line.startswith('file:///') and
                        not 'workspace' in line.lower()):
                        filtered_lines.append(line)
                
                # Write filtered requirements
                with open(filtered_requirements, 'w') as f:
                    f.write('\n'.join(filtered_lines))
                
                # Install filtered requirements
                subprocess.run(["pip", "install", "-r", filtered_requirements], check=True)
                print("✅ Requirements installed successfully")
                
                # Clean up temp file
                os.remove(filtered_requirements)
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Some packages failed to install: {e}")
                print("🔧 Most packages probably installed successfully!")
            except Exception as e:
                print(f"⚠️ Requirements processing failed: {e}")
                print("🔧 Continuing anyway - VastAI likely has most packages already")

        # Final status report - be honest about what we actually have
        print("🎉 Setup process complete! Status report:")
        
        # Check what we actually have
        custom_scheduler_dir = os.path.join(self.trainer_dir, "custom_scheduler")
        sd_scripts_dir = os.path.join(self.trainer_dir, "sd_scripts")
        
        # Test custom optimizers actually work
        try:
            import LoraEasyCustomOptimizer.came
            import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
            print("   ✅ Derrian's custom optimizers (CAME, REX) - VERIFIED WORKING")
        except ImportError as e:
            print(f"   ❌ Custom optimizers import failed: {e}")
            if os.path.exists(custom_scheduler_dir):
                print("   📁 custom_scheduler directory exists but optimizers not importable")
                print("   💡 This might be a Python path or installation issue")
            else:
                print("   📁 custom_scheduler directory missing entirely")
            
        if os.path.exists(sd_scripts_dir):
            # Check for actual training scripts
            train_scripts = [f for f in os.listdir(sd_scripts_dir) if f.endswith('train_network.py')]
            if train_scripts:
                print("   ✅ SD scripts (training engine)")
            else:
                print("   ⚠️ SD scripts directory exists but no training scripts found")
        else:
            print("   ❌ SD scripts not found - will be downloaded by training manager")
            
        # Check if we have a functional backend
        if os.path.exists(self.trainer_dir) and len([f for f in os.listdir(self.trainer_dir) if f != '.ipynb_checkpoints']) > 0:
            print("   ✅ Derrian's backend repository")
        else:
            print("   ❌ Derrian's backend is empty or missing")
            
        print("🚀 Training manager will download missing components when needed!")
        return True

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


