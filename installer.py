#!/usr/bin/env python3
"""
LoRA Easy Training - Unified Command-Line Installer
"""

import os
import sys
import subprocess
import shutil
import platform

def get_python_command():
    """Detects the best available Python command."""
    for cmd in ['python3', 'python']:
        if shutil.which(cmd):
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0 and 'Python 3' in result.stdout:
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    raise RuntimeError("❌ Python 3 not found. Please install Python 3.8+")

class UnifiedInstaller:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.python_cmd = get_python_command()
        self.pip_cmd = [self.python_cmd, '-m', 'pip']
        
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")
        
        self.submodules = {
            'derrian_backend': {
                'url': 'https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend.git',
                'path': self.derrian_dir,
                'description': "Derrian's backend with Kohya scripts, LyCORIS, and custom optimizers"
            }
        }

    def print_banner(self):
        print("=" * 70)
        print("🚀 LoRA Easy Training - Unified Command-Line Installer")
        print("=" * 70)
        print(f"🐍 Using Python: {self.python_cmd}")
        print(f"🔧 Project Root: {self.project_root}")
        print()

    def run_command(self, command, description, cwd=None):
        print(f"🔄 {description}...")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
            print(f"✅ {description} successful.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed.")
            print(f"   Error: {e}")
            print(f"   Stderr: {e.stderr}")
            print(f"   Stdout: {e.stdout}")
            return False

    def setup_submodules(self):
        print("🔗 Cloning and initializing all submodules...")
        os.makedirs(self.trainer_dir, exist_ok=True)
        
        for name, config in self.submodules.items():
            path = config['path']
            url = config['url']
            
            if os.path.exists(path) and os.listdir(path):
                print(f"   - Submodule '{name}' already exists. Skipping clone.")
                continue

            if not self.run_command(['git', 'clone', url, path], f"Cloning {name}"):
                return False

        # Initialize nested submodules (sd_scripts, lycoris)
        if not self.run_command(['git', 'submodule', 'update', '--init', '--recursive'], "Initializing nested submodules", cwd=self.derrian_dir):
            return False
            
        return True

    def install_dependencies(self):
        requirements_file = os.path.join(self.project_root, "requirements-backend.txt")
        if not os.path.exists(requirements_file):
            print(f"❌ CRITICAL: requirements-backend.txt not found!")
            return False
            
        return self.run_command(self.pip_cmd + ['install', '-r', requirements_file], "Installing Python packages")

    def apply_special_fixes_and_installs(self):
        print("🔧 Applying special fixes and performing editable installs...")
        
        # --- Editable Installs ---
        editable_installs = {
            "LyCORIS": self.lycoris_dir,
            "Custom Optimizers": os.path.join(self.derrian_dir, "custom_scheduler"),
            "Kohya's SD Scripts": self.sd_scripts_dir
        }
        
        for name, path in editable_installs.items():
            if os.path.exists(os.path.join(path, 'setup.py')):
                if not self.run_command(self.pip_cmd + ['install', '-e', '.'], f"Editable install for {name}", cwd=path):
                    print(f"   - ⚠️  Could not install {name} in editable mode. Training might still work.")
        
        # --- Platform-Specific Fixes ---
        if platform.system() == "Windows":
            print("   - Applying Windows-specific fix for bitsandbytes...")
            try:
                bnb_src_dir = os.path.join(self.sd_scripts_dir, 'bitsandbytes_windows')
                result = subprocess.run([self.python_cmd, '-c', 'import site; print(site.getsitepackages()[0])'], capture_output=True, text=True, check=True)
                site_packages = result.stdout.strip()
                bnb_dest_dir = os.path.join(site_packages, 'bitsandbytes')
                
                if os.path.exists(bnb_dest_dir):
                    # This can be improved with logic to select the correct CUDA version DLL
                    dll_to_copy = 'libbitsandbytes_cuda118.dll' 
                    src_file = os.path.join(bnb_src_dir, dll_to_copy)
                    dest_file = os.path.join(bnb_dest_dir, 'libbitsandbytes_cudaall.dll')
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, dest_file)
                        print(f"     ✅ Copied {dll_to_copy} to {dest_file}")
                    else:
                        print(f"     ⚠️ Could not find source DLL: {src_file}")
                else:
                    print(f"     ⚠️ bitsandbytes directory not found in site-packages. Cannot apply fix.")
            except Exception as e:
                print(f"     ❌ Error applying bitsandbytes fix: {e}")

        # --- PyTorch version file fix ---
        print("   - Checking if PyTorch version patch is needed...")
        try:
            import torch
            if torch.__version__ in ["2.0.0", "2.0.1"]:
                print(f"   - Applying patch for PyTorch {torch.__version__}...")
                fix_script_path = os.path.join(self.derrian_dir, 'fix_torch.py')
                if os.path.exists(fix_script_path):
                    self.run_command([self.python_cmd, fix_script_path], "Applying PyTorch patch")
            else:
                print(f"   - PyTorch version is {torch.__version__}. No patch needed.")
        except ImportError:
            print("   - ⚠️  Could not import PyTorch. Skipping version patch check.")
        except Exception as e:
            print(f"   - ❌ Error applying PyTorch patch: {e}")
        
        return True

    def run_installation(self):
        self.print_banner()
        
        if not self.setup_submodules():
            print("❌ Halting installation due to submodule setup failure.")
            return

        if not self.install_dependencies():
            print("❌ Halting installation due to dependency installation failure.")
            return
            
        if not self.apply_special_fixes_and_installs():
            print("⚠️ Some special fixes or editable installs failed.")

        print("\n" + "=" * 70)
        print("✅ Installation complete!")
        print("   You can now start Jupyter and use the training notebooks.")
        print("   Run: jupyter notebook")
        print("=" * 70)

if __name__ == "__main__":
    installer = UnifiedInstaller()
    installer.run_installation()