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

def get_subprocess_environment(project_root=None):
    """
    Create standardized environment for subprocess calls with proper PYTHONPATH and CUDA setup.
    This prevents the CAME optimizer import errors and other module not found issues.
    """
    if project_root is None:
        project_root = os.getcwd()
    
    # Start with current environment
    env = os.environ.copy()
    
    # Memory optimization
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Setup PYTHONPATH for custom optimizers (CAME, etc.)
    derrian_dir = os.path.join(project_root, "trainer", "derrian_backend")
    custom_scheduler_dir = os.path.join(derrian_dir, "custom_scheduler")
    
    if os.path.exists(custom_scheduler_dir):
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{custom_scheduler_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = custom_scheduler_dir
    
    # Setup CUDA/CuDNN library paths for Linux
    if sys.platform.startswith('linux'):
        cuda_path = env.get("CUDA_PATH", "/usr/local/cuda")
        new_ld_library_path = f"{cuda_path}/lib64:{cuda_path}/extras/CUPTI/lib64"
        
        # Add common CuDNN paths
        cudnn_paths = [
            f"{cuda_path}/lib",
            f"{cuda_path}/targets/x86_64-linux/lib", 
            "/usr/lib/x86_64-linux-gnu",
        ]
        for p in cudnn_paths:
            if os.path.exists(p):
                new_ld_library_path += f":{p}"

        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{new_ld_library_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = new_ld_library_path
    
    # ROCm support for AMD GPUs
    if not env.get('HSA_OVERRIDE_GFX_VERSION'):
        env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    
    return env

class SetupManager:
    def __init__(self):
        # Light initialization - just paths, no detection spam!
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        self.sd_scripts_dir = os.path.join(self.derrian_dir, "sd_scripts")
        self.lycoris_dir = os.path.join(self.derrian_dir, "lycoris")
        
        # Lazy-loaded properties
        self._environment_detected = False
        self._is_vastai = None
        self._correct_venv_path = None
        
        # Submodule URLs - Only need derrian_backend which includes sd_scripts and lycoris
        self.submodules = {
            'derrian_backend': {
                'url': 'https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend.git',
                'path': self.derrian_dir,
                'description': 'Derrian\'s backend with Kohya scripts, LyCORIS, and custom optimizers'
            }
        }
        
    def _detect_environment(self):
        """Detect environment only when actually needed - not on every widget load!"""
        if self._environment_detected:
            return
        
        # Check for VastAI first
        self._is_vastai = bool(os.environ.get('VAST_CONTAINERLABEL') or '/workspace' in self.project_root)
        if self._is_vastai:
            self._correct_venv_path = '/venv/main/bin/pip'
            print(f"🐳 VastAI detected - using correct venv: {self._correct_venv_path}")
        else:
            # Detect current Python/pip environment (conda, venv, etc.)
            self._correct_venv_path = self._detect_current_pip()
            
        self._environment_detected = True
    
    def _detect_current_pip(self):
        """Detect the pip executable for the current Python environment"""
        import sys
        import os
        
        # Get current Python executable
        current_python = sys.executable
        
        # Determine corresponding pip and environment type
        if 'conda' in current_python or 'miniconda' in current_python or 'anaconda' in current_python:
            # Conda environment - use python -m pip to ensure same env
            pip_cmd = [current_python, '-m', 'pip']
            env_type = "conda"
            env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        elif 'venv' in current_python or '.venv' in current_python:
            # Virtual environment
            pip_cmd = [current_python, '-m', 'pip'] 
            env_type = "venv"
            env_name = os.path.basename(os.path.dirname(os.path.dirname(current_python)))
        else:
            # System Python (fallback)
            pip_cmd = ['pip']
            env_type = "system"
            env_name = "system"
        
        print(f"🐍 Environment detected: {env_type} ({env_name})")
        print(f"🔧 Using Python: {current_python}")
        
        # Return pip command as list for subprocess
        return pip_cmd
    
    def validate_and_fix_deep_dependencies(self):
        """Comprehensive dependency validation and auto-fixing"""
        print("🔍 Running comprehensive dependency validation...")
        print("=" * 60)
        
        all_good = True
        
        # 1. CUDA/cuDNN Compatibility Check
        print("🎮 Checking CUDA/cuDNN compatibility...")
        cuda_status = self._check_cuda_cudnn_versions()
        if not cuda_status['compatible']:
            print(f"⚠️ CUDA issue detected: {cuda_status['issue']}")
            if self._fix_cuda_cudnn_mismatch(cuda_status):
                print("✅ CUDA issue fixed!")
            else:
                print("❌ CUDA issue persists - manual intervention needed")
                all_good = False
        else:
            print("✅ CUDA/cuDNN compatibility looks good")
        
        # 2. PyTorch CUDA Compatibility  
        print("\n🔥 Checking PyTorch CUDA compatibility...")
        pytorch_status = self._check_pytorch_cuda_compatibility()
        if not pytorch_status['compatible']:
            print(f"⚠️ PyTorch issue: {pytorch_status['issue']}")
            if self._reinstall_correct_pytorch(pytorch_status):
                print("✅ PyTorch issue fixed!")
            else:
                print("❌ PyTorch issue persists")
                all_good = False
        else:
            print("✅ PyTorch CUDA compatibility verified")
        
        # 3. Derrian Backend Dependencies
        print("\n🚀 Checking Derrian backend dependencies...")
        backend_status = self._check_derrian_backend_deps()
        if not backend_status['complete']:
            print(f"⚠️ Backend issue: {backend_status['issue']}")
            if self._fix_derrian_backend_install(backend_status):
                print("✅ Backend dependencies fixed!")
            else:
                print("❌ Backend issue persists")
                all_good = False
        else:
            print("✅ Derrian backend dependencies complete")
        
        print("\n" + "=" * 60)
        if all_good:
            print("🎉 All dependencies validated and ready for training!")
        else:
            print("⚠️ Some issues remain - check logs above")
            
        return all_good
    
    def _detect_gpu_type(self):
        """Detect GPU type and acceleration method"""
        gpu_info = {
            'type': 'unknown',
            'devices': [],
            'acceleration': None,
            'rocm_available': False,
            'zluda_available': False,
            'directml_available': False
        }
        
        try:
            import torch
            
            # Check for AMD GPU via ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                gpu_info['type'] = 'amd'
                gpu_info['acceleration'] = 'rocm'
                gpu_info['rocm_available'] = True
                if torch.cuda.is_available():  # ROCm uses cuda interface
                    gpu_info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                return gpu_info
            
            # Check for NVIDIA GPU
            if torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0).lower()
                    if 'nvidia' in device_name or 'geforce' in device_name or 'rtx' in device_name or 'gtx' in device_name:
                        gpu_info['type'] = 'nvidia'
                        gpu_info['acceleration'] = 'cuda'
                        gpu_info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        return gpu_info
                    elif 'amd' in device_name or 'radeon' in device_name:
                        # AMD GPU detected via CUDA interface (ZLUDA likely)
                        gpu_info['type'] = 'amd'
                        gpu_info['acceleration'] = 'zluda'
                        gpu_info['zluda_available'] = True
                        gpu_info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        return gpu_info
                except:
                    pass
            
            # Check for DirectML (Windows AMD fallback)
            if sys.platform == "win32":
                try:
                    import torch_directml
                    gpu_info['acceleration'] = 'directml'
                    gpu_info['directml_available'] = True
                    if torch_directml.is_available():
                        gpu_info['type'] = 'amd'  # Assume AMD for DirectML
                        gpu_info['devices'] = ['DirectML Device']
                        return gpu_info
                except ImportError:
                    pass
            
            # Check system for AMD GPUs via other methods
            gpu_info.update(self._detect_amd_gpu_system())
            
        except ImportError:
            gpu_info['issue'] = 'PyTorch not installed'
        
        return gpu_info
    
    def _detect_amd_gpu_system(self):
        """Detect AMD GPU via system calls"""
        amd_info = {}
        
        if sys.platform == "win32":
            # Windows: Check via wmic
            try:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    amd_info['type'] = 'amd'
                    amd_info['devices'] = ['AMD GPU (detected via system)']
            except:
                pass
        else:
            # Linux: Check via lspci
            try:
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=5)
                if 'AMD' in result.stdout or 'ATI' in result.stdout:
                    amd_info['type'] = 'amd'
                    # Extract GPU names
                    lines = result.stdout.split('\n')
                    gpu_lines = [line for line in lines if ('AMD' in line or 'ATI' in line) and ('VGA' in line or 'Display' in line)]
                    amd_info['devices'] = gpu_lines[:3]  # Limit output
            except:
                pass
        
        return amd_info
    
    def _check_cuda_cudnn_versions(self):
        """Check CUDA and cuDNN version compatibility (NVIDIA) or ROCm compatibility (AMD)"""
        gpu_info = self._detect_gpu_type()
        
        if gpu_info['type'] == 'amd':
            return self._check_amd_compatibility(gpu_info)
        elif gpu_info['type'] == 'nvidia':
            return self._check_nvidia_cuda_compatibility()
        else:
            return {
                'compatible': False,
                'issue': 'No supported GPU detected',
                'gpu_type': 'unknown',
                'fix_type': 'install_gpu_support'
            }
    
    def _check_nvidia_cuda_compatibility(self):
        """Check NVIDIA CUDA compatibility"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'CUDA not available in PyTorch',
                    'cuda_version': None,
                    'cudnn_version': None,
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda'
                }
            
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            
            # Check for known problematic combinations
            if cudnn_version and str(cudnn_version).startswith('9'):
                return {
                    'compatible': False,
                    'issue': f'cuDNN {cudnn_version} incompatible (need 8.x)',
                    'cuda_version': cuda_version,
                    'cudnn_version': cudnn_version,
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda8'
                }
            
            return {
                'compatible': True,
                'cuda_version': cuda_version,
                'cudnn_version': cudnn_version,
                'gpu_type': 'nvidia'
            }
            
        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed',
                'gpu_type': 'nvidia',
                'fix_type': 'install_pytorch_cuda'
            }
    
    def _check_amd_compatibility(self, gpu_info):
        """Check AMD GPU compatibility with available acceleration methods"""
        acceleration = gpu_info.get('acceleration')
        
        if acceleration == 'rocm':
            return self._check_rocm_compatibility(gpu_info)
        elif acceleration == 'zluda':
            return self._check_zluda_compatibility(gpu_info)
        elif acceleration == 'directml':
            return self._check_directml_compatibility(gpu_info)
        else:
            # No acceleration detected, suggest options
            return {
                'compatible': False,
                'issue': 'AMD GPU detected but no acceleration method available',
                'gpu_type': 'amd',
                'devices': gpu_info.get('devices', []),
                'fix_type': 'install_amd_support',
                'suggestions': self._get_amd_support_suggestions()
            }
    
    def _check_rocm_compatibility(self, gpu_info):
        """Check ROCm compatibility"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ROCm PyTorch installed but GPU not accessible',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'fix_rocm_setup'
                }
            
            # Test ROCm functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                hip_version = torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'
                
                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'hip_version': hip_version,
                    'devices': gpu_info.get('devices', [])
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ROCm test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'fix_rocm_setup'
                }
                
        except ImportError:
            return {
                'compatible': False,
                'issue': 'ROCm PyTorch not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_rocm_pytorch'
            }
    
    def _check_zluda_compatibility(self, gpu_info):
        """Check ZLUDA compatibility"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ZLUDA detected but CUDA interface not working',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }
            
            # Test ZLUDA functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                
                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'cuda_version': torch.version.cuda,
                    'devices': gpu_info.get('devices', []),
                    'note': 'ZLUDA experimental - performance may vary'
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ZLUDA test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }
                
        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed for ZLUDA',
                'gpu_type': 'amd',
                'fix_type': 'install_pytorch_zluda'
            }
    
    def _check_directml_compatibility(self, gpu_info):
        """Check DirectML compatibility"""
        try:
            import torch_directml
            
            if not torch_directml.is_available():
                return {
                    'compatible': False,
                    'issue': 'DirectML installed but not available',
                    'gpu_type': 'amd',
                    'acceleration': 'directml',
                    'fix_type': 'fix_directml_setup'
                }
            
            return {
                'compatible': True,
                'gpu_type': 'amd',
                'acceleration': 'directml',
                'devices': gpu_info.get('devices', []),
                'note': 'DirectML may have limited LoRA training support'
            }
            
        except ImportError:
            return {
                'compatible': False,
                'issue': 'DirectML not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_directml'
            }
    
    def _get_amd_support_suggestions(self):
        """Get platform-specific AMD support suggestions"""
        suggestions = []
        
        if sys.platform.startswith('linux'):
            suggestions.extend([
                'Install ROCm PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0',
                'Try ZLUDA: Download from https://github.com/vosen/ZLUDA',
                'Ensure ROCm drivers installed: AMD ROCm 6.1+'
            ])
        elif sys.platform == 'win32':
            suggestions.extend([
                'Try ZLUDA: Download from https://github.com/vosen/ZLUDA',
                'Install DirectML: pip install torch-directml',
                'Note: ROCm not supported on Windows'
            ])
        else:
            suggestions.append('AMD GPU support limited on this platform')
        
        return suggestions
    
    def _check_pytorch_cuda_compatibility(self):
        """Check if PyTorch installation matches GPU type (CUDA for NVIDIA, ROCm for AMD)"""
        gpu_info = self._detect_gpu_type()
        
        if gpu_info['type'] == 'amd':
            return self._check_amd_pytorch_compatibility(gpu_info)
        elif gpu_info['type'] == 'nvidia':
            return self._check_nvidia_pytorch_compatibility()
        else:
            return {
                'compatible': False,
                'issue': 'No supported GPU detected',
                'fix_type': 'install_gpu_support'
            }
    
    def _check_nvidia_pytorch_compatibility(self):
        """Check NVIDIA PyTorch compatibility"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'PyTorch CUDA not available',
                    'torch_cuda': None,
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda'
                }
            
            torch_cuda = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            # Basic sanity check
            if device_count == 0:
                return {
                    'compatible': False,
                    'issue': 'No CUDA devices detected',
                    'gpu_type': 'nvidia',
                    'fix_type': 'check_cuda_install'
                }
            
            # Test basic CUDA operations
            try:
                test_tensor = torch.randn(10, device='cuda')
                _ = test_tensor * 2.0
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'CUDA operation failed: {e}',
                    'gpu_type': 'nvidia',
                    'fix_type': 'reinstall_pytorch_cuda'
                }
            
            return {
                'compatible': True,
                'torch_cuda': torch_cuda,
                'device_count': device_count,
                'device_name': device_name,
                'gpu_type': 'nvidia'
            }
            
        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed',
                'gpu_type': 'nvidia',
                'fix_type': 'install_pytorch'
            }
    
    def _check_amd_pytorch_compatibility(self, gpu_info):
        """Check AMD PyTorch compatibility"""
        acceleration = gpu_info.get('acceleration')
        
        if acceleration == 'rocm':
            return self._check_amd_rocm_pytorch()
        elif acceleration == 'zluda':
            return self._check_amd_zluda_pytorch()
        elif acceleration == 'directml':
            return self._check_amd_directml_pytorch()
        else:
            return {
                'compatible': False,
                'issue': 'AMD GPU detected but no PyTorch acceleration available',
                'gpu_type': 'amd',
                'fix_type': 'install_amd_pytorch',
                'suggestions': self._get_amd_support_suggestions()
            }
    
    def _check_amd_rocm_pytorch(self):
        """Check AMD ROCm PyTorch compatibility"""
        try:
            import torch
            
            # Check for ROCm build
            if not hasattr(torch.version, 'hip') or torch.version.hip is None:
                return {
                    'compatible': False,
                    'issue': 'AMD GPU detected but PyTorch CUDA build installed (need ROCm build)',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'install_rocm_pytorch'
                }
            
            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ROCm PyTorch installed but GPU not accessible',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'hip_version': torch.version.hip,
                    'fix_type': 'fix_rocm_setup'
                }
            
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            # Test ROCm functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                _ = test_tensor * 2.0
                
                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'hip_version': torch.version.hip,
                    'device_count': device_count,
                    'device_name': device_name
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ROCm test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'rocm',
                    'fix_type': 'fix_rocm_setup'
                }
                
        except ImportError:
            return {
                'compatible': False,
                'issue': 'ROCm PyTorch not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_rocm_pytorch'
            }
    
    def _check_amd_zluda_pytorch(self):
        """Check AMD ZLUDA PyTorch compatibility"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'compatible': False,
                    'issue': 'ZLUDA setup detected but CUDA interface not working',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }
            
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            # Test ZLUDA functionality
            try:
                test_tensor = torch.randn(10, device='cuda')
                _ = test_tensor * 2.0
                
                return {
                    'compatible': True,
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'torch_cuda': torch.version.cuda,
                    'device_count': device_count,
                    'device_name': device_name,
                    'note': 'ZLUDA experimental - some features may not work'
                }
            except Exception as e:
                return {
                    'compatible': False,
                    'issue': f'ZLUDA test failed: {str(e)}',
                    'gpu_type': 'amd',
                    'acceleration': 'zluda',
                    'fix_type': 'fix_zluda_setup'
                }
                
        except ImportError:
            return {
                'compatible': False,
                'issue': 'PyTorch not installed for ZLUDA',
                'gpu_type': 'amd',
                'fix_type': 'install_pytorch_zluda'
            }
    
    def _check_amd_directml_pytorch(self):
        """Check AMD DirectML PyTorch compatibility"""
        try:
            import torch_directml
            
            if not torch_directml.is_available():
                return {
                    'compatible': False,
                    'issue': 'DirectML installed but not available',
                    'gpu_type': 'amd',
                    'acceleration': 'directml',
                    'fix_type': 'fix_directml_setup'
                }
            
            return {
                'compatible': True,
                'gpu_type': 'amd',
                'acceleration': 'directml',
                'device_count': 1,  # DirectML typically shows as single device
                'device_name': 'DirectML Device',
                'note': 'DirectML has limited LoRA training support'
            }
            
        except ImportError:
            return {
                'compatible': False,
                'issue': 'DirectML not installed',
                'gpu_type': 'amd',
                'fix_type': 'install_directml'
            }
    
    def _check_derrian_backend_deps(self):
        """Check if Derrian backend and CAME optimizers are properly installed"""
        issues = []
        
        # Check if sd_scripts directory exists
        if not os.path.exists(self.sd_scripts_dir):
            issues.append("sd_scripts directory missing")
        
        # Check for key training scripts
        key_scripts = ['train_network.py', 'sdxl_train_network.py']
        for script in key_scripts:
            if not os.path.exists(os.path.join(self.sd_scripts_dir, script)):
                issues.append(f"Missing training script: {script}")
        
        # Check for CAME optimizer
        came_available = False
        try:
            import LoraEasyCustomOptimizer.came
            came_available = True
        except ImportError:
            # Try alternative paths
            sys_path_backup = sys.path.copy()
            try:
                custom_scheduler_dir = os.path.join(self.derrian_dir, "custom_scheduler")
                if os.path.exists(custom_scheduler_dir):
                    sys.path.insert(0, custom_scheduler_dir)
                    import LoraEasyCustomOptimizer.came
                    came_available = True
            except ImportError:
                pass
            finally:
                sys.path = sys_path_backup
        
        if not came_available:
            issues.append("CAME optimizer not available")
        
        # Check for LyCORIS
        lycoris_available = False
        lycoris_dir = os.path.join(self.sd_scripts_dir, "networks")
        if os.path.exists(lycoris_dir):
            lycoris_files = ['lora.py', 'dylora.py']  # Basic check
            if any(os.path.exists(os.path.join(lycoris_dir, f)) for f in lycoris_files):
                lycoris_available = True
        
        if not lycoris_available:
            issues.append("LyCORIS networks not available")
        
        return {
            'complete': len(issues) == 0,
            'issues': issues,
            'issue': '; '.join(issues) if issues else None
        }
    
    def _fix_cuda_cudnn_mismatch(self, status):
        """Attempt to fix GPU compatibility issues (NVIDIA CUDA or AMD)"""
        print("🔧 Attempting to fix GPU compatibility issues...")
        
        fix_type = status.get('fix_type')
        gpu_type = status.get('gpu_type', 'nvidia')
        
        # NVIDIA GPU fixes
        if gpu_type == 'nvidia':
            if fix_type in ['reinstall_pytorch_cuda', 'reinstall_pytorch_cuda8']:
                return self._reinstall_pytorch_with_cuda8()
            elif fix_type == 'install_pytorch':
                return self._install_pytorch_with_cuda()
        
        # AMD GPU fixes
        elif gpu_type == 'amd':
            if fix_type == 'install_rocm_pytorch':
                return self._install_rocm_pytorch()
            elif fix_type == 'install_pytorch_zluda':
                return self._install_pytorch_zluda()
            elif fix_type == 'install_directml':
                return self._install_directml()
            elif fix_type == 'install_amd_support':
                return self._install_amd_support()
            elif fix_type == 'install_amd_pytorch':
                return self._install_amd_support()
            elif fix_type == 'fix_rocm_setup':
                return self._fix_rocm_setup()
            elif fix_type == 'fix_zluda_setup':
                return self._fix_zluda_setup()
            elif fix_type == 'fix_directml_setup':
                return self._fix_directml_setup()
        
        # Generic GPU support
        elif fix_type == 'install_gpu_support':
            print("❓ No supported GPU detected. Please check:")
            print("  • NVIDIA GPU: Install NVIDIA drivers + CUDA")
            print("  • AMD GPU (Linux): Install ROCm drivers")
            print("  • AMD GPU (Windows): Install AMD drivers")
            return False
        
        print(f"❌ Unknown fix type: {fix_type} for {gpu_type} GPU")
        return False
    
    def _reinstall_correct_pytorch(self, status):
        """Reinstall PyTorch with correct CUDA support"""
        return self._reinstall_pytorch_with_cuda8()
    
    def _reinstall_pytorch_with_cuda8(self):
        """Install PyTorch with cuDNN 8.x compatibility"""
        print("📦 Installing PyTorch with cuDNN 8.x compatibility...")
        
        try:
            # Uninstall existing PyTorch
            pip_cmd = self.correct_venv_path
            if isinstance(pip_cmd, list):
                uninstall_cmd = pip_cmd + ['uninstall', '-y', 'torch', 'torchvision', 'torchaudio']
            else:
                uninstall_cmd = [pip_cmd, 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio']
            
            subprocess.run(uninstall_cmd, check=True, capture_output=True)
            
            # Install PyTorch with CUDA 11.8 (compatible with cuDNN 8.x)
            if isinstance(pip_cmd, list):
                install_cmd = pip_cmd + [
                    'install', 'torch', 'torchvision', 'torchaudio', 
                    '--index-url', 'https://download.pytorch.org/whl/cu118'
                ]
            else:
                install_cmd = [pip_cmd, 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118']
            
            result = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            print("✅ PyTorch with CUDA 11.8 installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to reinstall PyTorch: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False
    
    def _install_pytorch_with_cuda(self):
        """Install PyTorch with CUDA from scratch"""
        return self._reinstall_pytorch_with_cuda8()
    
    # AMD GPU Support Methods
    def _install_rocm_pytorch(self):
        """Install ROCm PyTorch for AMD GPUs"""
        print("🔥 Installing ROCm PyTorch for AMD GPU support...")
        
        venv_python = get_venv_python_path(self.project_root)
        
        if sys.platform.startswith('linux'):
            commands = [
                # Uninstall existing PyTorch first
                [venv_python, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
                # Install ROCm PyTorch
                [venv_python, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                 "--index-url", "https://download.pytorch.org/whl/rocm6.0"]
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        print(f"⚠️ Command failed: {' '.join(cmd)}")
                        print(f"Error: {result.stderr}")
                        return False
                except subprocess.TimeoutExpired:
                    print("⚠️ Installation timed out")
                    return False
            
            print("✅ ROCm PyTorch installation completed")
            return True
        else:
            print("❌ ROCm PyTorch only supported on Linux")
            return False
    
    def _install_pytorch_zluda(self):
        """Install PyTorch for ZLUDA compatibility"""
        print("🔥 Installing PyTorch for ZLUDA compatibility...")
        
        # ZLUDA uses regular CUDA PyTorch - the magic happens in the ZLUDA runtime
        return self._reinstall_pytorch_with_cuda8()
    
    def _install_directml(self):
        """Install DirectML for AMD GPU support on Windows"""
        print("🔥 Installing DirectML for AMD GPU support...")
        
        if sys.platform != "win32":
            print("❌ DirectML only supported on Windows")
            return False
        
        venv_python = get_venv_python_path(self.project_root)
        
        commands = [
            # Install DirectML
            [venv_python, "-m", "pip", "install", "torch-directml"]
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"⚠️ Command failed: {' '.join(cmd)}")
                    print(f"Error: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("⚠️ Installation timed out")
                return False
        
        print("✅ DirectML installation completed")
        return True
    
    def _install_amd_support(self):
        """Install appropriate AMD GPU support based on platform"""
        print("🔥 Installing AMD GPU support...")
        
        if sys.platform.startswith('linux'):
            print("🐧 Linux detected - installing ROCm PyTorch...")
            return self._install_rocm_pytorch()
        elif sys.platform == 'win32':
            print("🪟 Windows detected - installing DirectML...")
            return self._install_directml()
        else:
            print("❌ AMD GPU support not available on this platform")
            return False
    
    def _fix_rocm_setup(self):
        """Fix ROCm setup issues"""
        print("🔧 Fixing ROCm setup issues...")
        
        # Check for common ROCm environment issues
        fixes_applied = []
        
        # Set HSA_OVERRIDE_GFX_VERSION if needed
        if not os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
            fixes_applied.append('Set HSA_OVERRIDE_GFX_VERSION=10.3.0')
        
        # Try reinstalling ROCm PyTorch
        if self._install_rocm_pytorch():
            fixes_applied.append('Reinstalled ROCm PyTorch')
        
        if fixes_applied:
            print(f"✅ Applied fixes: {', '.join(fixes_applied)}")
            return True
        else:
            print("❌ Unable to fix ROCm setup")
            return False
    
    def _fix_zluda_setup(self):
        """Fix ZLUDA setup issues"""
        print("🔧 Fixing ZLUDA setup issues...")
        print("ℹ️ ZLUDA requires manual setup - please ensure:")
        print("  1. ZLUDA runtime libraries are in PATH")
        print("  2. CUDA PyTorch is installed")
        print("  3. AMD drivers support compute")
        print("📖 See: https://github.com/vosen/ZLUDA")
        
        # Try reinstalling regular PyTorch for ZLUDA
        if self._reinstall_pytorch_with_cuda8():
            print("✅ Reinstalled PyTorch for ZLUDA compatibility")
            return True
        else:
            print("❌ Unable to fix ZLUDA setup")
            return False
    
    def _fix_directml_setup(self):
        """Fix DirectML setup issues"""
        print("🔧 Fixing DirectML setup issues...")
        
        # Try reinstalling DirectML
        if self._install_directml():
            return True
        else:
            print("❌ Unable to fix DirectML setup")
            return False
    
    def _fix_derrian_backend_install(self, status):
        """Fix Derrian backend installation issues"""
        print("🔧 Attempting to fix Derrian backend issues...")
        
        issues = status.get('issues', [])
        success = True
        
        # Fix missing sd_scripts
        if any('sd_scripts' in issue for issue in issues):
            print("📦 Fixing sd_scripts installation...")
            if not self._install_sd_scripts():
                success = False
        
        # Fix CAME optimizer
        if any('CAME' in issue for issue in issues):
            print("📦 Fixing CAME optimizer installation...")
            if not self._install_came_optimizer():
                success = False
        
        return success
    
    def _install_sd_scripts(self):
        """Ensure sd_scripts is properly installed"""
        try:
            # This should trigger the existing setup logic
            return self.setup_environment()
        except Exception as e:
            print(f"❌ Failed to install sd_scripts: {e}")
            return False
    
    def _install_came_optimizer(self):
        """Ensure CAME optimizer is properly installed"""
        try:
            custom_scheduler_dir = os.path.join(self.derrian_dir, "custom_scheduler")
            if os.path.exists(custom_scheduler_dir):
                # Run the custom optimizer setup
                setup_script = os.path.join(custom_scheduler_dir, "setup.py")
                if os.path.exists(setup_script):
                    pip_cmd = self.correct_venv_path
                    if isinstance(pip_cmd, list):
                        python_cmd = pip_cmd[0]  # Get the Python executable
                    else:
                        python_cmd = 'python'
                    
                    result = subprocess.run(
                        [python_cmd, setup_script, 'install'], 
                        cwd=custom_scheduler_dir,
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    print("✅ CAME optimizer installed successfully")
                    return True
            
            print("❌ CAME optimizer setup files not found")
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install CAME optimizer: {e}")
            return False
    
    @property
    def is_vastai(self):
        """Lazy property - only detects environment when asked"""
        if not self._environment_detected:
            self._detect_environment()
        return self._is_vastai
    
    @property 
    def correct_venv_path(self):
        """Lazy property - only detects environment when asked"""
        if not self._environment_detected:
            self._detect_environment()
        return self._correct_venv_path
        
    def check_setup_status(self):
        """Holy shit dude you forgot to set up go back up mode! 🚨"""
        if not os.path.exists(self.derrian_dir):
            print("🚨 SETUP MISSING: trainer/derrian_backend not found!")
            print("💡 Go back and run the setup widget first!")
            return False
        if not os.path.exists(self.sd_scripts_dir):
            print("🚨 SETUP INCOMPLETE: sd_scripts submodule missing!")
            print("💡 Re-run setup to initialize submodules!")
            return False
        return True

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
        
        # Determine the correct pip path for sd_scripts venv (same logic as _fix_triton_installation)
        if sys.platform == "win32":
            sd_venv_pip = os.path.join(self.sd_scripts_dir, "venv", "Scripts", "pip.exe")
        else:
            sd_venv_pip = os.path.join(self.sd_scripts_dir, "venv", "bin", "pip")
        
        # Check if sd_scripts venv exists, otherwise use system/VastAI pip
        if os.path.exists(sd_venv_pip):
            pip_cmd = sd_venv_pip
            print(f"🎯 Using SD scripts venv pip: {pip_cmd}")
        else:
            pip_cmd = self.correct_venv_path if self.is_vastai else 'pip'
            print(f"🎯 Using system pip: {pip_cmd}")
        
        if os.path.exists(requirements_file):
            print("📦 Installing SD scripts requirements...")
            
            try:
                # Install from the correct working directory so -e . resolves properly
                print(f"🔧 Installing requirements from {self.sd_scripts_dir} directory...")
                
                # Handle both list and string pip commands
                if isinstance(pip_cmd, list):
                    install_cmd = pip_cmd + ['install', '-r', 'requirements.txt']
                else:
                    install_cmd = [pip_cmd, 'install', '-r', 'requirements.txt']
                
                subprocess.run(install_cmd, check=True, cwd=self.sd_scripts_dir)
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
                                subprocess.run([pip_cmd, 'install', line], check=True, capture_output=True)
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
            "onnx==1.15.0",  # For ONNX model format support (Kohya version)
            "onnxruntime==1.17.1",  # For ONNX runtime CPU (required for WD14 taggers)
            "protobuf==3.20.3"  # Required for ONNX compatibility
        ]
        
        for package in additional_packages:
            try:
                subprocess.run([pip_cmd, 'install', package], check=True)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {package}: {e}")
        
        # Try to install GPU-accelerated ONNX runtime if CUDA is available
        try:
            print("🎮 Checking for CUDA to install GPU-accelerated ONNX runtime...")
            # Check if CUDA is available
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected, installing ONNX runtime GPU...")
                # Try CUDA 12 version first (matches modern setups)
                try:
                    subprocess.run([pip_cmd, 'install', 'onnxruntime-gpu==1.17.1', 
                                  '--extra-index-url', 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/'], 
                                  check=True)
                    print("✅ ONNX runtime GPU (CUDA 12) installed")
                except subprocess.CalledProcessError:
                    # Fallback to standard GPU version
                    try:
                        subprocess.run([pip_cmd, 'install', 'onnxruntime-gpu==1.17.1'], check=True)
                        print("✅ ONNX runtime GPU (standard) installed")
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️ GPU ONNX runtime install failed: {e}")
            else:
                print("ℹ️ No NVIDIA GPU detected, using CPU-only ONNX runtime")
        except FileNotFoundError:
            print("ℹ️ nvidia-smi not found, using CPU-only ONNX runtime")
        
        # Fix Triton installation for Docker/VastAI compatibility
        self._fix_triton_installation()
        
        if not os.path.exists(requirements_file):
            print("⚠️ SD scripts requirements.txt not found")
    
    def _fix_triton_installation(self):
        """Fix Triton installation for bitsandbytes AdamW8bit compatibility"""
        print("🔧 Installing Triton for bitsandbytes (AdamW8bit) compatibility...")
        
        # Determine the correct pip path for sd_scripts venv
        if sys.platform == "win32":
            sd_venv_pip = os.path.join(self.sd_scripts_dir, "venv", "Scripts", "pip.exe")
        else:
            sd_venv_pip = os.path.join(self.sd_scripts_dir, "venv", "bin", "pip")
        
        # Check if sd_scripts venv exists, otherwise use system/VastAI pip
        if os.path.exists(sd_venv_pip):
            pip_cmd = sd_venv_pip
            print(f"🎯 Using SD scripts venv pip: {pip_cmd}")
        else:
            pip_cmd = self.correct_venv_path if self.is_vastai else 'pip'
            print(f"🎯 Using system pip: {pip_cmd}")
        
        # Test if Triton is already working in the target environment
        try:
            test_cmd = [pip_cmd.replace('pip', 'python') if 'bin' in pip_cmd or 'Scripts' in pip_cmd else sys.executable, 
                       '-c', 'import triton; print("Triton version:", triton.__version__)']
            result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
            print(f"✅ Triton already working: {result.stdout.strip()}")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Triton not found or not working, installing...")
        
        # Install Triton with CUDA 12.4 compatibility (matches Kohya's PyTorch)
        triton_install_methods = [
            # Method 1: PyTorch CUDA 12.4 index (matches Kohya exactly)
            [pip_cmd, 'install', '--force-reinstall', 'triton', '--index-url=https://download.pytorch.org/whl/cu124'],
            
            # Method 2: Force Linux x86_64 platform with CUDA 12.4
            [pip_cmd, 'install', '--force-reinstall', 'triton', '--platform=linux_x86_64', '--only-binary=:all:', 
             '--index-url=https://download.pytorch.org/whl/cu124'],
            
            # Method 3: Specific Triton version known to work with PyTorch 2.5.1
            [pip_cmd, 'install', '--force-reinstall', 'triton==3.0.0'],
            
            # Method 4: Fallback to PyPI with no cache
            [pip_cmd, 'install', '--force-reinstall', '--no-cache-dir', 'triton']
        ]
        
        for i, method in enumerate(triton_install_methods):
            try:
                print(f"🔄 Method {i+1}: Installing Triton...")
                subprocess.run(method, check=True, cwd=self.sd_scripts_dir if os.path.exists(self.sd_scripts_dir) else None)
                
                # Test if it works in the target environment
                test_cmd = [pip_cmd.replace('pip', 'python') if 'bin' in pip_cmd or 'Scripts' in pip_cmd else sys.executable,
                           '-c', 'import triton; print("✅ Triton working!")']
                subprocess.run(test_cmd, check=True, capture_output=True)
                print(f"✅ Triton installation method {i+1} succeeded!")
                
                # Also test bitsandbytes import since that's what we really care about
                try:
                    test_bnb_cmd = [pip_cmd.replace('pip', 'python') if 'bin' in pip_cmd or 'Scripts' in pip_cmd else sys.executable,
                                   '-c', 'import bitsandbytes; print("✅ bitsandbytes can use Triton!")']
                    subprocess.run(test_bnb_cmd, check=True, capture_output=True)
                    print("✅ bitsandbytes + Triton integration working!")
                except subprocess.CalledProcessError:
                    print("⚠️ Triton installed but bitsandbytes integration may have issues")
                
                return
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Method {i+1} failed: {e}")
                continue
        
        print("⚠️ All Triton installation methods failed")
        print("💡 AdamW8bit will fallback to slower implementations or regular AdamW")
        print("💡 Training will still work, just without 8-bit optimization benefits")
            
    def _verify_installation(self):
        """Verify all components are working"""
        
        # Check SD scripts
        train_script = os.path.join(self.sd_scripts_dir, "train_network.py")
        if os.path.exists(train_script):
            print("   ✅ Kohya SD training scripts")
        else:
            print("   ❌ Kohya SD training scripts missing")
            
        # Determine the correct python path for sd_scripts venv to test imports
        if sys.platform == "win32":
            sd_venv_python = os.path.join(self.sd_scripts_dir, "venv", "Scripts", "python.exe")
        else:
            sd_venv_python = os.path.join(self.sd_scripts_dir, "venv", "bin", "python")
        
        # Use sd_scripts venv if it exists, otherwise use system python
        if os.path.exists(sd_venv_python):
            test_python = sd_venv_python
            print(f"   🎯 Testing imports with SD scripts venv: {test_python}")
        else:
            test_python = sys.executable
            print(f"   🎯 Testing imports with system Python: {test_python}")
            
        # Test imports in the target environment
        def test_import(module_name, description):
            try:
                result = subprocess.run([test_python, '-c', f'import {module_name}'], 
                                      check=True, capture_output=True, text=True)
                print(f"   ✅ {description}")
                return True
            except subprocess.CalledProcessError:
                print(f"   ❌ {module_name} import failed: {description}")
                return False
        
        # Test LyCORIS import
        if os.path.exists(self.lycoris_dir):
            print("   ✅ LyCORIS directory available (DoRA, LoHa, LoKr, etc.)")
        else:
            print("   ❌ LyCORIS directory missing")
        
        # Test core package imports in the sd_scripts environment
        test_import('pytorch_optimizer', 'PyTorch Optimizer: CAME, Prodigy, etc.')
        test_import('safetensors', 'SafeTensors: model loading')
        
        # Test custom optimizers
        try:
            result = subprocess.run([test_python, '-c', 'import LoraEasyCustomOptimizer'], 
                                  check=True, capture_output=True, text=True)
            print("   ✅ Custom optimizers: Derrian's utilities")
        except subprocess.CalledProcessError:
            print("   ❌ Custom optimizers: No module named 'LoraEasyCustomOptimizer'")
            
        # Test SD scripts library (most important!)
        try:
            # Add sd_scripts to Python path and test import
            import_test = f"import sys; sys.path.insert(0, '{self.sd_scripts_dir}'); import library"
            result = subprocess.run([test_python, '-c', import_test], 
                                  check=True, capture_output=True, text=True)
            print("   ✅ Derrian's utilities: library module")
        except subprocess.CalledProcessError:
            print("   ❌ Derrian's utilities: No module named 'library'")
            
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


