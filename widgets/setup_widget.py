# widgets/setup_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.managers import SetupManager, ModelManager
import os
import subprocess
import shutil

class SetupWidget:
    def __init__(self, setup_manager=None, model_manager=None):
        # Use dependency injection - accept manager instances or create defaults
        if setup_manager is None:
            setup_manager = SetupManager()
        if model_manager is None:
            model_manager = ModelManager()
            
        self.setup_manager = setup_manager
        self.model_manager = model_manager
        self.container_info = self._detect_container_environment()
        self.create_widgets()
    
    def _detect_container_environment(self):
        """Detects if running in a container and what type"""
        info = {
            'is_container': False,
            'is_vastai': False,
            'is_colab': False,
            'has_gpu': False,
            'cuda_visible': '',
            'environment': 'local'
        }
        
        # Check for container indicators
        if os.path.exists('/.dockerenv') or os.environ.get('CONTAINER') == 'docker':
            info['is_container'] = True
            
        # Check for VastAI (the important one!)
        if os.environ.get('VAST_CONTAINERLABEL') or '/workspace' in os.getcwd():
            info['is_vastai'] = True
            info['environment'] = 'vastai'
            
        # Check for GPU availability
        info['cuda_visible'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        info['has_gpu'] = bool(info['cuda_visible'] or os.path.exists('/proc/driver/nvidia/version'))
        
        return info

    def _get_environment_status_html(self):
        """Creates HTML status display for current environment"""
        if self.container_info['is_vastai']:
            return "<div style='padding: 10px; border: 1px solid #28a745; border-radius: 5px; margin: 10px 0;'><strong>☁️ VastAI Container Detected</strong><br>Optimized settings will be applied automatically.</div>"
        elif self.container_info['is_container']:
            return "<div style='padding: 10px; border: 1px solid #007acc; border-radius: 5px; margin: 10px 0;'><strong>📦 Docker Container Detected</strong><br>Container environment active.</div>"
        else:
            gpu_status = "🟢 GPU Available" if self.container_info['has_gpu'] else "🔴 No GPU Detected"
            return f"<div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 10px 0;'><strong>💻 Local Environment</strong> | {gpu_status}</div>"

    def create_widgets(self):
        """Creates all the UI components for the widget."""
        header_icon = "🚩"
        env_status = self._get_environment_status_html()
        header_main = widgets.HTML(f"<h2>{header_icon} 1. Setup & Models</h2>{env_status}")
        
        # --- Environment Section ---
        env_desc = widgets.HTML("<h3>▶️ Environment Setup</h3><p>Install training backend and dependencies. Run this first before any other steps.</p>")
        
        # Environment buttons
        button_layout = widgets.Layout(width='200px', margin='2px')
        self.validate_button = widgets.Button(description="🔍 Validate Environment", button_style='info', layout=button_layout)
        self.env_button = widgets.Button(description="🚩 Setup Environment", button_style='primary', layout=button_layout)
        
        button_box = widgets.HBox([self.validate_button, self.env_button])
        self.env_status = widgets.HTML("<div style='padding: 8px; border: 1px solid #007acc;'><strong>📊 Status:</strong> Ready for setup</div>")
        self.env_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        env_box = widgets.VBox([env_desc, button_box, self.env_status, self.env_output])

        # --- Model Section ---
        model_desc = widgets.HTML("<h3>▶️ Base Model</h3><p>Choose a pre-trained model or enter a custom URL from HuggingFace or Civitai.</p>")
        
        # Popular model presets - adjust based on environment
        model_presets = self._get_model_presets()
        
        self.model_preset = widgets.Dropdown(
            options=list(model_presets.keys()),
            description='Preset:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )
        
        self.model_url = widgets.Text(
            placeholder="Enter custom HuggingFace or Civitai URL", 
            layout=widgets.Layout(width='99%')
        )
        
        self.model_name = widgets.Text(
            description="Model Name:",
            placeholder="(Optional) Custom filename",
            layout=widgets.Layout(width='99%')
        )
        
        self.model_button = widgets.Button(description="📥 Download Model", button_style='success')
        self.model_status = widgets.HTML("<div style='padding: 8px; border: 1px solid #007acc;'><strong>📊 Status:</strong> Ready to download</div>")
        self.model_output = widgets.Output(layout=widgets.Layout(height='250px', overflow='scroll', border='1px solid #ddd'))
        
        # Update URL when preset changes
        def on_preset_change(change):
            if change['new'] in model_presets and model_presets[change['new']]:
                self.model_url.value = model_presets[change['new']]
        self.model_preset.observe(on_preset_change, names='value')
        
        model_box = widgets.VBox([
            model_desc,
            self.model_preset,
            self.model_url,
            self.model_name,
            self.model_button,
            self.model_status,
            self.model_output
        ])

        # --- VAE Section ---
        vae_desc = widgets.HTML("<h3>▶️ VAE (Optional)</h3><p>VAE is not required for most models but can improve quality. SDXL models benefit from the SDXL VAE.</p>")
        
        vae_presets = {
            "None (Skip VAE)": "",
            "Custom URL (enter below)": "custom",
            "SDXL VAE (recommended for XL models)": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
        }
        
        self.vae_preset = widgets.Dropdown(
            options=list(vae_presets.keys()),
            description='Preset:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )
        
        self.vae_url = widgets.Text(
            placeholder="(Optional) Enter custom VAE URL", 
            layout=widgets.Layout(width='99%')
        )
        
        self.vae_name = widgets.Text(
            description="VAE Name:",
            placeholder="(Optional) Custom filename",
            layout=widgets.Layout(width='99%')
        )
        
        self.vae_button = widgets.Button(description="📥 Download VAE", button_style='success')
        self.vae_status = widgets.HTML("<div style='padding: 8px; border: 1px solid #007acc;'><strong>📊 Status:</strong> Ready to download</div>")
        self.vae_output = widgets.Output(layout=widgets.Layout(height='250px', overflow='scroll', border='1px solid #ddd'))
        
        # Update VAE URL when preset changes
        def on_vae_preset_change(change):
            if change['new'] in vae_presets:
                if vae_presets[change['new']] == "custom":
                    self.vae_url.value = ""
                else:
                    self.vae_url.value = vae_presets[change['new']]
        self.vae_preset.observe(on_vae_preset_change, names='value')
        
        vae_box = widgets.VBox([
            vae_desc,
            self.vae_preset,
            self.vae_url,
            self.vae_name,
            self.vae_button,
            self.vae_status,
            self.vae_output
        ])

        # --- API Tokens Section ---
        api_desc = widgets.HTML("<h3>▶️ API Tokens (Optional)</h3><p>Required only for private repositories or gated models.</p>")
        self.hf_token = widgets.Password(
            description="HF Token:", 
            placeholder="HuggingFace token for private repos", 
            layout=widgets.Layout(width='99%')
        )
        self.civitai_token = widgets.Password(
            description="Civitai Token:", 
            placeholder="Civitai API key for downloads", 
            layout=widgets.Layout(width='99%')
        )
        api_box = widgets.VBox([api_desc, self.hf_token, self.civitai_token])

        # --- Accordion for Sections ---
        self.accordion = widgets.Accordion(children=[
            env_box,
            model_box,
            vae_box,
            api_box
        ])
        self.accordion.set_title(0, "▶️ Environment Setup")
        self.accordion.set_title(1, "▶️ Base Model Download")
        self.accordion.set_title(2, "▶️ VAE Download")
        self.accordion.set_title(3, "▶️ API Tokens")
        
        # --- Status Display ---
        self.status_output = widgets.Output()
        
        # --- Main Widget Box ---
        self.widget_box = widgets.VBox([
            header_main,
            self.accordion,
            self.status_output
        ])

        # --- Button Click Events ---
        self.validate_button.on_click(self.run_validate_environment)
        self.env_button.on_click(self.run_setup_environment)
        self.model_button.on_click(self.run_download_model)
        self.vae_button.on_click(self.run_download_vae)
        
    def _get_model_presets(self):
        """Returns model presets optimized for current environment"""
        base_presets = {
            "Custom URL (enter below)": "",
            "(XL) Illustrious v0.1": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
            "(XL) NoobAI Epsilon v1.0": "https://huggingface.co/Laxhar/noobai-XL-1.0/resolve/main/NoobAI-XL-v1.0.safetensors", 
            "(XL) PonyDiffusion v6": "https://huggingface.co/AstraliteHeart/pony-diffusion-v6/resolve/main/v6.safetensors",
            "(XL) Animagine 3.1": "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors",
            "(XL) SDXL 1.0 Base": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
            "(1.5) Anime Full Final": "https://huggingface.co/hollowstrawberry/stable-diffusion-guide/resolve/main/models/animefull-final-pruned-fp16.safetensors",
            "(1.5) AnyLora": "https://huggingface.co/Lykon/AnyLoRA/resolve/main/AnyLoRA_noVae_fp16-pruned.safetensors",
            "(1.5) SD 1.5": "https://huggingface.co/hollowstrawberry/stable-diffusion-guide/resolve/main/models/sd-v1-5-pruned-noema-fp16.safetensors"
        }
        
        # For VastAI, prioritize the most popular/stable models
        if self.container_info['is_vastai']:
            vastai_order = {
                "(XL) Illustrious v0.1 ⭐ Popular": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
                "(XL) PonyDiffusion v6 ⭐ Popular": base_presets["(XL) PonyDiffusion v6"],
                "(XL) NoobAI Epsilon v1.0 ⭐ Popular": base_presets["(XL) NoobAI Epsilon v1.0"],
                "Custom URL (enter below)": ""
            }
            vastai_order.update({k: v for k, v in base_presets.items() if k not in vastai_order and "Custom" not in k})
            return vastai_order
            
        return base_presets

    def run_setup_environment(self, b):
        """Sets up the training environment"""
        self.env_output.clear_output()
        self.env_status.value = "<div style='padding: 8px; border: 1px solid #6c757d;'><strong>⚙️ Status:</strong> Setting up environment...</div>"
        with self.env_output:
            if self.container_info['environment'] != 'local':
                print(f"🔧 Setting up training environment for {self.container_info['environment']}...")
            else:
                print("🔧 Setting up training environment...")
                
            success = self.setup_manager.setup_environment()
            if success:
                status_msg = "Environment setup complete! You can now download models and prepare datasets."
                if self.container_info['is_vastai']:
                    status_msg += "<br>📝 VastAI optimizations applied automatically."
                self.env_status.value = f"<div style='padding: 8px; border: 1px solid #28a745;'><strong>✅ Status:</strong> {status_msg}</div>"
            else:
                self.env_status.value = "<div style='padding: 8px; border: 1px solid #dc3545;'><strong>❌ Status:</strong> Environment setup failed. Check logs.</div>"
    
    def run_validate_environment(self, b):
        """Validates the current environment without installing anything"""
        self.env_output.clear_output()
        self.env_status.value = "<div style='padding: 8px; border: 1px solid #6c757d;'><strong>⚙️ Status:</strong> Validating environment...</div>"
        with self.env_output:
            print("🔍 Running comprehensive environment validation...\n")
            
            # Check system info
            try:
                import platform
                print(f"🖥️  System: {platform.system()} {platform.architecture()[0]}")
                print(f"🐍 Python: {platform.python_version()}")
                
                # Check memory if available
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemTotal:' in line:
                                ram_gb = int(line.split()[1]) / (1024 * 1024)
                                print(f"💾 RAM: {ram_gb:.1f} GB")
                                break
                except:
                    print("💾 RAM: Unknown")
            except Exception as e:
                print(f"System info error: {e}")
            
            # Check required commands
            print("\n🔧 Checking required commands...")
            required_commands = ['git', 'aria2c', 'python3', 'pip', 'curl', 'wget']
            missing = []
            
            for cmd in required_commands:
                if shutil.which(cmd):
                    print(f"   ✅ {cmd}")
                else:
                    print(f"   ❌ {cmd} (missing)")
                    missing.append(cmd)
            
            # Check GPU
            print("\n🎮 Checking GPU support...")
            if os.path.exists('/proc/driver/nvidia/version'):
                print("   ✅ NVIDIA driver detected")
                
                # Try nvidia-smi
                if shutil.which('nvidia-smi'):
                    try:
                        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            for line in result.stdout.strip().split('\n'):
                                if line.strip():
                                    parts = line.split(', ')
                                    if len(parts) >= 2:
                                        name, memory = parts[0], parts[1]
                                        print(f"   🎮 GPU: {name.strip()} ({int(memory)/1024:.1f} GB VRAM)")
                    except Exception as e:
                        print(f"   ⚠️ nvidia-smi error: {e}")
                else:
                    print("   ⚠️ nvidia-smi not available")
            else:
                print("   ❌ No NVIDIA driver detected")
            
            # Check PyTorch if available
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"   ✅ PyTorch CUDA available ({torch.cuda.device_count()} devices)")
                else:
                    print("   ⚠️ PyTorch CUDA not available")
            except ImportError:
                print("   ℹ️ PyTorch not installed (will be installed during setup)")
            
            # Check storage space
            print("\n💽 Checking storage space...")
            try:
                import shutil as disk_util
                total, used, free = disk_util.disk_usage('.')
                free_gb = free / (1024**3)
                total_gb = total / (1024**3)
                usage_pct = (used / total) * 100
                print(f"   💾 Storage: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({usage_pct:.1f}% used)")
                
                if free_gb < 10:
                    print("   ⚠️ Low disk space! Training may require 10+ GB")
            except Exception as e:
                print(f"   ⚠️ Storage check failed: {e}")
            
            # Check network connectivity
            print("\n🌐 Checking network connectivity...")
            test_urls = ['https://github.com', 'https://huggingface.co', 'https://civitai.com']
            
            for url in test_urls:
                try:
                    result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 
                                           '--connect-timeout', '5', url], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip() == '200':
                        service = url.split('//')[1].split('.')[0]
                        print(f"   ✅ {service}")
                    else:
                        print(f"   ⚠️ {url}: HTTP {result.stdout.strip()}")
                except Exception as e:
                    print(f"   ❌ {url}: {e}")
            
            # Check SD Scripts installation
            print("\n📂 Checking SD Scripts installation...")
            trainer_dir = os.path.join(os.path.dirname(__file__), '..', 'trainer')
            sd_scripts_dir = os.path.join(trainer_dir, 'derrian_backend', 'sd_scripts')
            
            if os.path.exists(trainer_dir):
                print("   ✅ Trainer directory exists")
                
                if os.path.exists(sd_scripts_dir):
                    print("   ✅ SD Scripts directory exists")
                    
                    # Check for key training scripts
                    key_scripts = ['sdxl_train_network.py', 'train_network.py', 'networks/lora.py']
                    scripts_found = 0
                    for script in key_scripts:
                        script_path = os.path.join(sd_scripts_dir, script)
                        if os.path.exists(script_path):
                            scripts_found += 1
                            print(f"   ✅ {script}")
                        else:
                            print(f"   ❌ {script} (missing)")
                    
                    if scripts_found >= 2:
                        print("   ✅ SD Scripts appear properly installed")
                    else:
                        print("   ❌ SD Scripts incomplete - run Setup Environment")
                else:
                    print("   ❌ SD Scripts directory missing - run Setup Environment")
            else:
                print("   ❌ Trainer directory missing - run Setup Environment")
            
            # Summary
            print("\n" + "="*50)
            if missing:
                self.env_status.value = f"<div style='padding: 8px; border: 1px solid #ffc107;'><strong>⚠️ Status:</strong> Validation complete - {len(missing)} missing dependencies</div>"
                print("\n📝 Missing commands can be installed during setup:")
                for cmd in missing:
                    print(f"   • {cmd}")
            else:
                self.env_status.value = "<div style='padding: 8px; border: 1px solid #28a745;'><strong>✅ Status:</strong> Validation complete - Environment ready for setup!</div>"
            
            # Container detection info
            if self.container_info['environment'] != 'local':
                print(f"\n📦 Container Environment: {self.container_info['environment']}")
                print("   Optimized settings will be applied automatically.")

    def run_download_model(self, b):
        """Downloads the selected model"""
        self.model_output.clear_output()
        self.model_status.value = "<div style='padding: 8px; border: 1px solid #6c757d;'><strong>⚙️ Status:</strong> Downloading model...</div>"
        with self.model_output:
            model_url = self.model_url.value.strip()
            if not model_url:
                self.model_status.value = "<div style='padding: 8px; border: 1px solid #dc3545;'><strong>❌ Status:</strong> Please enter a model URL or select a preset.</div>"
                print("❌ Please enter a model URL or select a preset.")
                return
            
            # Determine which token to use
            api_token = ""
            if "civitai.com" in model_url and self.civitai_token.value:
                api_token = self.civitai_token.value
            elif "huggingface.co" in model_url and self.hf_token.value:
                api_token = self.hf_token.value
            
            model_name = self.model_name.value.strip() if self.model_name.value.strip() else None
            
            result = self.model_manager.download_model(
                model_url, 
                model_name=model_name, 
                api_token=api_token
            )
            
            if result:
                self.model_status.value = f"<div style='padding: 8px; border: 1px solid #28a745;'><strong>✅ Status:</strong> Model downloaded: {os.path.basename(result)}</div>"
            else:
                self.model_status.value = "<div style='padding: 8px; border: 1px solid #dc3545;'><strong>❌ Status:</strong> Model download failed. Check logs.</div>"

    def run_download_vae(self, b):
        """Downloads the selected VAE"""
        self.vae_output.clear_output()
        self.vae_status.value = "<div style='padding: 8px; border: 1px solid #6c757d;'><strong>⚙️ Status:</strong> Downloading VAE...</div>"
        with self.vae_output:
            vae_url = self.vae_url.value.strip()
            if not vae_url:
                self.vae_status.value = "<div style='padding: 8px; border: 1px solid #17a2b8;'><strong>ℹ️ Status:</strong> No VAE URL provided, skipping.</div>"
                print("ℹ️ No VAE URL provided, skipping VAE download.")
                return
            
            # Determine which token to use
            api_token = ""
            if "civitai.com" in vae_url and self.civitai_token.value:
                api_token = self.civitai_token.value
            elif "huggingface.co" in vae_url and self.hf_token.value:
                api_token = self.hf_token.value
            
            vae_name = self.vae_name.value.strip() if self.vae_name.value.strip() else None
            
            result = self.model_manager.download_vae(
                vae_url, 
                vae_name=vae_name, 
                api_token=api_token
            )
            
            if result:
                self.vae_status.value = f"<div style='padding: 8px; border: 1px solid #28a745;'><strong>✅ Status:</strong> VAE downloaded: {os.path.basename(result)}</div>"
            else:
                self.vae_status.value = "<div style='padding: 8px; border: 1px solid #dc3545;'><strong>❌ Status:</strong> VAE download failed. Check logs.</div>"

    def display(self):
        """Displays the widget."""
        display(self.widget_box)
