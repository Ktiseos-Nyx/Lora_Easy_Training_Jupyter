# core/training_manager.py
import subprocess
import os
import sys
import toml
import json
from typing import Dict, List, Optional, Any

class HybridTrainingManager:
    """🧪 FRANKENSTEIN TRAINING MANAGER 💥
    
    Combines the best features from:
    - Kohya-ss (stable foundation)
    - LyCORIS (advanced methods)
    - Derrian Distro (CAME + REX)
    - OneTrainer (fused back pass)
    - HakuLatent (future research)
    """
    
    def __init__(self):
        # Use current working directory to match where notebook is running
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.config_dir = os.path.join(self.project_root, "training_configs")  # Much better name!
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.output_dir = os.path.join(self.project_root, "output")
        self.logging_dir = os.path.join(self.project_root, "logs")
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # 🔧 Setup Python path for Derrian's custom optimizers
        self._setup_custom_optimizers()
        
        # 🔬 Advanced feature support
        self.advanced_optimizers = self._init_advanced_optimizers()
        self.standard_optimizers = self._init_standard_optimizers()
        self.lycoris_methods = self._init_lycoris_methods()
        self.experimental_features = self._init_experimental_features()
    
    
    
    def _find_training_script(self):
        """Find the training script in various possible locations"""
        possible_scripts = [
            # Derrian's backend might have scripts in different locations
            os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts", "sdxl_train_network.py"),
            os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts", "train_network.py"), 
            os.path.join(self.trainer_dir, "sdxl_train_network.py"),
            os.path.join(self.trainer_dir, "train_network.py"),
            # Also check for kohya scripts structure
            os.path.join(self.sd_scripts_dir, "sdxl_train_network.py"),
            os.path.join(self.sd_scripts_dir, "train_network.py")
        ]
        
        for script_path in possible_scripts:
            if os.path.exists(script_path):
                script_type = "SDXL" if "sdxl" in os.path.basename(script_path) else "SD 1.5"
                print(f"✅ Found {script_type} training script: {script_path}")
                return script_path
        
        return None
    
    def _setup_custom_optimizers(self):
        """Add paths for submodule components (silent mode)"""
        # Add LyCORIS to Python path
        lycoris_dir = os.path.join(self.trainer_dir, "derrian_backend", "lycoris")
        if os.path.exists(lycoris_dir) and lycoris_dir not in sys.path:
            sys.path.insert(0, lycoris_dir)
            
        # Add Derrian's custom optimizers to path  
        derrian_dir = os.path.join(self.trainer_dir, "derrian_backend")
        custom_scheduler_dir = os.path.join(derrian_dir, "custom_scheduler")
        if os.path.exists(custom_scheduler_dir) and custom_scheduler_dir not in sys.path:
            sys.path.insert(0, custom_scheduler_dir)
            
        # Silently check for custom optimizers
            
        # Try to import and verify CAME is available (from custom_scheduler)
        try:
            import LoraEasyCustomOptimizer.came
            pass  # CAME available
        except ImportError as e:
            print(f"⚠️ CAME optimizer not found: {e}")
            
            # Try alternative import paths for Derrian's structure
            alt_paths = [
                'came', 'optimizers.came', 'custom_optimizers.came',
                'custom_scheduler.LoraEasyCustomOptimizer.came'
            ]
            
            for alt_path in alt_paths:
                try:
                    __import__(alt_path)
                    print(f"✅ Found CAME at: {alt_path}")
                    break
                except ImportError:
                    continue
            else:
                print("💡 CAME optimizer not found in any expected location")
            
        # Try to import REX scheduler
        try:
            import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
            print("✅ REX scheduler found and ready!")
        except ImportError as e:
            print(f"⚠️ REX scheduler not found: {e}")
            print("💡 Custom optimizers might not be available - using standard optimizers")
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float with fallback"""
        try:
            if isinstance(value, str):
                return float(value.strip()) if value.strip() else default
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_warmup_steps(self, config):
        """Calculate warmup steps based on ACTUAL training steps, not hardcoded bias"""
        try:
            # Get actual training parameters
            dataset_size = config.get('dataset_size', 0)
            if dataset_size == 0:
                # Fallback: try to count images from dataset directory
                dataset_dir = config.get('dataset_dir', '')
                if dataset_dir:
                    from core.image_utils import count_images_in_directory
                    dataset_size = count_images_in_directory(dataset_dir)
            
            num_repeats = config.get('num_repeats', 1)
            epochs = config.get('epochs', 1)
            batch_size = config.get('train_batch_size', 1)
            warmup_ratio = self._safe_float(config.get('lr_warmup_ratio', 0.0))
            
            if dataset_size > 0 and batch_size > 0:
                # Calculate REAL total training steps
                total_steps = (dataset_size * num_repeats * epochs) // batch_size
                warmup_steps = int(total_steps * warmup_ratio)
                
                print(f"🔢 UNBIASED WARMUP CALCULATION:")
                print(f"   📸 Dataset size: {dataset_size}")
                print(f"   🔄 Repeats: {num_repeats}")
                print(f"   📅 Epochs: {epochs}")
                print(f"   📦 Batch size: {batch_size}")
                print(f"   ⚡ Total steps: {total_steps}")
                print(f"   🌡️ Warmup ratio: {warmup_ratio}")
                print(f"   🔥 Warmup steps: {warmup_steps}")
                
                return warmup_steps
            else:
                print("⚠️ Cannot calculate warmup steps - missing dataset info, using fallback")
                # Fallback: estimate reasonable total steps based on common dataset sizes
                estimated_dataset_size = 100  # Conservative estimate
                estimated_total_steps = (estimated_dataset_size * num_repeats * epochs) // batch_size
                warmup_steps = int(estimated_total_steps * warmup_ratio)
                
                print(f"📊 FALLBACK WARMUP CALCULATION:")
                print(f"   📸 Estimated dataset size: {estimated_dataset_size}")
                print(f"   ⚡ Estimated total steps: {estimated_total_steps}")
                print(f"   🔥 Fallback warmup steps: {warmup_steps}")
                
                return warmup_steps
                
        except Exception as e:
            print(f"❌ Error calculating warmup steps: {e}")
            # Final fallback - still use warmup ratio calculation
            epochs = config.get('epochs', 1)
            estimated_steps = max(epochs * 50, 100)  # Minimum reasonable steps
            warmup_ratio = self._safe_float(config.get('lr_warmup_ratio', 0.0))
            return int(estimated_steps * warmup_ratio)
    
    def _detect_model_type(self, config):
        """Detect model type from model path or config to choose appropriate SD scripts"""
        model_path = config.get('model_path', '').lower()
        
        # Check for Flux indicators
        if any(flux_indicator in model_path for flux_indicator in [
            'flux', 'schnell', 'dev', 'black-forest-labs'
        ]):
            return 'flux'
        
        # Check for SD3 indicators  
        if any(sd3_indicator in model_path for sd3_indicator in [
            'sd3', 'stable-diffusion-3', 'sd3-medium', 'sd3-large'
        ]):
            return 'sd3'
        
        # Check for SDXL indicators
        if any(sdxl_indicator in model_path for sdxl_indicator in [
            'sdxl', 'xl', '1024', 'base_1.0', 'refiner'
        ]):
            return 'sdxl'
        
        # Default to SD 1.5/2.x
        return 'sd15'
    
    
        
    def _init_advanced_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize advanced optimizer configurations"""
        return {
            'came': {
                'optimizer_type': 'LoraEasyCustomOptimizer.came.CAME',
                'scheduler': 'LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts',
                'loss_type': 'huber',
                'args': ['weight_decay=0.04', 'eps=1e-16'],
                'scheduler_args': ['min_lr=1e-9', 'gamma=0.9', 'd=0.9'],
                'description': 'Derrian\'s memory-efficient optimizer'
            },
            'prodigy_plus': {
                'optimizer_type': 'ProdigyPlusScheduleFree',
                'scheduler': 'constant',  # Schedule-free
                'args': ['decouple=True', 'weight_decay=0.01', 'schedule_free=True'],
                'learning_rate_override': 1.0,
                'description': 'OneTrainer\'s learning rate AND schedule free'
            },
            'stable_adamw': {
                'optimizer_type': 'StableAdamW',
                'args': ['weight_decay=0.01', 'betas=[0.9,0.999]', 'eps=1e-8'],
                'description': 'Research-grade stability improvements'
            },
            'adopt': {
                'optimizer_type': 'ADOPT',
                'args': ['clip_threshold=1.0', 'eps=1e-6'],
                'description': 'Bleeding-edge adaptive gradient clipping'
            }
        }
    
    def _init_standard_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard optimizer configurations"""
        return {
            'Prodigy': {
                'optimizer_type': 'Prodigy',  # Let SD scripts handle import
                'args': ['decouple=True', 'weight_decay=0.01', 'betas=[0.9,0.999]', 'd_coef=2', 'use_bias_correction=True'],
                'warmup_args': ['safeguard_warmup=True'],
                'description': 'Adaptive learning rate optimizer'
            },
            'AdamW': {
                'optimizer_type': 'AdamW',
                'args': ['weight_decay=0.01', 'betas=[0.9,0.999]'],
                'description': 'Standard AdamW optimizer'
            },
            'AdamW8bit': {
                'optimizer_type': 'bitsandbytes.optim.AdamW8bit',
                'args': ['weight_decay=0.1', 'betas=[0.9,0.99]'],
                'description': '8-bit AdamW for memory efficiency'
            },
            'AdaFactor': {
                'optimizer_type': 'transformers.optimization.Adafactor',
                'args': ['scale_parameter=False', 'relative_step=False', 'warmup_init=False'],
                'description': 'Memory-efficient AdaFactor optimizer'
            },
            'Came': {
                'optimizer_type': 'LoraEasyCustomOptimizer.came.CAME',
                'args': ['weight_decay=0.04'],
                'description': 'Custom CAME optimizer'
            },
            'DAdaptation': {
                'optimizer_type': 'DAdaptation',
                'args': ['weight_decay=0.01'],
                'description': 'D-Adaptation optimizer'
            },
            'DadaptAdam': {
                'optimizer_type': 'DadaptAdam',
                'args': ['weight_decay=0.01'],
                'description': 'D-Adaptation Adam variant'
            },
            'DadaptLion': {
                'optimizer_type': 'DadaptLion',
                'args': ['weight_decay=0.01'],
                'description': 'D-Adaptation Lion variant'
            },
            'Lion': {
                'optimizer_type': 'Lion',
                'args': ['weight_decay=0.01', 'betas=[0.9,0.99]'],
                'description': 'Lion optimizer'
            },
            'SGDNesterov': {
                'optimizer_type': 'SGDNesterov',
                'args': ['momentum=0.9', 'weight_decay=0.01'],
                'description': 'SGD with Nesterov momentum'
            },
            'SGDNesterov8bit': {
                'optimizer_type': 'bitsandbytes.optim.SGD8bit',
                'args': ['momentum=0.9', 'weight_decay=0.01'],
                'description': '8-bit SGD with Nesterov momentum'
            }
        }
    
    def _init_lycoris_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize LyCORIS method configurations - OFFICIAL Algo-List.md specifications"""
        return {
            # DoRA removed - handled in main LoRA type selection
            'loha': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=loha'],
                'description': 'LoHa: Low-rank Hadamard (dim≤32, alpha≤dim, rank≤dim²)',
                'recommendations': 'Recommended: dim≤32, alpha from 1 to dim. Higher dim may cause unstable loss.'
            },
            'lokr': {
                'network_module': 'lycoris.kohya', 
                'network_args': ['algo=lokr'],  # Removed factor, let LyCORIS handle
                'description': 'LoKr: Low-rank Kronecker (small/large LoKr configs available)',
                'recommendations': 'Can use "full dimension" by setting very large dimension.'
            },
            'ia3': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=ia3'],
                'description': '(IA)³: Experimental method (requires higher learning rate, <1MB files)',
                'recommendations': 'Requires higher learning rate. Produces very small files (<1 MB).'
            },
            'dylora': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=dylora'],  # Removed block_size, use defaults
                'description': 'DyLoRA: Updates one row/column per step (large dim recommended)',
                'recommendations': 'Recommended: large dim with alpha=dim/4 to dim.'
            },
            'boft': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=boft'],
                'description': 'BOFT: Advanced Diag-OFT using butterfly operation for orthogonal matrix',
                'recommendations': 'Advanced version of Diag-OFT with butterfly operations.'
            },
            'glora': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=glora'],
                'description': 'GLoRA: Generalized LoRA (experimental)',
                'recommendations': 'Part of experimental methods, limited documentation available.'
            },
            'full': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=full'],
                'description': 'Full: Native fine-tuning (largest files, potentially best results)',
                'recommendations': 'Can potentially give the best result if tuned correctly. Largest file but no slower training.'
            },
            'diag_oft': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=diag-oft'],  # Fixed: uses hyphen not underscore!
                'description': 'Diag-OFT: Preserves hyperspherical energy (converges faster than LoRA)',
                'recommendations': 'Preserves hyperspherical energy. Converges faster than LoRA.'
            }
            # Note: norms not mentioned in official Algo-List.md
        }
        
    def _init_experimental_features(self) -> Dict[str, Dict[str, Any]]:
        """Initialize experimental feature configurations"""
        return {
            'fused_back_pass': {
                'enabled': False,
                'description': 'OneTrainer memory optimization',
                'requirements': ['compatible_optimizer', 'no_gradient_accumulation']
            },
            'eq_vae': {
                'enabled': False,
                'description': 'HakuLatent Equivariance Regularization',
                'status': 'future_implementation'
            },
            'adversarial_loss': {
                'enabled': False,
                'description': 'GAN-style training improvements', 
                'status': 'research_grade'
            },
            'multi_resolution': {
                'enabled': False,
                'description': 'Train on multiple resolutions simultaneously',
                'status': 'experimental'
            }
        }

    def _create_dataset_toml(self, config):
        dataset_config = {
            "general": {
                "resolution": config['resolution'],
                "shuffle_caption": config.get('shuffle_caption', True),
                "keep_tokens": config.get('keep_tokens', 0),
                "flip_aug": config['flip_aug'],
                "caption_extension": ".txt",
                "enable_bucket": config.get('enable_bucket', True),
                "bucket_no_upscale": config.get('bucket_no_upscale', False),
                "bucket_reso_steps": config.get('bucket_reso_steps', 64),
                "min_bucket_reso": config.get('min_bucket_reso', 256),
                "max_bucket_reso": config.get('max_bucket_reso', 2048),
                "caption_dropout_rate": config.get('caption_dropout_rate', 0.0),
                "caption_tag_dropout_rate": config.get('caption_tag_dropout_rate', 0.0),
            },
            "datasets": [
                {
                    "subsets": [
                        {
                            "num_repeats": config['num_repeats'],
                            "image_dir": os.path.abspath(config['dataset_dir']),
                        }
                    ]
                }
            ]
        }
        dataset_toml_path = os.path.join(self.config_dir, "dataset.toml")
        with open(dataset_toml_path, "w") as f:
            toml.dump(dataset_config, f)
        return dataset_toml_path

    def _create_config_toml(self, config):
        """🔧 Enhanced config generation with hybrid features"""
        
        # 🦄 LyCORIS Method Selection
        network_args = []
        network_module = "networks.lora"  # Default
        
        # Check for advanced LyCORIS methods
        lycoris_method = config.get('lycoris_method', 'none')
        if lycoris_method != 'none' and lycoris_method in self.lycoris_methods:
            lycoris_config = self.lycoris_methods[lycoris_method]
            network_module = lycoris_config['network_module']
            network_args = lycoris_config['network_args'].copy()
            
            # Add official LyCORIS network arguments from Network-Args.md
            
            # Add preset configuration (official presets: full, attn-mlp, attn-only, etc.)
            preset = config.get('lycoris_preset', 'attn-mlp')  # Default per Guidelines.md
            if preset != 'full':  # full is default, no need to specify
                network_args.append(f"preset={preset}")
            
            # Add dimension args if applicable
            if 'conv_dim' in config and config['conv_dim'] > 0:
                network_args.append(f"conv_dim={config['conv_dim']}")
            if 'conv_alpha' in config and config['conv_alpha'] > 0:
                network_args.append(f"conv_alpha={config['conv_alpha']}")
                
            # Add dropout options if specified
            if config.get('lycoris_dropout', 0) > 0:
                network_args.append(f"dropout={config['lycoris_dropout']}")
            if config.get('lycoris_rank_dropout', 0) > 0:
                network_args.append(f"rank_dropout={config['lycoris_rank_dropout']}")
            if config.get('lycoris_module_dropout', 0) > 0:
                network_args.append(f"module_dropout={config['lycoris_module_dropout']}")
                
            # Add advanced options if enabled
            if config.get('lycoris_use_tucker', False):
                network_args.append("use_tucker=True")
            if config.get('lycoris_use_scalar', False):
                network_args.append("use_scalar=True")
            if config.get('lycoris_train_norm', False):
                network_args.append("train_norm=True")
            if config.get('lycoris_bypass_mode', False):
                network_args.append("bypass_mode=True")
                
            # Algorithm-specific parameters
            if lycoris_method == 'lokr' and config.get('lycoris_factor', 0) != 0:
                network_args.append(f"factor={config['lycoris_factor']}")
            
            # DoRA weight decomposition (can be used with most algorithms)
            if config.get('lycoris_dora_wd', False):
                network_args.append("dora_wd=True")
                
            print(f"🦄 Using LyCORIS method: {lycoris_method} - {lycoris_config['description']}")
            if preset != 'attn-mlp':
                print(f"📋 Preset: {preset}")
            if config.get('lycoris_dropout', 0) > 0:
                print(f"💧 Dropout: {config['lycoris_dropout']}")
        
        # Fallback to basic LoRA types (updated for latest Kohya-SS)
        elif config['lora_type'] == "LoRA":
            network_module = "networks.lora"  # Standard LoRA
            network_args = []
            # Add LoRA+ support if enabled
            if config.get('enable_loraplus', False):
                loraplus_ratio = config.get('loraplus_lr_ratio', 16)
                network_args.append(f"loraplus_lr_ratio={loraplus_ratio}")
                print(f"✨ Using LoRA+ with ratio {loraplus_ratio}")
        elif config['lora_type'] == "LoCon":
            network_module = "lycoris.kohya"
            network_args = [f"algo=locon"]  # Official docs show locon, not lora
            # Add conv dimensions if specified (LoCon supports conv layers)
            if config.get('conv_dim', 0) > 0:
                network_args.extend([f"conv_dim={config['conv_dim']}", f"conv_alpha={config.get('conv_alpha', 4)}"])
            print("🔧 Using LoCon - LoRA with Convolution support (official: dim≤64, alpha≤dim/2)")
        elif config['lora_type'] == "DoRA (Weight Decomposition)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=lora", f"dora_wd=True"]
            # Add conv dimensions for DoRA if specified
            if config.get('conv_dim', 0) > 0:
                network_args.extend([f"conv_dim={config['conv_dim']}", f"conv_alpha={config.get('conv_alpha', 4)}"])
            print("🎯 Using DoRA - expect 2-3x slower training but higher quality!")
        elif config['lora_type'] == "LoHa (Hadamard Product)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=loha"]
            # LoHa can use conv dimensions but they're optional
            if config.get('conv_dim', 0) > 0:
                network_args.extend([f"conv_dim={config['conv_dim']}", f"conv_alpha={config.get('conv_alpha', 4)}"])
        elif config['lora_type'] == "LoKr (Kronecker Product)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=lokr"]
            # LoKr can use conv dimensions but they're optional
            if config.get('conv_dim', 0) > 0:
                network_args.extend([f"conv_dim={config['conv_dim']}", f"conv_alpha={config.get('conv_alpha', 4)}"])
        elif config['lora_type'] == "(IA)³ (Few Parameters)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=ia3"]  # IA3 uses learnable scalar scaling, no conv dimensions needed
        elif config['lora_type'] == "DyLoRA":
            network_module = "lycoris.kohya"
            network_args = [f"algo=dylora"]
            # DyLoRA can use conv dimensions
            if config.get('conv_dim', 0) > 0:
                network_args.extend([f"conv_dim={config['conv_dim']}", f"conv_alpha={config.get('conv_alpha', 4)}"])
        elif config['lora_type'] == "BOFT (Butterfly Transform)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=boft"]
            # BOFT can use conv dimensions
            if config.get('conv_dim', 0) > 0:
                network_args.extend([f"conv_dim={config['conv_dim']}", f"conv_alpha={config.get('conv_alpha', 4)}"])
        elif config['lora_type'] == "GLoRA (Generalized LoRA)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=glora"]
            # GLoRA: Does NOT use conv_dim/conv_alpha - it extracts dimensions from original modules
            # Uses different weight structure (a1,a2,b1,b2) instead of lora_down/lora_up
            print("🌟 Using GLoRA - Generalized LoRA with a1/a2/b1/b2 weight structure!")
            print("ℹ️ Note: GLoRA automatically detects layer dimensions (no conv_dim/conv_alpha needed)")

        # 🚀 Advanced Optimizer Handling
        optimizer_args = []
        optimizer_type = config['optimizer']  # Default to widget selection
        lr_scheduler_type = None
        lr_scheduler_args = None
        loss_type = None
        
        # Check for advanced optimizer
        advanced_optimizer = config.get('advanced_optimizer', 'standard')
        if advanced_optimizer != 'standard' and advanced_optimizer in self.advanced_optimizers:
            adv_config = self.advanced_optimizers[advanced_optimizer]
            optimizer_type = adv_config['optimizer_type']
            optimizer_args = adv_config['args'].copy()
            
            # Auto-configure scheduler if specified
            if 'scheduler' in adv_config:
                lr_scheduler_type = adv_config['scheduler']
            if 'scheduler_args' in adv_config:
                lr_scheduler_args = adv_config['scheduler_args']
            if 'loss_type' in adv_config:
                loss_type = adv_config['loss_type']
            
            print(f"🚀 Using advanced optimizer: {advanced_optimizer} - {adv_config['description']}")
            
            # Check if CAME optimizer is actually available
            if advanced_optimizer == 'came':
                try:
                    import LoraEasyCustomOptimizer.came
                    print("✅ CAME optimizer verified and ready!")
                except ImportError as e:
                    print(f"❌ CAME optimizer not available: {e}")
                    print("🔄 Falling back to standard AdamW optimizer...")
                    optimizer_type = "AdamW"
                    optimizer_args = ["weight_decay=0.01", "betas=[0.9,0.999]"]
                    lr_scheduler_type = None
                    lr_scheduler_args = None
                    advanced_optimizer = 'standard'  # Reset to prevent other CAME-specific logic
            
            # Override learning rate if specified (and optimizer is still advanced)
            if advanced_optimizer != 'standard' and 'learning_rate_override' in adv_config:
                config['unet_lr'] = adv_config['learning_rate_override']
                config['text_encoder_lr'] = adv_config['learning_rate_override']
                print(f"📊 Learning rate auto-set to: {adv_config['learning_rate_override']}")
        
        # Standard optimizer configurations - use dictionary lookup for cleaner code
        elif config['optimizer'] in self.standard_optimizers:
            std_config = self.standard_optimizers[config['optimizer']]
            optimizer_type = std_config['optimizer_type']
            optimizer_args.extend(std_config['args'])
            
            # Handle special cases like Prodigy warmup
            if config['optimizer'] == 'Prodigy' and config.get('lr_warmup_ratio', 0) > 0:
                if 'warmup_args' in std_config:
                    optimizer_args.extend(std_config['warmup_args'])
            
            print(f"🔧 Using {config['optimizer']}: {std_config['description']}")
        
        # If no match, use the optimizer name as-is and let SD scripts handle it
        else:
            print(f"⚠️ Unknown optimizer '{config['optimizer']}' - using as-is")

        # Handle REX scheduler
        if config['lr_scheduler'] == "rex" and not lr_scheduler_type:
            try:
                import LoraEasyCustomOptimizer.RexAnnealingWarmRestarts
                lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
                lr_scheduler_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]
                print("✅ REX scheduler verified and ready!")
            except ImportError as e:
                print(f"❌ REX scheduler not available: {e}")
                print("🔄 Falling back to cosine scheduler...")
                lr_scheduler_type = None  # Use default cosine
                lr_scheduler_args = None

        training_args = {
            "lowram": True,
            "pretrained_model_name_or_path": os.path.abspath(config['model_path']),
            "max_train_epochs": config['epochs'],
            "train_batch_size": config['train_batch_size'],
            "mixed_precision": config['precision'],
            "save_precision": config['precision'],
            "save_every_n_epochs": config['save_every_n_epochs'],
            "save_last_n_epochs": config['keep_only_last_n_epochs'],
            "output_name": config['project_name'],
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "cache_latents": config['cache_latents'],
            "cache_latents_to_disk": config['cache_latents_to_disk'], 
            "cache_text_encoder_outputs": config['cache_text_encoder_outputs'],
            "min_snr_gamma": self._safe_float(config['min_snr_gamma']) if config['min_snr_gamma_enabled'] else None,
            "ip_noise_gamma": config['ip_noise_gamma'] if config['ip_noise_gamma_enabled'] else None,
            "multires_noise_iterations": config.get('multires_noise_iterations', 6) if config['multinoise'] else None,
            "multires_noise_discount": config.get('multires_noise_discount', 0.3) if config['multinoise'] else None,
            "xformers": config['cross_attention'] == "xformers",
            "sdpa": config['cross_attention'] == "sdpa",
            "log_with": "wandb" if config.get('wandb_key') else None,
            "wandb_api_key": config['wandb_key'] if config['wandb_key'] else None,
            # Advanced training options
            "noise_offset": self._safe_float(config.get('noise_offset', 0.0)) if self._safe_float(config.get('noise_offset', 0.0)) > 0 else None,
            "adaptive_noise_scale": self._safe_float(config.get('adaptive_noise_scale', 0.0)) if self._safe_float(config.get('adaptive_noise_scale', 0.0)) > 0 else None,
            "zero_terminal_snr": config.get('zero_terminal_snr', False),
            "clip_skip": config.get('clip_skip', 2),
            "vae_batch_size": config.get('vae_batch_size', 1),
            "no_half_vae": config.get('no_half_vae', False),
            # User-configurable training flags (no more hardcoding!)
            "gradient_checkpointing": config.get('gradient_checkpointing', True),
            "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 1),
            "max_grad_norm": float(config.get('max_grad_norm', 1.0)),
            "full_fp16": config.get('full_fp16', False),
            "random_crop": config.get('random_crop', False),
            "fp8_base": config.get('fp8_base', False),
        }

        # Add v_parameterization only if explicitly enabled
        if config.get('v_parameterization', False):
            training_args["v_parameterization"] = True
            print("✅ V-Parameterization enabled for v-pred models")

        # Apply experimental features
        training_args = self._apply_experimental_features(config, training_args)

        config_toml = {
            "network_arguments": {
                "unet_lr": config['unet_lr'],
                "text_encoder_lr": config['text_encoder_lr'],
                "network_dim": config['network_dim'],
                "network_alpha": config['network_alpha'],
                "network_module": network_module,
                "network_args": network_args,
                "network_weights": config['continue_from_lora'] if config['continue_from_lora'] else None,
            },
            "optimizer_arguments": {
                "learning_rate": config['unet_lr'],
                "lr_scheduler": config['lr_scheduler'],
                "lr_scheduler_num_cycles": config['lr_scheduler_number'] if config['lr_scheduler'] == "cosine_with_restarts" else None,
                "lr_warmup_steps": self._calculate_warmup_steps(config) if self._safe_float(config['lr_warmup_ratio']) > 0 else None, # Proper calculation based on actual training steps
                "optimizer_type": optimizer_type,
                "optimizer_args": optimizer_args if optimizer_args else None,
                "lr_scheduler_type": lr_scheduler_type,
                "lr_scheduler_args": lr_scheduler_args,
            },
            "training_arguments": training_args
        }
        config_toml_path = os.path.join(self.config_dir, "config.toml")
        with open(config_toml_path, "w") as f:
            toml.dump(config_toml, f)
        return config_toml_path

    def _validate_advanced_config(self, config) -> Dict[str, any]:
        """🛡️ Validate advanced configuration for compatibility issues (pure function)"""
        validation_result = {
            'warnings': [],
            'suggested_changes': {},
            'conflicts': []
        }
        
        # Check memory requirements and auto-adjust for VRAM constraints
        batch_size = config.get('train_batch_size', 1)
        resolution = config.get('resolution', 1024)
        
        # VRAM usage estimation (very rough)
        estimated_vram_gb = (batch_size * resolution * resolution) / (1024 * 1024 * 200)  # Rough estimate
        
        if estimated_vram_gb > 20:  # Likely too much for most cards
            new_batch_size = max(1, batch_size // 2)
            validation_result['warnings'].append(f"💾 High VRAM usage detected")
            validation_result['suggested_changes']['train_batch_size'] = new_batch_size
            
        # Auto-enable basic memory saving for large models (but respect user choices)
        if resolution >= 1024:
            validation_result['warnings'].append("💾 Large resolution detected - enabling basic memory optimizations")
            if not config.get('cache_latents'):
                validation_result['suggested_changes']['cache_latents'] = True
                validation_result['warnings'].append("ℹ️ Auto-enabled latent caching for memory savings")
            if not config.get('cache_latents_to_disk'):
                validation_result['suggested_changes']['cache_latents_to_disk'] = True
                validation_result['warnings'].append("ℹ️ Auto-enabled disk caching for memory savings")
        
        # Validate text encoder conflicts (but don't auto-fix)
        if config.get('cache_text_encoder_outputs') and config.get('shuffle_caption'):
            validation_result['conflicts'].append("🚨 CONFLICT: Cannot use caption shuffling with text encoder caching!")
            validation_result['conflicts'].append("💡 Please disable one of these options in the training widget")
            
        if config.get('cache_text_encoder_outputs') and config.get('text_encoder_lr', 0) > 0:
            validation_result['conflicts'].append("🚨 CONFLICT: Cannot cache text encoder while training it!")
            validation_result['conflicts'].append("💡 Set Text Encoder LR to 0 or disable text encoder caching")
        
        # Check fused back pass compatibility
        if config.get('fused_back_pass', False):
            if config.get('train_batch_size', 1) > 1:
                validation_result['warnings'].append("⚠️ Fused Back Pass requires gradient accumulation = 1 (batch size 1)")
                validation_result['suggested_changes']['train_batch_size'] = 1
                
            advanced_optimizer = config.get('advanced_optimizer', 'standard')
            if advanced_optimizer not in ['came', 'prodigy_plus', 'stable_adamw']:
                validation_result['warnings'].append("⚠️ Fused Back Pass may not be compatible with this optimizer")
        
        # Check DoRA training time warning
        if config.get('lycoris_method') == 'dora':
            validation_result['warnings'].append("🐌 DoRA training is 2-3x slower but higher quality - be patient!")
        
        # Check CAME + REX pairing
        advanced_optimizer = config.get('advanced_optimizer', 'standard') 
        if advanced_optimizer == 'came' and config.get('lr_scheduler') != 'rex':
            validation_result['warnings'].append("💡 CAME optimizer works best with REX scheduler - auto-switching")
            validation_result['suggested_changes']['lr_scheduler'] = 'rex'
        
        # Check Prodigy learning rate
        if advanced_optimizer == 'prodigy_plus':
            if config.get('unet_lr', 1.0) != 1.0:
                validation_result['warnings'].append("📊 Prodigy Plus is learning rate free - setting LR to 1.0")
                validation_result['suggested_changes']['unet_lr'] = 1.0
                validation_result['suggested_changes']['text_encoder_lr'] = 1.0
        
        return validation_result
    
    def _apply_experimental_features(self, config, training_args: Dict[str, Any]) -> Dict[str, Any]:
        """🔬 Apply experimental features to training arguments"""
        
        # Fused Back Pass (OneTrainer) - Currently disabled, requires OneTrainer backend
        if config.get('fused_back_pass', False):
            print("⚠️ Fused Back Pass requires OneTrainer integration - feature disabled")
            print("💡 Using standard gradient checkpointing for VRAM optimization instead")
        
        # ⚠️ IMPORTANT: DO NOT HARDCODE TRAINING ARGUMENTS HERE!
        # All training settings must come from user widgets, not manager assumptions.
        # If you hardcode values here, users can't control their own training!
        
        # Only add features that are explicitly enabled by the user
        # No auto-enabling of experimental features
        
        return training_args

    def launch_from_files(self, config_paths, monitor_widget=None):
        """🚀 Launch training from existing TOML files - DEAD SIMPLE APPROACH"""
        
        print("🔍 FRANKENSTEIN TRAINING MANAGER - FILE HUNTING MODE! 💥")
        print(f"📁 Using config files: {list(config_paths.keys())}")
        
        # Extract the file paths
        config_toml_path = config_paths.get("config.toml")
        dataset_toml_path = config_paths.get("dataset.toml") 
        
        if not config_toml_path or not dataset_toml_path:
            print("❌ Missing required TOML files!")
            return False
        
        print(f"✅ Config TOML: {config_toml_path}")
        print(f"✅ Dataset TOML: {dataset_toml_path}")
        
        # Find the training script
        train_script = self._find_training_script()
        if not train_script:
            print("❌ No training script found!")
            return False
        
        # Check for venv python
        venv_python_path = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        if os.path.exists(venv_python_path):
            venv_python = venv_python_path
        else:
            venv_python = "python"
        
        print("\n🚀 Starting training from TOML files...")
        
        # Set memory optimization environment variables
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['CUDA_LAUNCH_BLOCKING'] = '1'
        
        try:
            # Set up training monitoring
            if monitor_widget:
                monitor_widget.clear_log()
                monitor_widget.update_phase("Launching training from files...", "info")
            
            # Change to scripts directory for proper LyCORIS imports (like Holo's approach)
            original_cwd = os.getcwd()
            scripts_dir = self.sd_scripts_dir if os.path.exists(self.sd_scripts_dir) else self.trainer_dir
            os.chdir(scripts_dir)
            
            process = subprocess.Popen(
                [venv_python, train_script,
                 "--config_file", config_toml_path,
                 "--dataset_config", dataset_toml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                
                # Update monitor widget if provided
                if monitor_widget:
                    monitor_widget.parse_training_output(line)
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, [venv_python, train_script])

            print("\n✅ TRAINING COMPLETE! 🎉")
            return True

        except subprocess.CalledProcessError as e:
            print(f"💥 Training error occurred: {e}")
            return False
        except Exception as e:
            print(f"🚨 Unexpected error: {e}")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def prepare_config_only(self, config):
        """🛠️ Generate TOML files without starting training"""
        print("📝 Generating configuration files...")
        
        dataset_toml_path = self._create_dataset_toml(config)
        config_toml_path = self._create_config_toml(config)
        
        print(f"✅ Generated: {dataset_toml_path}")
        print(f"✅ Generated: {config_toml_path}")
        
        return {"config.toml": config_toml_path, "dataset.toml": dataset_toml_path}

    def start_training(self, config, monitor_widget=None):
        """🚀 Start hybrid training with all advanced features"""
        
        print("🧪 FRANKENSTEIN TRAINING MANAGER ACTIVATED! 💥")
        print("Preparing hybrid training configuration...")
        
        # Validate advanced configuration (pure function - no side effects)
        validation = self._validate_advanced_config(config)
        
        # Apply suggested changes to config
        for key, value in validation['suggested_changes'].items():
            config[key] = value
            
        # Print all validation results
        for warning in validation['warnings']:
            print(warning)
        for conflict in validation['conflicts']:
            print(conflict)
        
        # Show advanced features summary
        if config.get('advanced_mode_enabled', False):
            self._print_advanced_features_summary(config)
        
        dataset_toml_path = self._create_dataset_toml(config)
        config_toml_path = self._create_config_toml(config)

        # Check for venv python first, fall back to system python
        venv_python_path = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        if os.path.exists(venv_python_path):
            venv_python = venv_python_path
        else:
            venv_python = "python"  # Use system python (common in containers)
        
        # Find the training script in flexible locations
        train_script = self._find_training_script()
        if not train_script:
            print("❌ No training script found in any expected location!")
            print("💡 Please ensure the environment setup completed successfully!")
            return

        print("\n🚀 Starting hybrid training...")
        
        # Set memory optimization environment variables
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce fragmentation
        env['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
        print("💾 Memory optimization: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        try:
            # Set up training monitoring
            if monitor_widget:
                monitor_widget.clear_log()
                monitor_widget.update_phase("Initializing training process...", "info")
                # Set total epochs for progress tracking
                monitor_widget.total_epochs = config['epochs']
            
            # Change to scripts directory for proper LyCORIS imports (like Holo's approach)
            original_cwd = os.getcwd()
            scripts_dir = self.sd_scripts_dir if os.path.exists(self.sd_scripts_dir) else self.trainer_dir
            os.chdir(scripts_dir)
            
            process = subprocess.Popen(
                [venv_python, train_script,
                 "--config_file", config_toml_path,
                 "--dataset_config", dataset_toml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Still print to console
                
                # Update monitor widget if provided
                if monitor_widget:
                    monitor_widget.parse_training_output(line)
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, [venv_python, train_script])

            print("\n✅ FRANKENSTEIN TRAINING COMPLETE! 🎉")
            print("Either it worked perfectly or it blew up spectacularly! 😄")

        except subprocess.CalledProcessError as e:
            print(f"💥 Training error occurred: {e}")
            print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
            if "CAME" in str(e) or "LoraEasyCustomOptimizer" in str(e):
                print("1. 🧪 CAME/REX optimizer issue - custom optimizers may not be installed")
                print("2. 🔄 Try disabling Advanced Mode and using standard AdamW optimizer")
                print("3. 📦 Run environment setup again to install custom optimizers")
                print("4. 🚀 Switch to Prodigy optimizer as alternative to CAME")
            elif "bitsandbytes" in str(e) or "triton" in str(e):
                print("1. 📊 Try switching from AdamW8bit to AdamW (compatibility issues)")
                print("2. 🔧 bitsandbytes/triton version conflict detected")
                print("3. 🚀 Use standard optimizers: AdamW, Prodigy, Lion")
            else:
                print("1. 💾 Reduce batch size if getting CUDA out of memory")
                print("2. 🎯 Check that model path and dataset directory exist")
                print("3. 📊 Try different optimizer (AdamW, Prodigy)")
                print("4. 🔧 Check file paths and permissions")
            print("5. 📝 Check the detailed error log above for specific issues")
            return False
        except Exception as e:
            print(f"🚨 Unexpected error: {e}")
            print("🧪 Welcome to the bleeding edge - backup and try again!")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def _print_advanced_features_summary(self, config):
        """📊 Print summary of active advanced features"""
        print("\n🔬 ADVANCED FEATURES ACTIVE:")
        
        advanced_optimizer = config.get('advanced_optimizer', 'standard')
        if advanced_optimizer != 'standard':
            print(f"  🚀 Optimizer: {advanced_optimizer}")
        
        lycoris_method = config.get('lycoris_method', 'none')
        if lycoris_method != 'none':
            print(f"  🦄 LyCORIS Method: {lycoris_method}")
        
        if config.get('fused_back_pass', False):
            print(f"  ⚡ Fused Back Pass: Enabled (VRAM optimization)")
        
        experimental_features = config.get('experimental_features', {})
        active_experiments = [k for k, v in experimental_features.items() if v]
        if active_experiments:
            print(f"  🔬 Experimental: {len(active_experiments)} features enabled")
        
        print("  💀 No guarantees - you asked for this! 😈\n")


# Backwards compatibility - alias the old class name
TrainingManager = HybridTrainingManager