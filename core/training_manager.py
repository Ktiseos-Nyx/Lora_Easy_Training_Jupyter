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
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "sd_scripts")
        self.output_dir = os.path.join(self.project_root, "output")
        self.logging_dir = os.path.join(self.project_root, "logs")
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # 🔧 Setup Python path for Derrian's custom optimizers
        self._setup_custom_optimizers()
        
        # 🔬 Advanced feature support
        self.advanced_optimizers = self._init_advanced_optimizers()
        self.lycoris_methods = self._init_lycoris_methods()
        self.experimental_features = self._init_experimental_features()
    
    
    
    def _find_training_script(self):
        """Find the training script in various possible locations"""
        possible_scripts = [
            # Derrian's backend might have scripts in different locations
            os.path.join(self.trainer_dir, "sd_scripts", "sdxl_train_network.py"),
            os.path.join(self.trainer_dir, "sd_scripts", "train_network.py"), 
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
        """Add Derrian's backend to Python path so we can import custom optimizers"""
        if os.path.exists(self.trainer_dir) and self.trainer_dir not in sys.path:
            sys.path.insert(0, self.trainer_dir)
            print(f"🔧 Added {self.trainer_dir} to Python path for custom optimizers")
            
        # Also add custom_scheduler directory to path
        custom_scheduler_dir = os.path.join(self.trainer_dir, "custom_scheduler")
        if os.path.exists(custom_scheduler_dir) and custom_scheduler_dir not in sys.path:
            sys.path.insert(0, custom_scheduler_dir)
            print(f"🔧 Added {custom_scheduler_dir} to Python path for CAME/REX optimizers")
            
        # Debug: Check what's actually in the trainer directory
        print("🔍 Checking trainer directory contents...")
        try:
            contents = os.listdir(self.trainer_dir)
            print(f"Found: {contents}")
            
            # Look for optimizer-related files/folders
            optimizer_files = [f for f in contents if 'optim' in f.lower() or 'came' in f.lower() or 'rex' in f.lower() or 'custom_scheduler' in f.lower()]
            if optimizer_files:
                print(f"📁 Optimizer-related files: {optimizer_files}")
                
            # Check custom_scheduler directory specifically
            custom_scheduler_dir = os.path.join(self.trainer_dir, "custom_scheduler")
            if os.path.exists(custom_scheduler_dir):
                try:
                    scheduler_contents = os.listdir(custom_scheduler_dir)
                    print(f"📁 custom_scheduler contents: {scheduler_contents}")
                except Exception as e:
                    print(f"⚠️ Error reading custom_scheduler: {e}")
            
            # Check for Python files
            py_files = [f for f in contents if f.endswith('.py')]
            if py_files:
                print(f"🐍 Python files: {py_files}")
                
        except Exception as e:
            print(f"❌ Error listing directory: {e}")
            
        # Try to import and verify CAME is available (from custom_scheduler)
        try:
            import LoraEasyCustomOptimizer.came
            print("✅ CAME optimizer found and ready!")
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
    
    def _init_lycoris_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize LyCORIS method configurations"""
        return {
            'dora': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=dora', 'use_tucker=False'],
                'description': 'Weight Decomposition - trains like full fine-tune'
            },
            'lokr': {
                'network_module': 'lycoris.kohya', 
                'network_args': ['algo=lokr', 'factor=-1'],
                'description': 'Kronecker Product - mathematically efficient'
            },
            'loha': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=loha'],
                'description': 'Hadamard Product - good balance'
            },
            'ia3': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=ia3'],
                'description': 'Implicit Attention - parameter efficient'
            },
            'boft': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=boft'],
                'description': 'Butterfly Transform - structured adaptation'
            },
            'glora': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=glora'],
                'description': 'Generalized LoRA - flexible adaptation'
            }
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
                "enable_bucket": True,
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
                            "image_dir": config['dataset_dir'],
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
        network_args = None
        network_module = "networks.lora"  # Default
        
        # Check for advanced LyCORIS methods
        lycoris_method = config.get('lycoris_method', 'none')
        if lycoris_method != 'none' and lycoris_method in self.lycoris_methods:
            lycoris_config = self.lycoris_methods[lycoris_method]
            network_module = lycoris_config['network_module']
            network_args = lycoris_config['network_args'].copy()
            
            # Add dimension args if applicable
            if 'conv_dim' in config:
                network_args.append(f"conv_dim={config['conv_dim']}")
            if 'conv_alpha' in config:
                network_args.append(f"conv_alpha={config['conv_alpha']}")
                
            print(f"🦄 Using LyCORIS method: {lycoris_method} - {lycoris_config['description']}")
        
        # Fallback to basic LoRA types
        elif config['lora_type'] == "LoCon":
            network_module = "lycoris.kohya"
            network_args = [f"algo=locon", f"conv_dim={config['conv_dim']}", f"conv_alpha={config['conv_alpha']}"]
        elif config['lora_type'] == "LoKR":
            network_module = "lycoris.kohya"
            network_args = [f"algo=lokr", f"conv_dim={config['conv_dim']}", f"conv_alpha={config['conv_alpha']}"]
        elif config['lora_type'] == "DyLoRA":
            network_module = "lycoris.kohya"
            network_args = [f"algo=dylora", f"conv_dim={config['conv_dim']}", f"conv_alpha={config['conv_alpha']}"]
        elif config['lora_type'] == "DoRA (Weight Decomposition)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=dora", f"conv_dim={config['conv_dim']}", f"conv_alpha={config['conv_alpha']}"]
            print("🎯 Using DoRA - expect 2-3x slower training but higher quality!")
        elif config['lora_type'] == "LoHa (Hadamard Product)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=loha", f"conv_dim={config['conv_dim']}", f"conv_alpha={config['conv_alpha']}"]
        elif config['lora_type'] == "(IA)³ (Few Parameters)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=ia3"]  # IA3 doesn't use conv dimensions
        elif config['lora_type'] == "GLoRA (Generalized LoRA)":
            network_module = "lycoris.kohya"
            network_args = [f"algo=glora", f"conv_dim={config['conv_dim']}", f"conv_alpha={config['conv_alpha']}"]

        # 🚀 Advanced Optimizer Handling
        optimizer_args = []
        optimizer_type = config['optimizer']
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
        
        # Standard optimizer configurations - ensure all use correct module paths
        elif config['optimizer'] == "Prodigy":
            # Prodigy is usually in torch.optim or external package
            optimizer_args.extend(["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True"])
            if config['lr_warmup_ratio'] > 0:
                optimizer_args.append("safeguard_warmup=True")
        elif config['optimizer'] == "AdamW8bit":
            # AdamW8bit is in bitsandbytes
            optimizer_type = "bitsandbytes.optim.AdamW8bit"
            optimizer_args.extend(["weight_decay=0.1", "betas=[0.9,0.99]"])
        elif config['optimizer'] == "AdaFactor":
            # AdaFactor is in transformers
            optimizer_type = "transformers.optimization.Adafactor"
            optimizer_args.extend(["scale_parameter=False", "relative_step=False", "warmup_init=False"])
        elif config['optimizer'] == "Came":
            optimizer_type = "LoraEasyCustomOptimizer.came.CAME"
            optimizer_args.extend(["weight_decay=0.04"])
        elif config['optimizer'] == "DAdaptation":
            # DAdaptation is external package
            pass  # Let SD scripts handle the import
        elif config['optimizer'] == "DadaptAdam":
            # DAdaptAdam is in dadaptation package
            pass  # Let SD scripts handle the import
        elif config['optimizer'] == "DadaptLion":
            # DadaptLion is in dadaptation package  
            pass  # Let SD scripts handle the import
        # AdamW, Lion, SGDNesterov, SGDNesterov8bit are in torch.optim - use defaults

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
            "pretrained_model_name_or_path": config['model_path'],
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
            "min_snr_gamma": config['min_snr_gamma'] if config['min_snr_gamma_enabled'] else None,
            "ip_noise_gamma": config['ip_noise_gamma'] if config['ip_noise_gamma_enabled'] else None,
            "multires_noise_iterations": 6 if config['multinoise'] else None, # Default value from sample notebook
            "multires_noise_discount": 0.3 if config['multinoise'] else None, # Default value from sample notebook
            "xformers": True if config['cross_attention'] == "xformers" else None,
            "sdpa": True if config['cross_attention'] == "sdpa" else None,
            "log_with": "wandb" if config['wandb_key'] else None,
            "wandb_api_key": config['wandb_key'] if config['wandb_key'] else None,
            # Advanced training options
            "noise_offset": config.get('noise_offset', 0.0) if config.get('noise_offset', 0.0) > 0 else None,
            "adaptive_noise_scale": config.get('adaptive_noise_scale', 0.0) if config.get('adaptive_noise_scale', 0.0) > 0 else None,
            "zero_terminal_snr": config.get('zero_terminal_snr', False) if config.get('zero_terminal_snr', False) else None,
            "clip_skip": config.get('clip_skip', 2),
            "vae_batch_size": config.get('vae_batch_size', 1),
            "no_half_vae": config.get('no_half_vae', False) if config.get('no_half_vae', False) else None,
            # Memory optimization flags
            "gradient_checkpointing": True,  # Always enable for memory savings
            "gradient_accumulation_steps": max(1, 4 // config['train_batch_size']),  # Auto-adjust for small batch sizes
            "max_grad_norm": 1.0,  # Gradient clipping for stability
            "full_fp16": config['precision'] == 'fp16',  # More aggressive FP16 if selected
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
                "lr_warmup_steps": int(config['lr_warmup_ratio'] * config['epochs'] * 100) if config['lr_warmup_ratio'] > 0 else None, # Simplified calculation
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

    def _validate_advanced_config(self, config) -> List[str]:
        """🛡️ Validate advanced configuration for compatibility issues"""
        warnings = []
        
        # Check memory requirements and auto-adjust for VRAM constraints
        batch_size = config.get('train_batch_size', 1)
        resolution = config.get('resolution', 1024)
        
        # VRAM usage estimation (very rough)
        estimated_vram_gb = (batch_size * resolution * resolution) / (1024 * 1024 * 200)  # Rough estimate
        
        if estimated_vram_gb > 20:  # Likely too much for most cards
            new_batch_size = max(1, batch_size // 2)
            warnings.append(f"💾 High VRAM usage detected - reducing batch size from {batch_size} to {new_batch_size}")
            config['train_batch_size'] = new_batch_size
            
        # Auto-enable basic memory saving for large models (but respect user choices)
        if resolution >= 1024:
            warnings.append("💾 Large resolution detected - enabling basic memory optimizations")
            if not config.get('cache_latents'):
                config['cache_latents'] = True
                warnings.append("ℹ️ Auto-enabled latent caching for memory savings")
            if not config.get('cache_latents_to_disk'):
                config['cache_latents_to_disk'] = True
                warnings.append("ℹ️ Auto-enabled disk caching for memory savings")
        
        # Validate text encoder conflicts (but don't auto-fix)
        if config.get('cache_text_encoder_outputs') and config.get('shuffle_caption'):
            warnings.append("🚨 CONFLICT: Cannot use caption shuffling with text encoder caching!")
            warnings.append("💡 Please disable one of these options in the training widget")
            
        if config.get('cache_text_encoder_outputs') and config.get('text_encoder_lr', 0) > 0:
            warnings.append("🚨 CONFLICT: Cannot cache text encoder while training it!")
            warnings.append("💡 Set Text Encoder LR to 0 or disable text encoder caching")
        
        # Check fused back pass compatibility
        if config.get('fused_back_pass', False):
            if config.get('train_batch_size', 1) > 1:
                warnings.append("⚠️ Fused Back Pass requires gradient accumulation = 1 (batch size 1)")
                config['train_batch_size'] = 1
                
            advanced_optimizer = config.get('advanced_optimizer', 'standard')
            if advanced_optimizer not in ['came', 'prodigy_plus', 'stable_adamw']:
                warnings.append("⚠️ Fused Back Pass may not be compatible with this optimizer")
        
        # Check DoRA training time warning
        if config.get('lycoris_method') == 'dora':
            warnings.append("🐌 DoRA training is 2-3x slower but higher quality - be patient!")
        
        # Check CAME + REX pairing
        advanced_optimizer = config.get('advanced_optimizer', 'standard') 
        if advanced_optimizer == 'came' and config.get('lr_scheduler') != 'rex':
            warnings.append("💡 CAME optimizer works best with REX scheduler - auto-switching")
            config['lr_scheduler'] = 'rex'
        
        # Check Prodigy learning rate
        if advanced_optimizer == 'prodigy_plus':
            if config.get('unet_lr', 1.0) != 1.0:
                warnings.append("📊 Prodigy Plus is learning rate free - setting LR to 1.0")
                config['unet_lr'] = 1.0
                config['text_encoder_lr'] = 1.0
        
        return warnings
    
    def _apply_experimental_features(self, config, training_args: Dict[str, Any]) -> Dict[str, Any]:
        """🔬 Apply experimental features to training arguments"""
        
        # Fused Back Pass (OneTrainer) - Currently disabled, requires OneTrainer backend
        if config.get('fused_back_pass', False):
            print("⚠️ Fused Back Pass requires OneTrainer integration - feature disabled")
            print("💡 Using standard gradient checkpointing for VRAM optimization instead")
        
        # Aggressive Gradient Checkpointing
        if config.get('gradient_checkpointing', False):
            training_args['gradient_checkpointing'] = True
            training_args['deepspeed_zero_offload'] = True
            print("💾 Aggressive gradient checkpointing enabled")
        
        # Multi-Resolution Training
        experimental_features = config.get('experimental_features', {})
        if experimental_features.get('experimental_2', False):  # Multi-res checkbox
            training_args['enable_bucket'] = True
            training_args['bucket_reso_steps'] = 32  # More granular
            training_args['random_crop'] = True
            print("🌊 Multi-resolution training enabled")
        
        return training_args

    def start_training(self, config, monitor_widget=None):
        """🚀 Start hybrid training with all advanced features"""
        
        print("🧪 FRANKENSTEIN TRAINING MANAGER ACTIVATED! 💥")
        print("Preparing hybrid training configuration...")
        
        # Validate advanced configuration
        warnings = self._validate_advanced_config(config)
        for warning in warnings:
            print(warning)
        
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
            
            process = subprocess.Popen(
                [venv_python, train_script,
                 "--config_file", config_toml_path,
                 "--dataset_config", dataset_toml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.project_root,
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