# core/training_manager.py
import subprocess
import os
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
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.runtime_store_dir = os.path.join(self.trainer_dir, "runtime_store")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "sd_scripts")
        self.output_dir = os.path.join(self.project_root, "output")
        self.logging_dir = os.path.join(self.project_root, "logs")
        os.makedirs(self.runtime_store_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # 🔬 Advanced feature support
        self.advanced_optimizers = self._init_advanced_optimizers()
        self.lycoris_methods = self._init_lycoris_methods()
        self.experimental_features = self._init_experimental_features()
        
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
                "shuffle_caption": True,
                "keep_tokens": 0,
                "flip_aug": config['flip_aug'],
                "caption_extension": ".txt",
                "enable_bucket": True,
                "bucket_no_upscale": False,
                "bucket_reso_steps": 64,
                "min_bucket_reso": 256,
                "max_bucket_reso": 2048,
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
        dataset_toml_path = os.path.join(self.runtime_store_dir, "dataset.toml")
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
            
            # Override learning rate if specified
            if 'learning_rate_override' in adv_config:
                config['unet_lr'] = adv_config['learning_rate_override']
                config['text_encoder_lr'] = adv_config['learning_rate_override']
                print(f"📊 Learning rate auto-set to: {adv_config['learning_rate_override']}")
        
        # Standard optimizer configurations
        elif config['optimizer'] == "Prodigy":
            optimizer_args.extend(["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True"])
            if config['lr_warmup_ratio'] > 0:
                optimizer_args.append("safeguard_warmup=True")
        elif config['optimizer'] == "AdamW8bit":
            optimizer_args.extend(["weight_decay=0.1", "betas=[0.9,0.99]"])
        elif config['optimizer'] == "AdaFactor":
            optimizer_args.extend(["scale_parameter=False", "relative_step=False", "warmup_init=False"])
        elif config['optimizer'] == "Came":
            optimizer_args.extend(["weight_decay=0.04"])

        # Handle REX scheduler
        if config['lr_scheduler'] == "rex" and not lr_scheduler_type:
            lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
            lr_scheduler_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]

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
                "optimizer_type": config['optimizer'],
                "optimizer_args": optimizer_args if optimizer_args else None,
                "lr_scheduler_type": lr_scheduler_type,
                "lr_scheduler_args": lr_scheduler_args,
            },
            "training_arguments": {
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
                "v_parameterization": config['v_parameterization'] if config['v_parameterization'] else None,
                "min_snr_gamma": config['min_snr_gamma'] if config['min_snr_gamma_enabled'] else None,
                "ip_noise_gamma": config['ip_noise_gamma'] if config['ip_noise_gamma_enabled'] else None,
                "multires_noise_iterations": 6 if config['multinoise'] else None, # Default value from sample notebook
                "multires_noise_discount": 0.3 if config['multinoise'] else None, # Default value from sample notebook
                "xformers": True if config['cross_attention'] == "xformers" else None,
                "sdpa": True if config['cross_attention'] == "sdpa" else None,
                "log_with": "wandb" if config['wandb_key'] else None,
                "wandb_api_key": config['wandb_key'] if config['wandb_key'] else None,
            }
        }
        config_toml_path = os.path.join(self.runtime_store_dir, "config.toml")
        with open(config_toml_path, "w") as f:
            toml.dump(config_toml, f)
        return config_toml_path

    def _validate_advanced_config(self, config) -> List[str]:
        """🛡️ Validate advanced configuration for compatibility issues"""
        warnings = []
        
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
        
        # Fused Back Pass (OneTrainer)
        if config.get('fused_back_pass', False):
            training_args['fused_backward_pass'] = True
            training_args['gradient_accumulation_steps'] = 1
            print("⚡ Fused Back Pass enabled - VRAM optimization active")
        
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

    def start_training(self, config):
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

        venv_python = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        train_script = os.path.join(self.sd_scripts_dir, "sdxl_train_network.py") # Assuming SDXL for now

        print("\n🚀 Starting hybrid training...")
        try:
            process = subprocess.Popen(
                [venv_python, train_script,
                 "--config_file", config_toml_path,
                 "--dataset_config", dataset_toml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.project_root
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, [venv_python, train_script])

            print("\n✅ FRANKENSTEIN TRAINING COMPLETE! 🎉")
            print("Either it worked perfectly or it blew up spectacularly! 😄")

        except subprocess.CalledProcessError as e:
            print(f"💥 Training error occurred: {e}")
            print("🔧 This is experimental - try adjusting settings or using standard mode")
        except Exception as e:
            print(f"🚨 Unexpected error: {e}")
            print("🧪 Welcome to the bleeding edge - backup and try again!")
    
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