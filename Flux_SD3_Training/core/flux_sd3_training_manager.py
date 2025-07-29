# Flux_SD3_Training/core/flux_sd3_training_manager.py
import os
import sys
import toml
import subprocess
from typing import Dict, List, Optional, Any

class FluxSD3LoRAManager:
    """🚀 Flux/SD3 LoRA Training Manager
    
    Train LoRA adapters for next-generation diffusion models:
    - Flux.1 (Black Forest Labs)
    - SD3/SD3.5 (Stability AI) 
    - AuraFlow (Fal AI)
    - HiDream/HunyuanDiT (Tencent)
    
    Uses Kohya's flux_train_network.py and sd3_train_network.py scripts
    with memory optimizations for consumer GPUs.
    """
    
    def __init__(self):
        # Project structure
        self.project_root = os.getcwd()
        self.flux_sd3_dir = os.path.join(self.project_root, "Flux_SD3_Training")
        self.output_dir = os.path.join(self.flux_sd3_dir, "output")
        self.config_dir = os.path.join(self.flux_sd3_dir, "configs") 
        self.logging_dir = os.path.join(self.flux_sd3_dir, "logs")
        
        # Kohya scripts integration
        self.sd_scripts_dir = os.path.join(self.project_root, "trainer", "derrian_backend", "sd_scripts")
        
        # Create directories
        for dir_path in [self.output_dir, self.config_dir, self.logging_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize configurations
        self.model_configs = self._init_model_configs()
        self.memory_profiles = self._init_memory_profiles()
        self.training_presets = self._init_training_presets()
        
        print("🚀 Flux/SD3 LoRA Training Manager initialized!")
        print(f"📁 Workspace: {self.flux_sd3_dir}")
    
    def _init_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model configurations for Flux and SD3 variants"""
        return {
            'flux_dev': {
                'model_name': 'black-forest-labs/FLUX.1-dev',
                'script': 'flux_train_network.py',
                'resolution': 1024,
                'text_encoders': ['clip_l', 't5xxl'],
                'memory_base': 12,  # GB VRAM baseline
                'description': 'Flux.1 Dev - High quality diffusion model'
            },
            'flux_schnell': {
                'model_name': 'black-forest-labs/FLUX.1-schnell', 
                'script': 'flux_train_network.py',
                'resolution': 1024,
                'text_encoders': ['clip_l', 't5xxl'],
                'memory_base': 10,  # Slightly more efficient
                'description': 'Flux.1 Schnell - Fast inference variant'
            },
            'sd3_medium': {
                'model_name': 'stabilityai/stable-diffusion-3-medium',
                'script': 'sd3_train_network.py', 
                'resolution': 1024,
                'text_encoders': ['clip_l', 'clip_g', 't5xxl'],
                'memory_base': 14,  # SD3 is memory hungry
                'description': 'SD3 Medium - Stability AI latest'
            },
            'sd3_large': {
                'model_name': 'stabilityai/stable-diffusion-3-large',
                'script': 'sd3_train_network.py',
                'resolution': 1024, 
                'text_encoders': ['clip_l', 'clip_g', 't5xxl'],
                'memory_base': 18,  # Even more memory hungry
                'description': 'SD3 Large - Maximum quality'
            },
            'auraflow': {
                'model_name': 'fal/AuraFlow',
                'script': 'flux_train_network.py',  # Uses Flux-style training
                'resolution': 1024,
                'text_encoders': ['t5xxl'],
                'memory_base': 10,
                'description': 'AuraFlow - Community Flux variant'
            }
        }
    
    def _init_memory_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Memory optimization profiles for Flux/SD3 training"""
        return {
            'ultra_low_memory': {  # 8-12GB VRAM
                'train_batch_size': 1,
                'gradient_accumulation_steps': 32,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': True,
                'optimizer': 'AdamW8bit',
                'cache_latents': True,
                'cache_latents_to_disk': True,
                'lowram': True,
                'blocks_to_swap': 8,  # Flux/SD3 specific
                'description': 'Maximum memory savings (8-12GB VRAM)'
            },
            'low_memory': {  # 12-16GB VRAM  
                'train_batch_size': 1,
                'gradient_accumulation_steps': 16,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': True,
                'optimizer': 'AdamW8bit',
                'cache_latents': True,
                'cache_latents_to_disk': False,
                'lowram': False,
                'blocks_to_swap': 4,
                'description': 'Balanced efficiency (12-16GB VRAM)'
            },
            'standard': {  # 16-20GB VRAM
                'train_batch_size': 2,
                'gradient_accumulation_steps': 8,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': False,
                'optimizer': 'AdamW',
                'cache_latents': True,
                'cache_latents_to_disk': False,
                'lowram': False,
                'blocks_to_swap': 0,
                'description': 'Standard training (16-20GB VRAM)'
            },
            'high_performance': {  # 24GB+ VRAM
                'train_batch_size': 4,
                'gradient_accumulation_steps': 4,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': False,
                'optimizer': 'AdamW',
                'cache_latents': False,  # Keep in memory
                'cache_latents_to_disk': False,
                'lowram': False,
                'blocks_to_swap': 0,
                'description': 'High performance (24GB+ VRAM)'
            }
        }
    
    def _init_training_presets(self) -> Dict[str, Dict[str, Any]]:
        """Training presets optimized for Flux/SD3 models"""
        return {
            'concept_learning': {
                'learning_rate': 1e-4,
                'text_encoder_lr': 5e-5,
                'epochs': 10,
                'warmup_ratio': 0.1,
                'scheduler': 'cosine_with_restarts',
                'network_dim': 16,
                'network_alpha': 8,
                'description': 'Learn new concepts and objects'
            },
            'style_training': {
                'learning_rate': 8e-5,
                'text_encoder_lr': 4e-5,
                'epochs': 15,
                'warmup_ratio': 0.05,
                'scheduler': 'cosine',
                'network_dim': 32,
                'network_alpha': 16,
                'description': 'Artistic styles and aesthetics'
            },
            'character_training': {
                'learning_rate': 1.2e-4,
                'text_encoder_lr': 6e-5,
                'epochs': 12,
                'warmup_ratio': 0.1,
                'scheduler': 'constant_with_warmup',
                'network_dim': 24,
                'network_alpha': 12,
                'description': 'Characters and specific people'
            },
            'fine_detail': {
                'learning_rate': 6e-5,
                'text_encoder_lr': 3e-5,
                'epochs': 20,
                'warmup_ratio': 0.15,
                'scheduler': 'linear',
                'network_dim': 64,
                'network_alpha': 32,
                'description': 'Fine details and complex scenes'
            }
        }
    
    def detect_optimal_memory_profile(self, model_type: str) -> str:
        """Auto-detect optimal memory profile based on VRAM and model"""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                model_memory_base = self.model_configs[model_type]['memory_base']
                
                print(f"🔍 Detected {vram_gb:.1f}GB VRAM")
                print(f"🤖 {model_type} baseline requirement: {model_memory_base}GB")
                
                # Account for model-specific memory requirements
                available_for_training = vram_gb - model_memory_base
                
                if available_for_training >= 8:
                    return 'high_performance'
                elif available_for_training >= 4:
                    return 'standard'
                elif available_for_training >= 2:
                    return 'low_memory'
                else:
                    return 'ultra_low_memory'
            else:
                print("⚠️ CUDA not available")
                return 'ultra_low_memory'
        except Exception as e:
            print(f"⚠️ Could not detect VRAM: {e}")
            return 'low_memory'
    
    def create_training_config(self, config: Dict[str, Any]) -> str:
        """Create Flux/SD3 LoRA training configuration"""
        
        model_type = config['model_type']
        model_config = self.model_configs[model_type]
        
        # Auto-detect memory profile if not specified
        memory_profile = config.get('memory_profile', self.detect_optimal_memory_profile(model_type))
        memory_settings = self.memory_profiles[memory_profile]
        
        # Get training preset
        preset_name = config.get('training_preset', 'concept_learning')
        preset = self.training_presets[preset_name]
        
        # Build training configuration
        training_config = {
            'model_arguments': {
                'pretrained_model_name_or_path': model_config['model_name'],
                'clip_l': config.get('clip_l_path'),  # For Flux/SD3
                't5xxl': config.get('t5xxl_path'),    # T5 encoder path
                'cache_latents': memory_settings['cache_latents'],
                'cache_latents_to_disk': memory_settings['cache_latents_to_disk'],
                'lowram': memory_settings['lowram']
            },
            'network_arguments': {
                'network_dim': config.get('network_dim', preset['network_dim']),
                'network_alpha': config.get('network_alpha', preset['network_alpha']),
                'network_module': 'networks.lora',
                'network_train_unet_only': config.get('unet_only', False)
            },
            'training_arguments': {
                'output_dir': self.output_dir,
                'logging_dir': self.logging_dir,
                'output_name': config['project_name'],
                'resolution': f"{model_config['resolution']},{model_config['resolution']}",
                'train_batch_size': memory_settings['train_batch_size'],
                'gradient_accumulation_steps': memory_settings['gradient_accumulation_steps'],
                'max_train_epochs': config.get('epochs', preset['epochs']),
                'learning_rate': config.get('learning_rate', preset['learning_rate']),
                'text_encoder_lr': config.get('text_encoder_lr', preset['text_encoder_lr']),
                'lr_scheduler': preset['scheduler'],
                'lr_warmup_ratio': preset['warmup_ratio'],
                'optimizer_type': memory_settings['optimizer'],
                'mixed_precision': memory_settings['mixed_precision'],
                'gradient_checkpointing': memory_settings['gradient_checkpointing'],
                'save_precision': 'fp16',
                'save_model_as': 'safetensors',
                'save_every_n_epochs': config.get('save_every_n_epochs', 2),
                'keep_tokens': config.get('keep_tokens', 1),
                'min_snr_gamma': config.get('min_snr_gamma', 8.0),
                'seed': 42
            },
            'flux_sd3_specific': {
                'blocks_to_swap': memory_settings.get('blocks_to_swap', 0),
                'training_script': model_config['script'],
                'text_encoders': model_config['text_encoders']
            }
        }
        
        # Add SD3-specific settings
        if 'sd3' in model_type:
            training_config['model_arguments']['clip_g'] = config.get('clip_g_path')
        
        # Save config
        config_filename = f"{config['project_name']}_{model_type}_config.toml"
        config_path = os.path.join(self.config_dir, config_filename)
        
        with open(config_path, 'w') as f:
            f.write(toml.dumps(training_config))
        
        print(f"📄 Flux/SD3 LoRA config saved: {config_path}")
        print(f"🧠 Memory profile: {memory_profile} ({memory_settings['description']})")
        print(f"🤖 Model: {model_config['description']}")
        print(f"🎯 Training: {preset_name} ({preset['description']})")
        
        return config_path
    
    def get_training_command(self, config_path: str, dataset_path: str) -> List[str]:
        """Generate Kohya training command for Flux/SD3"""
        
        # Load config to get script type
        with open(config_path, 'r') as f:
            config = toml.load(f)
        
        script_name = config['flux_sd3_specific']['training_script']
        script_path = os.path.join(self.sd_scripts_dir, script_name)
        
        # Base command
        cmd = ['python', script_path]
        
        # Add standard arguments
        cmd.extend([
            '--config_file', config_path,
            '--dataset_config', dataset_path,  # Will need dataset config
            '--output_dir', config['training_arguments']['output_dir'],
            '--logging_dir', config['training_arguments']['logging_dir']
        ])
        
        # Add Flux/SD3 specific arguments
        model_args = config['model_arguments']
        if model_args.get('clip_l'):
            cmd.extend(['--clip_l', model_args['clip_l']])
        if model_args.get('t5xxl'):
            cmd.extend(['--t5xxl', model_args['t5xxl']])
        if model_args.get('clip_g'):  # SD3 only
            cmd.extend(['--clip_g', model_args['clip_g']])
        
        # Memory optimizations
        if model_args.get('lowram'):
            cmd.append('--lowram')
        if config['flux_sd3_specific'].get('blocks_to_swap', 0) > 0:
            cmd.extend(['--blocks_to_swap', str(config['flux_sd3_specific']['blocks_to_swap'])])
        
        return cmd
    
    def validate_model_files(self, model_type: str, model_paths: Dict[str, str]) -> bool:
        """Validate that required model files exist"""
        model_config = self.model_configs[model_type]
        
        print(f"🔍 Validating {model_type} model files...")
        
        # Check required text encoders
        for encoder in model_config['text_encoders']:
            if encoder not in model_paths or not model_paths[encoder]:
                print(f"❌ Missing {encoder} path for {model_type}")
                return False
            
            if not os.path.exists(model_paths[encoder]):
                print(f"❌ {encoder} file not found: {model_paths[encoder]}")
                return False
            
            print(f"✅ {encoder}: {model_paths[encoder]}")
        
        return True
    
    def start_training(self, config_path: str, dataset_path: str) -> bool:
        """Start Flux/SD3 LoRA training"""
        print("🚀 Starting Flux/SD3 LoRA training...")
        
        try:
            # Get training command
            cmd = self.get_training_command(config_path, dataset_path)
            print(f"🔧 Training command: {' '.join(cmd)}")
            
            # Check if Kohya scripts exist
            with open(config_path, 'r') as f:
                config = toml.load(f)
            
            script_name = config['flux_sd3_specific']['training_script']
            script_path = os.path.join(self.sd_scripts_dir, script_name)
            
            if not os.path.exists(script_path):
                print(f"❌ Kohya script not found: {script_path}")
                print("💡 Make sure Derrian's backend is properly installed with Kohya scripts")
                return False
            
            print(f"✅ Found training script: {script_name}")
            print("⚠️ Training integration ready - execute command above to start training")
            
            # TODO: Actually execute training when ready
            # subprocess.run(cmd, check=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Training setup failed: {e}")
            return False