# T5_Training/core/t5_training_manager.py
import os
import sys
import toml
import subprocess
from typing import Dict, List, Optional, Any

class FluxSD3LoRAManager:
    """🚀 Flux/SD3 LoRA Training Manager for T5 Text Encoders
    
    Specialized for LoRA training on T5 text encoder components within:
    - Flux.1 (T5-XXL text encoder)
    - SD3/SD3.5 (T5-XXL text encoder)  
    - AuraFlow (T5-XXL text encoder)
    - Custom Flux variants
    
    Focus: LoRA adapters for T5 encoders while keeping diffusion UNet frozen
    """
    
    def __init__(self):
        # Project structure
        self.project_root = os.getcwd()
        self.t5_dir = os.path.join(self.project_root, "T5_Training")
        self.output_dir = os.path.join(self.t5_dir, "output")
        self.config_dir = os.path.join(self.t5_dir, "configs") 
        self.logging_dir = os.path.join(self.t5_dir, "logs")
        
        # Create directories
        for dir_path in [self.output_dir, self.config_dir, self.logging_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize configurations
        self.t5_model_configs = self._init_t5_model_configs()
        self.memory_profiles = self._init_memory_profiles()
        self.training_presets = self._init_training_presets()
        
        print("🤖 T5 Training Manager initialized!")
        print(f"📁 T5 workspace: {self.t5_dir}")
    
    def _init_t5_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize T5 model configurations for different architectures"""
        return {
            'auraflow_t5': {
                'model_name': 'google/t5-v1_1-xxl',
                'max_sequence_length': 256,
                'architecture': 'encoder-decoder',
                'parameters': '11B',
                'memory_requirement': 'high',
                'description': 'T5-XXL for AuraFlow diffusion models'
            },
            'hidream_t5': {
                'model_name': 'google/t5-v1_1-large', 
                'max_sequence_length': 512,
                'architecture': 'encoder-decoder',
                'parameters': '770M',
                'memory_requirement': 'medium',
                'description': 'T5-Large for HiDream and HunyuanDiT'
            },
            'custom_t5_base': {
                'model_name': 'google/t5-v1_1-base',
                'max_sequence_length': 512,
                'architecture': 'encoder-decoder', 
                'parameters': '220M',
                'memory_requirement': 'low',
                'description': 'T5-Base for custom implementations and experimentation'
            }
        }
    
    def _init_memory_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Memory optimization profiles for different hardware setups"""
        return {
            'ultra_low_memory': {  # 8GB VRAM
                'gradient_checkpointing': True,
                'mixed_precision': 'fp16',
                'batch_size': 1,
                'gradient_accumulation_steps': 32,
                'optimizer': 'AdaFactor',  # Most memory efficient
                'cache_strategy': 'disk_only',
                'description': 'Maximum memory savings for 8GB cards'
            },
            'low_memory': {  # 12GB VRAM
                'gradient_checkpointing': True,
                'mixed_precision': 'bf16',
                'batch_size': 2,
                'gradient_accumulation_steps': 16,
                'optimizer': 'AdamW8bit',
                'cache_strategy': 'hybrid',
                'description': 'Balanced efficiency for 12GB cards'
            },
            'standard': {  # 16GB+ VRAM
                'gradient_checkpointing': False,
                'mixed_precision': 'bf16',
                'batch_size': 4,
                'gradient_accumulation_steps': 8,
                'optimizer': 'AdamW',
                'cache_strategy': 'memory',
                'description': 'Standard training for 16GB+ cards'
            },
            'high_performance': {  # 24GB+ VRAM
                'gradient_checkpointing': False,
                'mixed_precision': 'bf16',
                'batch_size': 8,
                'gradient_accumulation_steps': 4,
                'optimizer': 'AdamW',
                'cache_strategy': 'memory',
                'description': 'High performance for 24GB+ cards'
            }
        }
    
    def _init_training_presets(self) -> Dict[str, Dict[str, Any]]:
        """Training presets for different use cases"""
        return {
            'concept_learning': {
                'learning_rate': 5e-5,
                'epochs': 3,
                'warmup_ratio': 0.1,
                'scheduler': 'cosine_with_restarts',
                'focus': 'encoder_only',
                'description': 'Learn new concepts and objects'
            },
            'style_adaptation': {
                'learning_rate': 1e-4,
                'epochs': 5,
                'warmup_ratio': 0.05,
                'scheduler': 'linear',
                'focus': 'full_model',
                'description': 'Adapt to artistic styles and aesthetics'
            },
            'prompt_following': {
                'learning_rate': 3e-5,
                'epochs': 2,
                'warmup_ratio': 0.15,
                'scheduler': 'constant_with_warmup',
                'focus': 'decoder_emphasis',
                'description': 'Improve prompt understanding and following'
            }
        }
    
    def detect_optimal_memory_profile(self) -> str:
        """Auto-detect optimal memory profile based on available VRAM"""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🔍 Detected {vram_gb:.1f}GB VRAM")
                
                if vram_gb >= 24:
                    return 'high_performance'
                elif vram_gb >= 16:
                    return 'standard'
                elif vram_gb >= 12:
                    return 'low_memory'
                else:
                    return 'ultra_low_memory'
            else:
                print("⚠️ CUDA not available - using CPU-optimized settings")
                return 'ultra_low_memory'
        except Exception as e:
            print(f"⚠️ Could not detect VRAM: {e}")
            return 'low_memory'  # Safe default
    
    def create_training_config(self, config: Dict[str, Any]) -> str:
        """Create T5 training configuration file"""
        
        # Auto-detect memory profile if not specified
        memory_profile = config.get('memory_profile', self.detect_optimal_memory_profile())
        memory_settings = self.memory_profiles[memory_profile]
        
        # Get model configuration
        model_config = self.t5_model_configs[config['model_type']]
        
        # Build comprehensive training config
        training_config = {
            'model_arguments': {
                'pretrained_model_name_or_path': model_config['model_name'],
                'max_sequence_length': model_config['max_sequence_length'],
                'torch_dtype': 'bfloat16' if memory_settings['mixed_precision'] == 'bf16' else 'float16'
            },
            'training_arguments': {
                'output_dir': self.output_dir,
                'logging_dir': self.logging_dir,
                'learning_rate': config.get('learning_rate', 5e-5),
                'num_train_epochs': config.get('epochs', 3),
                'per_device_train_batch_size': memory_settings['batch_size'],
                'gradient_accumulation_steps': memory_settings['gradient_accumulation_steps'],
                'warmup_ratio': config.get('warmup_ratio', 0.1),
                'lr_scheduler_type': config.get('scheduler', 'cosine'),
                'gradient_checkpointing': memory_settings['gradient_checkpointing'],
                'bf16': memory_settings['mixed_precision'] == 'bf16',
                'fp16': memory_settings['mixed_precision'] == 'fp16',
                'dataloader_num_workers': 2,
                'save_strategy': 'epoch',
                'evaluation_strategy': 'no',
                'logging_steps': 50,
                'save_total_limit': 3
            },
            'optimizer_arguments': {
                'optimizer_type': memory_settings['optimizer'],
                'weight_decay': 0.01,
                'eps': 1e-8
            },
            'dataset_arguments': {
                'max_train_samples': config.get('max_samples', None),
                'preprocessing_num_workers': 4,
                'cache_strategy': memory_settings['cache_strategy']
            }
        }
        
        # Save config file
        config_filename = f"{config['project_name']}_t5_config.toml"
        config_path = os.path.join(self.config_dir, config_filename)
        
        with open(config_path, 'w') as f:
            f.write(toml.dumps(training_config))
        
        print(f"📄 T5 training config saved: {config_path}")
        print(f"🧠 Memory profile: {memory_profile} ({memory_settings['description']})")
        print(f"🤖 Model: {model_config['model_name']} ({model_config['parameters']} parameters)")
        
        return config_path
    
    def validate_training_setup(self, config_path: str) -> bool:
        """Validate T5 training environment and configuration"""
        print("🔍 Validating T5 training setup...")
        
        try:
            # Check if config file exists
            if not os.path.exists(config_path):
                print(f"❌ Config file not found: {config_path}")
                return False
            
            # Try to load and validate config
            with open(config_path, 'r') as f:
                config = toml.load(f)
            
            # Check required sections
            required_sections = ['model_arguments', 'training_arguments', 'optimizer_arguments']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing config section: {section}")
                    return False
            
            # Test T5 model loading
            model_name = config['model_arguments']['pretrained_model_name_or_path']
            print(f"🤖 Testing T5 model loading: {model_name}")
            
            try:
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                print("✅ T5 tokenizer loaded successfully")
                
                # Test model loading (without loading weights to save time/memory)
                from transformers import T5Config
                model_config = T5Config.from_pretrained(model_name)
                print("✅ T5 model configuration validated")
                
            except Exception as e:
                print(f"❌ T5 model validation failed: {e}")
                return False
            
            print("✅ T5 training setup validation complete!")
            return True
            
        except Exception as e:
            print(f"❌ Setup validation failed: {e}")
            return False
    
    def get_training_command(self, config_path: str, dataset_path: str) -> List[str]:
        """Generate training command for T5 fine-tuning"""
        return [
            'python', '-m', 'transformers_trainer',  # This would need to be implemented
            '--config_file', config_path,
            '--train_file', dataset_path,
            '--do_train',
            '--overwrite_output_dir',
            '--report_to', 'none'  # Disable wandb for now
        ]
    
    def start_training(self, config_path: str, dataset_path: str) -> bool:
        """Start T5 training process"""
        print("🚀 Starting T5 training...")
        
        if not self.validate_training_setup(config_path):
            print("❌ Training setup validation failed")
            return False
        
        try:
            # Get training command
            cmd = self.get_training_command(config_path, dataset_path)
            print(f"🔧 Training command: {' '.join(cmd)}")
            
            # TODO: Implement actual training script integration
            print("⚠️ T5 training script integration coming soon!")
            print("💡 For now, this validates setup and generates configs")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False