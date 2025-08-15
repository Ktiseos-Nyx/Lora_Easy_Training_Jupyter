# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/kohya_training_manager.py
"""
Refactored Training Manager leveraging Kohya's library system instead of custom implementations.

This replaces our custom training managers with a lightweight wrapper around Kohya's existing
strategy pattern and model utilities. We maintain our widget-friendly interface while using
battle-tested backend implementations.
"""

import os
import subprocess
import sys
from typing import Any, Dict, List

import toml

# Import Kohya's library system (required for training)
kohya_path = os.path.join(os.getcwd(), "trainer", "derrian_backend", "sd_scripts")
sys.path.insert(0, kohya_path)

# Kohya SS library imports (core training logic)
try:
    import library.config_util as config_util
    import library.train_util as train_util
    from library import model_util
    # Strategy imports for different model types
    from library.strategy_base import (TextEncoderOutputsCachingStrategy,
                                       TextEncodingStrategy, TokenizeStrategy)
    from library.strategy_flux import (FluxTextEncodingStrategy,
                                       FluxTokenizeStrategy)
    from library.strategy_sd import SdTextEncodingStrategy, SdTokenizeStrategy
    from library.strategy_sd3 import (Sd3TextEncodingStrategy,
                                      Sd3TokenizeStrategy)
    from library.strategy_sdxl import (SdxlTextEncoderOutputsCachingStrategy,
                                       SdxlTextEncodingStrategy,
                                       SdxlTokenizeStrategy)
    from library.utils import setup_logging
    setup_logging()
    
    # Also set up our file-based logging for easier debugging
    from .logging_config import setup_file_logging
    setup_file_logging()
except ImportError as e:
    raise ImportError(
        f"❌ TRAINING SYSTEM UNAVAILABLE\n"
        f"🔧 SOLUTION: Run 'python installer.py' to install required dependencies\n"
        f"📍 TECHNICAL ERROR: {str(e)}\n"
        f"📝 NOTE: This system requires Kohya's sd-scripts for training functionality"
    ) from e

# Import Derrian's utilities (validation and processing functions)
derrian_path = os.path.join(os.getcwd(), "trainer", "derrian_backend")
sys.path.insert(0, derrian_path)
try:
    # Import specific utility functions from individual Python files
    from utils.process import *
    from utils.validation import *
except ImportError as e:
    raise ImportError(f"❌ Derrian utilities not found: {e}\n"
                     f"🔧 Please run: python installer.py\n"
                     f"📝 This system requires Derrian's validation and processing utilities.") from e

# Logging is set up during import above if Kohya is available
import logging

logger = logging.getLogger(__name__)


class KohyaTrainingManager:
    """
    🧪 HYBRID TRAINING MANAGER 💥
    
    Combines the best of both worlds:
    - Kohya SS library (strategy pattern, model utilities, core training logic)
    - Derrian Distro (validation, processing, experimental features, custom optimizers)
    
    This gives us both the stability of Kohya's proven library system AND
    the advanced features and validation from Derrian's distribution.
    
    Features restored from archived system:
    - Advanced validation using Derrian's validation.py system
    - Experimental features framework (Kohya-compatible only)
    - Enhanced logging (TensorBoard, WandB)
    - Sophisticated optimizer configurations with detailed args
    - Custom optimizers (CAME, StableAdamW, Compass, etc.)
    """

    # Model type detection patterns
    MODEL_TYPE_PATTERNS = {
        'sd15': ['v1-5', 'sd-v1', 'sd_v1'],
        'sd20': ['v2-0', 'sd-v2', 'sd_v2', '768-v-ema'],
        'sdxl': ['xl-base', 'sdxl', 'xl_base', 'xl', '-xl-', '_xl_', 'illustrious', 'pony', 'noobai', 'animagine'],
        'flux': ['flux', 'FLUX'],
        'sd3': ['sd3', 'SD3']
    }

    # Script mapping for different model types
    SCRIPT_MAPPING = {
        'sd15': 'train_network.py',
        'sd20': 'train_network.py',
        'sdxl': 'sdxl_train_network.py',
        'flux': 'flux_train_network.py',
        'sd3': 'sd3_train_network.py'
    }

    def __init__(self):
        """Initialize with current working directory (notebook compatibility)"""
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.config_dir = os.path.join(self.project_root, "training_configs")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.output_dir = os.path.join(self.project_root, "output")
        self.logging_dir = os.path.join(self.project_root, "logs")

        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)

        self.process = None

        # Initialize Kohya strategies for different model types
        self._init_strategies()

        # Detection systems
        self.supported_optimizers = self._get_supported_optimizers()
        self.supported_schedulers = self._get_supported_schedulers()
        self.model_configs = self._init_model_configs()
        self.memory_profiles = self._init_memory_profiles()

        # 🔧 Restore advanced features from archived system
        self._setup_custom_optimizers()
        self.advanced_optimizers = self._init_advanced_optimizers()
        self.standard_optimizers = self._init_standard_optimizers()
        self.lycoris_methods = self._init_lycoris_methods()
        self.experimental_features = self._init_experimental_features()
        self.logging_options = self._init_logging_options()

        # Reduced verbosity for startup - only log if debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("KohyaTrainingManager initialized with Kohya library system")

    def _init_strategies(self):
        """Initialize Kohya's strategy system for different model architectures"""

        self.strategies = {
            'sd15': {
                'tokenize': SdTokenizeStrategy,
                'text_encoding': SdTextEncodingStrategy,
            },
            'sd20': {
                'tokenize': SdTokenizeStrategy,
                'text_encoding': SdTextEncodingStrategy,
            },
            'sdxl': {
                'tokenize': SdxlTokenizeStrategy,
                'text_encoding': SdxlTextEncodingStrategy,
                'caching': SdxlTextEncoderOutputsCachingStrategy,
            },
            'flux': {
                'tokenize': FluxTokenizeStrategy,
                'text_encoding': FluxTextEncodingStrategy,
            },
            'sd3': {
                'tokenize': Sd3TokenizeStrategy,
                'text_encoding': Sd3TextEncodingStrategy,
            }
        }

    def _get_supported_optimizers(self) -> Dict[str, Dict]:
        """Get optimizers supported by current Kohya installation"""
        optimizers = {
            # Standard PyTorch optimizers
            'AdamW': {'description': 'AdamW optimizer (recommended)', 'stable': True},
            'AdamW8bit': {'description': '8-bit AdamW (memory efficient)', 'stable': True},
            'SGDNesterov': {'description': 'SGD with Nesterov momentum', 'stable': True},
            'SGDNesterov8bit': {'description': '8-bit SGD Nesterov', 'stable': True},

            # Advanced optimizers (if available)
            'Lion': {'description': 'Lion optimizer', 'stable': True},
            'Lion8bit': {'description': '8-bit Lion optimizer', 'stable': True},
            'DAdaptation': {'description': 'D-Adaptation (auto LR)', 'stable': False},
            'DAdaptAdam': {'description': 'D-Adapt Adam', 'stable': False},
            'DAdaptAdaGrad': {'description': 'D-Adapt AdaGrad', 'stable': False},
            'DAdaptLion': {'description': 'D-Adapt Lion', 'stable': False},
            'Prodigy': {'description': 'Prodigy optimizer', 'stable': False},
        }

        # Check for CAME optimizer (from Derrian distro)
        came_path = os.path.join(self.sd_scripts_dir, "library", "came_optimizer.py")
        if os.path.exists(came_path):
            optimizers['CAME'] = {'description': 'CAME optimizer (Derrian distro)', 'stable': False}

        return optimizers

    def _get_supported_schedulers(self) -> List[str]:
        """Get learning rate schedulers supported by Kohya"""
        return [
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup", "adafactor"
        ]

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

    def detect_model_type(self, model_path: str) -> str:
        """
        Detect model type from model path using Kohya's model utilities
        """
        if not KOHYA_AVAILABLE:
            logger.warning("Kohya not available, using basic detection")
            return self._basic_model_detection(model_path)

        try:
            # Use Kohya's model detection utilities
            model_path_lower = model_path.lower()

            # Check for explicit patterns first
            for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
                if any(pattern in model_path_lower for pattern in patterns):
                    logger.info(f"Detected model type: {model_type} (pattern match)")
                    return model_type

            # Use file size as additional heuristic for model type detection
            if os.path.exists(model_path) and (model_path.endswith('.safetensors') or model_path.endswith('.ckpt')):
                try:
                    file_size_gb = os.path.getsize(model_path) / (1024**3)  # Size in GB
                    logger.debug(f"Model file size: {file_size_gb:.1f}GB")
                    
                    # File size based detection (common sizes, largest first)
                    if file_size_gb > 10:  # Flux models are much larger (12-24GB)
                        logger.info(f"Detected Flux model based on file size: {file_size_gb:.1f}GB")
                        return 'flux'  
                    elif file_size_gb > 8:  # SD3 models are also large (8-12GB)
                        logger.info(f"Detected SD3 model based on file size: {file_size_gb:.1f}GB")
                        return 'sd3'
                    elif file_size_gb > 5.5:  # SDXL models are typically 6-7GB
                        logger.info(f"Detected SDXL model based on file size: {file_size_gb:.1f}GB")
                        return 'sdxl'
                    else:  # SD 1.5/2.0 models are typically 2-4GB
                        logger.info(f"Detected SD1.5/2.0 model based on file size: {file_size_gb:.1f}GB")
                        return 'sd15'
                        
                except Exception as e:
                    logger.warning(f"File size detection failed: {e}")
                    # Continue to fallback detection

        except Exception as e:
            logger.error(f"Model type detection failed: {e}")

        # Fallback to basic detection
        return self._basic_model_detection(model_path)

    def _basic_model_detection(self, model_path: str) -> str:
        """Basic model type detection fallback"""
        model_path_lower = model_path.lower()

        for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
            if any(pattern in model_path_lower for pattern in patterns):
                return model_type

        return 'sd15'  # Safe default

    def get_training_script(self, model_type: str) -> str:
        """Get the appropriate training script for model type"""
        script_name = self.SCRIPT_MAPPING.get(model_type, 'train_network.py')
        script_path = os.path.join(self.sd_scripts_dir, script_name)

        if not os.path.exists(script_path):
            logger.warning(f"Training script not found: {script_path}")
            # Fallback to basic script
            fallback_script = os.path.join(self.sd_scripts_dir, 'train_network.py')
            if os.path.exists(fallback_script):
                return fallback_script
            else:
                raise FileNotFoundError(f"No training script found for {model_type}")

        return script_path

    def create_config_toml(self, config: Dict) -> str:
        """
        Create training configuration TOML using Kohya's config utilities
        """
        model_type = config.get('model_type', 'sd15')

        if model_type in self.model_configs:
            return self._create_flux_sd3_config_toml(config)
        else:
            return self._create_standard_config_toml(config)

    def _create_flux_sd3_config_toml(self, config: Dict) -> str:
        """
        Create a TOML config for Flux/SD3 models.
        """
        model_type = config['model_type']
        model_config = self.model_configs[model_type]

        memory_profile_name = config.get('memory_profile') or self.detect_optimal_memory_profile(model_type)
        memory_profile = self.memory_profiles[memory_profile_name]

        preset_name = config.get('training_preset', 'concept_learning')
        preset = self.training_presets[preset_name]

        toml_config = {
            "model_arguments": {
                "pretrained_model_name_or_path": model_config['model_name'],
                "clip_l_path": config.get('clip_l_path'),
                "t5xxl_path": config.get('t5xxl_path'),
                "clip_g_path": config.get('clip_g_path'), # SD3 only
                "cache_latents": memory_profile.get('cache_latents', True),
                "cache_latents_to_disk": memory_profile.get('cache_latents_to_disk', True),
                "lowram": memory_profile.get('lowram', False),
            },
            "network_arguments": {
                "network_dim": config.get('network_dim', preset['network_dim']),
                "network_alpha": config.get('network_alpha', preset['network_alpha']),
                "network_module": 'networks.lora',
            },
            "training_arguments": {
                "output_dir": self.output_dir,
                "logging_dir": self.logging_dir,
                "output_name": config.get('output_name', 'lora'),
                "resolution": f"{model_config['resolution']},{model_config['resolution']}",
                "train_batch_size": memory_profile['train_batch_size'],
                "gradient_accumulation_steps": memory_profile['gradient_accumulation_steps'],
                "max_train_epochs": config.get('epochs', preset['epochs']),
                "learning_rate": config.get('learning_rate', preset['learning_rate']),
                "text_encoder_lr": config.get('text_encoder_lr', preset['text_encoder_lr']),
                "lr_scheduler": preset['scheduler'],
                "lr_warmup_ratio": preset['warmup_ratio'],
                "optimizer_type": memory_profile['optimizer'],
                "mixed_precision": memory_profile['mixed_precision'],
                "gradient_checkpointing": memory_profile['gradient_checkpointing'],
                "save_precision": 'fp16',
                "save_model_as": 'safetensors',
                "save_every_n_epochs": config.get('save_every_n_epochs', 1),
                "seed": config.get('seed', 42),
            },
            "flux_sd3_specific": {
                "blocks_to_swap": memory_profile.get('blocks_to_swap', 0),
            }
        }

        config_path = os.path.join(self.config_dir, f"{config.get('output_name', 'lora')}_config.toml")
        with open(config_path, 'w') as f:
            toml.dump(toml_config, f)

        logger.info(f"Created Flux/SD3 config TOML: {config_path}")
        return config_path

    def _create_standard_config_toml(self, config: Dict) -> str:
        """Fallback TOML config creation for SD1.5/SDXL"""
        config_path = os.path.join(self.config_dir, f"{config.get('output_name', 'lora')}_config.toml")

        # Basic config structure
        toml_config = {
            "model_arguments": {
                "pretrained_model_name_or_path": config.get('model_path', ''),
                "v2": config.get('v2', False),
                "v_parameterization": config.get('v_parameterization', False),
                "clip_skip": config.get('clip_skip', 2),
            },
            # Dataset config moved to separate dataset.toml file
            "training_arguments": {
                "output_dir": config.get('output_dir', self.output_dir),
                "output_name": config.get('output_name', 'lora'),
                "learning_rate": config.get('unet_lr'),
                "text_encoder_lr": config.get('text_encoder_lr'),
                "lr_scheduler": config.get('lr_scheduler', 'cosine'),
                "optimizer_type": config.get('optimizer', 'AdamW8bit'),
                "max_train_epochs": config.get('epochs', 10),
                "save_every_n_epochs": config.get('save_every_n_epochs', 1),
                "mixed_precision": config.get('mixed_precision', 'fp16'),
                "save_precision": config.get('save_precision', 'fp16'),
                "seed": config.get('seed', 42),
            },
            "network_arguments": {
                "network_module": config.get('network_module', 'networks.lora'),
                "network_dim": config.get('network_dim'),
                "network_alpha": config.get('network_alpha'),
            },
        }

        with open(config_path, 'w') as f:
            toml.dump(toml_config, f)

        logger.info(f"Created config TOML: {config_path}")
        return config_path

    def create_dataset_toml(self, config: Dict) -> str:
        """
        Create dataset configuration TOML using proper Kohya format
        """
        dataset_path = os.path.join(self.config_dir, f"{config.get('output_name', 'lora')}_dataset.toml")
        
        # Kohya dataset config format from sd_scripts docs
        dataset_config = {
            "general": {
                "shuffle_caption": config.get('shuffle_caption', True),
                "caption_extension": config.get('caption_extension', '.txt'),
                "keep_tokens": config.get('keep_tokens', 1)
            },
            "datasets": [{
                "resolution": config.get('resolution', 512),
                "batch_size": config.get('batch_size', 1),
                "subsets": [{
                    "image_dir": self._ensure_absolute_dataset_path(config.get('dataset_path', '')),
                    "num_repeats": config.get('num_repeats', 10)
                }]
            }]
        }
        
        with open(dataset_path, 'w') as f:
            toml.dump(dataset_config, f)
            
        logger.info(f"Created dataset TOML: {dataset_path}")
        return dataset_path

    def start_training(self, config: Dict, monitor_widget=None) -> bool:
        """
        Start training using Kohya's training scripts and our configuration
        """
        logger.info("🚀 Starting training with Kohya backend")

        try:
            # Detect model type if not provided
            if 'model_type' not in config:
                config['model_type'] = self.detect_model_type(config.get('model_path', ''))

            # Validate config using Derrian's comprehensive validation system
            logger.info("🔍 Validating training configuration...")
            passed, errors, validated_args, validated_dataset_args, tag_data = self.validate_config_with_derrian(config)
            
            if not passed:
                error_msg = f"❌ Configuration validation failed:\n" + "\n".join(f"  • {error}" for error in errors)
                logger.error(error_msg)
                if monitor_widget:
                    monitor_widget.update_phase("Configuration validation failed", "error")
                return False

            logger.info("✅ Configuration validation passed")

            # Create both configuration files using Derrian's proven TOML generators
            # Instead of our guessed implementations, use the actual Derrian functions
            try:
                # Change to Derrian backend directory for proper path handling
                original_cwd = os.getcwd()
                os.chdir(os.path.join(self.project_root, "trainer", "derrian_backend"))
                
                # Use validated args from Derrian's validation system
                _, config_path = process_args(validated_args)
                _, dataset_path = process_dataset_args(validated_dataset_args)
                
                # Convert relative paths to absolute paths
                config_path = os.path.abspath(config_path)
                dataset_path = os.path.abspath(dataset_path)
                
                logger.info(f"✅ Generated config: {config_path}")
                logger.info(f"✅ Generated dataset config: {dataset_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate TOML files with Derrian functions: {e}")
                # Fallback to our basic TOML generation
                logger.warning("Falling back to basic TOML generation")
                config_path = self.create_config_toml(config)
                dataset_path = self.create_dataset_toml(config)
            finally:
                # Always restore original working directory
                if 'original_cwd' in locals():
                    os.chdir(original_cwd)

            # Get training command with both config files
            cmd = self._get_training_command(config, config_path, dataset_path)

            # Always use current Python executable for environment-agnostic execution
            # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
            python_executable = sys.executable
            cmd.insert(0, python_executable)

            # Setup environment
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Change to scripts directory for execution
            original_cwd = os.getcwd()
            os.chdir(self.sd_scripts_dir)

            try:
                # Launch training process
                logger.info(f"Executing: {' '.join(cmd)}")

                if monitor_widget:
                    monitor_widget.clear_log()
                    monitor_widget.update_phase("Starting Kohya training...", "info")

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )

                # Monitor output
                for line in iter(self.process.stdout.readline, ''):
                    print(line, end='')
                    if monitor_widget:
                        monitor_widget.parse_training_output(line)

                self.process.stdout.close()
                return_code = self.process.wait()

                if return_code == 0:
                    logger.info("✅ Training completed successfully!")
                    return True
                else:
                    logger.error(f"❌ Training failed with return code: {return_code}")
                    return False

            finally:
                os.chdir(original_cwd)
                self.process = None

        except Exception as e:
            logger.error(f"Training error: {e}")
            if monitor_widget:
                monitor_widget.update_phase(f"Training failed: {e}", "error")
            return False

    def _get_training_command(self, config: Dict, config_path: str, dataset_path: str) -> List[str]:
        """
        Generate the training command for the specified model type using both config files.
        """
        model_type = config.get('model_type', 'sd15')
        script_name = self.SCRIPT_MAPPING.get(model_type, 'train_network.py')
        script_path = os.path.join(self.sd_scripts_dir, script_name)

        # Proper Kohya command format with both config files
        cmd = [
            script_path, 
            "--config_file", config_path,
            "--dataset_config", dataset_path
        ]

        # Add model-specific arguments for Flux/SD3
        if model_type in ['flux', 'sd3']:
            model_config = self.model_configs[model_type]
            if config.get('clip_l_path'):
                cmd.extend(["--clip_l_path", config['clip_l_path']])
            if config.get('t5xxl_path'):
                cmd.extend(["--t5xxl_path", config['t5xxl_path']])
            if model_type == 'sd3' and config.get('clip_g_path'):
                cmd.extend(["--clip_g_path", config['clip_g_path']])

        return cmd

    def stop_training(self):
        """Stop the training process"""
        if self.process and self.process.poll() is None:
            logger.info("🛑 Stopping training process...")
            self.process.terminate()
            self.process.wait()
            self.process = None
            logger.info("Training process stopped")

    def validate_config(self, config: Dict) -> List[str]:
        """
        Validate training configuration using Kohya's validation utilities
        """
        errors = []

        # Basic validation
        required_fields = ['model_path', 'dataset_path', 'output_name']
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")

        # Path validation
        if config.get('model_path') and not os.path.exists(config['model_path']):
            errors.append(f"Model path not found: {config['model_path']}")

        if config.get('dataset_path') and not os.path.exists(config['dataset_path']):
            errors.append(f"Dataset path not found: {config['dataset_path']}")

        # Optimizer validation
        optimizer = config.get('optimizer', 'AdamW8bit')
        if optimizer not in self.supported_optimizers:
            errors.append(f"Unsupported optimizer: {optimizer}")

        # Learning rate validation
        lr = config.get('learning_rate', 0.0001)
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append("Learning rate must be a positive number")

        # Model type validation
        model_type = config.get('model_type')
        if model_type and model_type not in self.SCRIPT_MAPPING:
            errors.append(f"Unsupported model type: {model_type}")

        return errors

    def get_model_info(self, model_path: str) -> Dict:
        """
        Get model information using Kohya's model utilities
        """
        info = {
            'path': model_path,
            'type': 'unknown',
            'exists': os.path.exists(model_path),
            'size': 0
        }

        if info['exists']:
            info['size'] = os.path.getsize(model_path)
            info['type'] = self.detect_model_type(model_path)

        return info

    # ==================== RESTORED ADVANCED FEATURES ====================
    # Methods restored from the archived training manager system

    def _setup_custom_optimizers(self):
        """Setup Python path for Derrian's custom optimizers"""
        custom_optimizer_path = os.path.join(self.trainer_dir, "derrian_backend", "custom_scheduler", "LoraEasyCustomOptimizer")
        if custom_optimizer_path not in sys.path:
            sys.path.insert(0, custom_optimizer_path)

    def _init_advanced_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize advanced optimizer configurations from archived system"""
        return {
            'StableAdamW': {
                'optimizer_type': 'LoraEasyCustomOptimizer.utils.StableAdamW',
                'args': ['weight_decay=0.01'],
                'description': 'Improved stability AdamW variant'
            },
            'Compass': {
                'optimizer_type': 'LoraEasyCustomOptimizer.compass.Compass',
                'args': ['weight_decay=0.01'],
                'description': 'Compass optimizer for better convergence'
            },
            'LPF_AdamW': {
                'optimizer_type': 'LoraEasyCustomOptimizer.lpfadamw.LPF_AdamW',
                'args': ['weight_decay=0.01'],
                'description': 'Low-pass filtered AdamW'
            }
        }

    def _init_standard_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard optimizer configurations with detailed args"""
        return {
            'Prodigy': {
                'optimizer_type': 'Prodigy',
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
            'Lion': {
                'optimizer_type': 'Lion',
                'args': ['weight_decay=0.01', 'betas=[0.9,0.99]'],
                'description': 'Lion optimizer'
            }
        }

    def _init_lycoris_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize LyCORIS method configurations - OFFICIAL Algo-List.md specifications"""
        return {
            'loha': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=loha'],
                'description': 'LoHa: Low-rank Hadamard (dim≤32, alpha≤dim, rank≤dim²)',
                'recommendations': 'Recommended: dim≤32, alpha from 1 to dim. Higher dim may cause unstable loss.'
            },
            'lokr': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=lokr'],
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
                'network_args': ['algo=dylora'],
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
        }

    def _init_experimental_features(self) -> Dict[str, Dict[str, Any]]:
        """Initialize experimental feature configurations (Kohya-compatible only)"""
        return {
            'multi_resolution': {
                'enabled': False,
                'description': 'Train on multiple resolutions simultaneously',
                'status': 'experimental'
            },
            'advanced_loss_scaling': {
                'enabled': False,
                'description': 'Dynamic loss scaling for mixed precision',
                'status': 'experimental'
            },
            'gradient_checkpointing': {
                'enabled': False,
                'description': 'Trade compute for memory (Kohya native)',
                'status': 'stable'
            }
        }

    def _init_logging_options(self):
        """Initialize logging system options"""
        return {
            'none': {
                'description': 'No logging (standard console output only)',
                'args': []
            },
            'tensorboard': {
                'description': 'TensorBoard logging for training metrics',
                'args': ['--log_with', 'tensorboard']
            },
            'wandb': {
                'description': 'Weights & Biases logging for experiment tracking',
                'args': ['--log_with', 'wandb']
            },
            'both': {
                'description': 'Both TensorBoard and WandB logging',
                'args': ['--log_with', 'all']
            }
        }

    def _detect_repeat_folders(self, dataset_dir):
        """
        Detect existing repeat folders in dataset directory using Kohya's format
        Returns list of tuples: (folder_name, repeats, concept_name)
        """
        if not os.path.exists(dataset_dir):
            return []

        repeat_folders = []

        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                # Use the same logic as our fixed calculator
                try:
                    parts = item.split('_', 1)
                    if len(parts) >= 2:
                        repeats = int(parts[0])
                        concept_name = parts[1]
                        repeat_folders.append((item, repeats, concept_name))
                except ValueError:
                    # Not a repeat folder, skip
                    continue

        return repeat_folders

    def validate_config_with_derrian(self, config: Dict) -> tuple[bool, List[str], Dict, Dict, Dict]:
        """
        🧪 HYBRID VALIDATION - Use Derrian's advanced validation system
        This is the missing piece we lost during refactoring!
        """
        # Try to use Derrian validation, fallback to basic if not available

        try:
            # Convert our config to Derrian's expected format
            derrian_args = self._convert_config_to_derrian_format(config)

            # Use Derrian's comprehensive validation system
            # The validate function was imported via "from utils.validation import *"
            passed, sdxl, errors, validated_args, validated_dataset_args, tag_data = validate(derrian_args)

            if passed:
                logger.info("✅ Derrian validation passed - config is safe for training")
            else:
                logger.warning(f"❌ Derrian validation failed with {len(errors)} errors")
                for error in errors:
                    logger.warning(f"   - {error}")

            return passed, errors, validated_args, validated_dataset_args, tag_data

        except Exception as e:
            logger.error(f"Derrian validation failed with exception: {e}")
            logger.warning("Falling back to basic validation")
            # Fallback to basic validation
            errors = self.validate_config(config)
            return len(errors) == 0, errors, config, {}, {}

    def _convert_config_to_derrian_format(self, config: Dict) -> Dict:
        """Convert our config format to what Derrian's validation expects - REAL conversion from archived version"""
        logger.debug("⚙️ Converting widget config to Derrian backend format...")
        
        # Proper Derrian args structure (from archived training_manager.py)
        derrian_args = {
            "basic": {
                "pretrained_model_name_or_path": config.get('model_path'),
                "output_dir": config.get('output_dir', self.output_dir),
                "output_name": config.get('output_name', 'lora'),
                "save_every_n_epochs": config.get('save_every_n_epochs', 1),
                "keep_only_last_n_epochs": config.get('keep_only_last_n_epochs', 0),
                "train_batch_size": config.get('batch_size', 1),
                "mixed_precision": config.get('mixed_precision', 'fp16'),
                "save_precision": config.get('save_precision', 'fp16'),
                "max_train_epochs": config.get('epochs', 10),
                "learning_rate": config.get('unet_lr', 1e-4),
                "text_encoder_lr": config.get('text_encoder_lr', 5e-5),
                "optimizer_type": config.get('optimizer', 'AdamW8bit'),
                "lr_scheduler": config.get('lr_scheduler', 'cosine'),
                "lr_warmup_ratio": config.get('lr_warmup_ratio', 0.1),
                "network_dim": config.get('network_dim', 16),
                "network_alpha": config.get('network_alpha', 8),
                "clip_skip": config.get('clip_skip', 2),
                "cache_latents": config.get('cache_latents', True),
                "cache_latents_to_disk": config.get('cache_latents_to_disk', False),
                "xformers": config.get('cross_attention') == "xformers",
                "sdpa": config.get('cross_attention') == "sdpa",
            },
            "sdxl": {
                "cache_text_encoder_outputs": config.get('cache_text_encoder_outputs', False),
            },
            "network": {
                "type": config.get('lora_type', 'lora'),
                "algo": config.get('lycoris_method', 'none'),
            }
        }

        # Proper Derrian dataset structure
        derrian_dataset = {
            "general": {
                "resolution": config.get('resolution', 512),
                "shuffle_caption": config.get('shuffle_caption', True),
                "keep_tokens": config.get('keep_tokens', 1),
                "flip_aug": config.get('flip_aug', False),
                "caption_extension": config.get('caption_extension', '.txt'),
                "enable_bucket": config.get('enable_bucket', True),
                "bucket_no_upscale": config.get('bucket_no_upscale', False),
                "bucket_reso_steps": config.get('bucket_reso_steps', 64),
                "min_bucket_reso": config.get('min_bucket_reso', 256),
                "max_bucket_reso": config.get('max_bucket_reso', 2048),
                "batch_size": config.get('batch_size', 1),
            },
            "subsets": [
                {
                    "image_dir": self._ensure_absolute_dataset_path(config.get('dataset_path', '')),
                    "num_repeats": config.get('num_repeats', 10),
                    "caption_extension": config.get('caption_extension', '.txt'),
                }
            ]
        }

        return {"args": derrian_args, "dataset": derrian_dataset}

    def _ensure_absolute_dataset_path(self, dataset_path):
        """
        Ensure dataset path is absolute and correctly points to the training dataset folder.
        
        Since sd_scripts runs from trainer/derrian_backend/sd_scripts/, we need absolute paths
        to find datasets in the project root.
        
        The dataset path should point directly to the folder containing images (e.g., "3_character_name"),
        NOT the parent "datasets" directory.
        """
        if not dataset_path:
            return dataset_path

        # Convert to absolute path if it's relative (relative to project root)
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.project_root, dataset_path)

        # Return the absolute path to the actual dataset folder
        # This should be the directory containing the training images
        return dataset_path
