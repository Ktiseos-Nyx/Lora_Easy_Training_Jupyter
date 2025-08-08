# core/sd15_training_manager.py
"""
SD 1.5 Training Manager - Optimized for Stable Diffusion 1.5 LoRA Training

This manager extends the base training functionality with SD 1.5 specific optimizations:
- Uses train_network.py instead of sdxl_train_network.py
- Optimized default settings for 512x512 resolution
- Lower VRAM requirements (6-8GB vs 12-24GB for SDXL)
- Faster training iterations and convergence
"""

import os
from pathlib import Path
from .training_manager import HybridTrainingManager

class SD15TrainingManager(HybridTrainingManager):
    """
    SD 1.5 optimized training manager that inherits from HybridTrainingManager
    but overrides specific methods for SD 1.5 training requirements.
    """
    
    def __init__(self, project_root=None):
        super().__init__(project_root)
        print("🎯 Initialized SD 1.5 Training Manager")
        print("📐 Optimized for: 512x512 resolution, 6-8GB VRAM, faster convergence")
    
    def _find_training_script(self):
        """Find SD 1.5 training script (train_network.py)"""
        print("🔍 Looking for SD 1.5 training script (train_network.py)...")
        
        # SD 1.5 specific scripts in order of preference
        possible_scripts = [
            os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts", "train_network.py"),
            os.path.join(self.trainer_dir, "train_network.py"),
            os.path.join(self.sd_scripts_dir, "train_network.py")
        ]
        
        for script_path in possible_scripts:
            if os.path.exists(script_path):
                print(f"✅ Using SD 1.5 training script: {os.path.basename(script_path)}")
                return script_path
        
        # If SD 1.5 script not found, warn but don't fallback to SDXL
        print("❌ SD 1.5 training script (train_network.py) not found!")
        print("💡 Please ensure the training backend is properly installed.")
        print("⚠️ Do NOT use sdxl_train_network.py for SD 1.5 models - it won't work correctly!")
        
        return None
    
    def get_recommended_settings(self):
        """Get SD 1.5 optimized default settings"""
        return {
            # Network settings optimized for SD 1.5
            'network_dim': 16,
            'network_alpha': 8,
            'network_module': 'networks.lora',  # Standard LoRA works great for SD 1.5
            
            # Resolution optimized for SD 1.5
            'resolution': 512,  # Native SD 1.5 resolution
            
            # Learning rates - SD 1.5 can handle slightly higher rates
            'learning_rate': 8e-4,  # Slightly higher than SDXL
            'text_encoder_lr': 1e-4,
            
            # Training settings optimized for faster SD 1.5 convergence
            'max_train_epochs': 10,  # Fewer epochs needed for SD 1.5
            'save_every_n_epochs': 2,
            
            # Batch size - can be higher due to lower VRAM usage
            'train_batch_size': 2,  # vs 1 for SDXL
            
            # Memory optimizations (less critical for SD 1.5 but still useful)
            'mixed_precision': 'fp16',
            'gradient_checkpointing': False,  # Less needed for SD 1.5
            
            # Scheduler settings
            'lr_scheduler': 'cosine_with_restarts',
            'lr_scheduler_num_cycles': 3,
            
            # Optimizer - CAME still beneficial for memory savings
            'optimizer_type': 'CAME',
            
            # Dataset settings
            'min_bucket_reso': 320,
            'max_bucket_reso': 768,  # Lower than SDXL
        }
    
    def validate_sd15_settings(self, config):
        """Validate settings are appropriate for SD 1.5 training"""
        warnings = []
        recommendations = []
        
        # Check resolution
        resolution = getattr(config, 'resolution', 512)
        if resolution > 768:
            warnings.append(f"⚠️ Resolution {resolution} is high for SD 1.5 (recommended: 512x512)")
            recommendations.append("Consider using 512x512 for optimal SD 1.5 training")
        
        # Check batch size
        batch_size = getattr(config, 'train_batch_size', 1)
        if batch_size == 1:
            recommendations.append("SD 1.5 can often handle batch_size=2 for faster training")
        
        # Check network settings
        dim = getattr(config, 'network_dim', 16)
        alpha = getattr(config, 'network_alpha', 8)
        if dim > 64:
            warnings.append(f"⚠️ Network dim {dim} is very high for SD 1.5 (may overfit)")
        
        # Check learning rates
        lr = getattr(config, 'learning_rate', 5e-4)
        if lr > 1e-3:
            warnings.append(f"⚠️ Learning rate {lr} is high (may cause instability)")
        
        # Print validation results
        if warnings:
            print("⚠️ SD 1.5 Training Warnings:")
            for warning in warnings:
                print(f"   {warning}")
        
        if recommendations:
            print("💡 SD 1.5 Training Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        if not warnings and not recommendations:
            print("✅ SD 1.5 training settings look good!")
        
        return len(warnings) == 0  # Return True if no warnings
    
    def start_training(self, config):
        """Start SD 1.5 training with validation"""
        print("🚀 Starting SD 1.5 LoRA Training...")
        print("=" * 60)
        
        # Validate SD 1.5 specific settings
        if not self.validate_sd15_settings(config):
            print("⚠️ There are warnings about your SD 1.5 settings.")
            print("💭 Consider reviewing the recommendations above.")
            print()
        
        # Call parent training method
        return super().start_training(config)
    
    def get_training_tips(self):
        """Get SD 1.5 specific training tips"""
        return [
            "🎯 SD 1.5 Training Tips:",
            "",
            "📐 Resolution: Use 512x512 for best results (native SD 1.5 resolution)",
            "⚡ Speed: SD 1.5 trains 2-3x faster than SDXL",
            "🧠 VRAM: Can train on 6-8GB GPUs (vs 12-24GB for SDXL)",
            "🎚️ Batch Size: Try batch_size=2 or higher if you have enough VRAM",
            "📊 Steps: Target 200-800 steps (faster convergence than SDXL)",
            "🔄 Learning Rates: Can use slightly higher rates than SDXL",
            "🎨 Network Dim: 16/8 or 32/16 work well for most SD 1.5 LoRAs",
            "⏱️ Epochs: Usually need fewer epochs than SDXL (8-15 typical)",
            "",
            "🏆 Popular SD 1.5 Base Models:",
            "   • SD 1.5 Base - Good starting point",
            "   • Anything v4/v5 - Anime/illustration focused", 
            "   • Realistic Vision - Photorealistic images",
            "   • Counterfeit v3 - Anime characters",
            "   • DreamShaper - Fantasy and creative content",
        ]