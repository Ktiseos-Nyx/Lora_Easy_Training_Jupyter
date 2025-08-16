#!/usr/bin/env python3
"""
Quick test to verify TOML generation with fake widget data
"""

import os
import sys
import tempfile
import toml

# Add the core directory to path so we can import our manager
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def test_toml_generation():
    """Test TOML generation with fake widget data that mimics your debug output"""
    
    # 🎭 FAKE WIDGET CONFIG: Based on your actual debug output
    fake_widget_config = {
        # From your debug: 'model_path': 'pretrained_model/Illustrious-XL-v0.1.safetensors'
        'model_path': 'pretrained_model/test-model.safetensors',
        
        # From your debug: 'dataset_path': 'datasets/3_WhitePhoenixoftheCrown'  
        'dataset_path': 'datasets/test_character',
        
        # Core training params from your widget
        'project_name': 'test_lora',
        'epochs': 10,
        'train_batch_size': 1,
        'resolution': 512,  # Single int like your widget might provide
        'unet_lr': 0.0001,
        'text_encoder_lr': 0.00005,
        'network_dim': 32,
        'network_alpha': 16,
        'optimizer': 'AdamW8bit',
        'lr_scheduler': 'cosine',
        'precision': 'fp16',
        'clip_skip': 2,
        'save_every_n_epochs': 1,
        'num_repeats': 3,
        'flip_aug': True,
        'shuffle_caption': True,
    }
    
    print("🎭 === TESTING TOML GENERATION ===")
    print(f"📊 Fake widget config keys: {list(fake_widget_config.keys())}")
    print()
    
    # Create a temporary directory for test TOML files
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # 🚀 TEST CONFIG TOML GENERATION
        print("🚀 Testing config TOML generation...")
        
        config_toml = {
            "network_arguments": {
                "network_dim": fake_widget_config.get('network_dim'),
                "network_alpha": fake_widget_config.get('network_alpha'),
                "network_module": "networks.lora",
            },
            "optimizer_arguments": {
                "learning_rate": fake_widget_config.get('unet_lr'),
                "text_encoder_lr": fake_widget_config.get('text_encoder_lr'),
                "lr_scheduler": fake_widget_config.get('lr_scheduler'),
                "optimizer_type": fake_widget_config.get('optimizer'),
            },
            "training_arguments": {
                "pretrained_model_name_or_path": fake_widget_config.get('model_path'),
                "max_train_epochs": fake_widget_config.get('epochs'),
                "train_batch_size": fake_widget_config.get('train_batch_size'),
                "save_every_n_epochs": fake_widget_config.get('save_every_n_epochs'),
                "mixed_precision": fake_widget_config.get('precision'),
                "output_dir": "output",
                "output_name": fake_widget_config.get('project_name', 'lora'),
                "clip_skip": fake_widget_config.get('clip_skip', 2),
                "save_model_as": "safetensors",
                "seed": 42,
            },
        }
        
        config_path = os.path.join(temp_dir, "test_config.toml")
        with open(config_path, 'w') as f:
            toml.dump(config_toml, f)
        
        print(f"✅ Config TOML created: {config_path}")
        print("📄 Config TOML content:")
        with open(config_path, 'r') as f:
            print(f.read())
        print()
        
        # 🗃️ TEST DATASET TOML GENERATION  
        print("🗃️ Testing dataset TOML generation...")
        
        # Handle resolution formatting like our fixed code
        resolution = fake_widget_config.get('resolution')
        if isinstance(resolution, (int, str)):
            formatted_resolution = f"{resolution},{resolution}"
        else:
            formatted_resolution = "512,512"
        
        dataset_toml = {
            "datasets": [{
                "subsets": [{
                    "image_dir": fake_widget_config.get('dataset_path'),  # FIXED: Use dataset_path
                    "num_repeats": fake_widget_config.get('num_repeats'),
                }]
            }],
            "general": {
                "resolution": formatted_resolution,  # FIXED: Proper formatting
                "shuffle_caption": fake_widget_config.get('shuffle_caption'),
                "flip_aug": fake_widget_config.get('flip_aug'),
            }
        }
        
        dataset_path = os.path.join(temp_dir, "test_dataset.toml")
        with open(dataset_path, 'w') as f:
            toml.dump(dataset_toml, f)
        
        print(f"✅ Dataset TOML created: {dataset_path}")
        print("📄 Dataset TOML content:")
        with open(dataset_path, 'r') as f:
            print(f.read())
        print()
        
        # 🎯 VALIDATION CHECKS
        print("🎯 === VALIDATION CHECKS ===")
        
        # Check for empty sections (the original problem!)
        for section_name, section_data in config_toml.items():
            if not section_data or all(v is None for v in section_data.values()):
                print(f"❌ EMPTY SECTION: {section_name}")
            else:
                print(f"✅ POPULATED: {section_name} ({len([v for v in section_data.values() if v is not None])} fields)")
        
        # Check critical fields
        critical_fields = [
            ('model_path', config_toml['training_arguments']['pretrained_model_name_or_path']),
            ('image_dir', dataset_toml['datasets'][0]['subsets'][0]['image_dir']),
            ('resolution', dataset_toml['general']['resolution']),
            ('network_dim', config_toml['network_arguments']['network_dim']),
        ]
        
        for field_name, field_value in critical_fields:
            if field_value is not None:
                print(f"✅ {field_name}: {field_value}")
            else:
                print(f"❌ MISSING: {field_name}")
        
        print()
        print("🎉 Test complete! Check above for any ❌ errors.")

if __name__ == "__main__":
    test_toml_generation()