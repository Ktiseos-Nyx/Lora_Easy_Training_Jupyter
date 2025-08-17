# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/training_widget.py

import os
import warnings

import ipywidgets as widgets
from IPython.display import display

from core.refactored_training_manager import RefactoredTrainingManager

from .training_monitor_widget import TrainingMonitorWidget

# Suppress FutureWarnings at import time
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

class TrainingWidget:
    def __init__(self, training_manager=None):
        # Use dependency injection - accept manager instance or create default
        if training_manager is None:
            training_manager = RefactoredTrainingManager()

        self.manager = training_manager
        self.create_widgets()

    def _parse_learning_rate(self, lr_text):
        """Parse learning rate from text input supporting both scientific notation and decimals"""
        try:
            # Handle both scientific notation (5e-4) and decimal (0.0005)
            return float(lr_text)
        except ValueError:
            print(f"⚠️ Invalid learning rate format: {lr_text}. Using default 1e-4")
            return 1e-4

    def create_widgets(self):
        header_icon = "⭐"
        header_main = widgets.HTML(f"<h2>{header_icon} 3. Training Configuration</h2>")

        # --- Project Settings ---
        project_desc = widgets.HTML("<h3>▶️ Project Settings</h3><p>Define your project name, the path to your base model, and your dataset directory. You can also specify an existing LoRA to continue training from and your Weights & Biases API key for logging.</p>")
        self.project_name = widgets.Text(description="Project Name:", placeholder="e.g., my-awesome-lora (no spaces or special characters)", layout=widgets.Layout(width='99%'))
        self.model_type = widgets.Dropdown(options=['SD1.5/2.0', 'SDXL', 'Flux', 'SD3'], value='SDXL', description='Model Type:', style={'description_width': 'initial'})
        # Model selection with auto-populated dropdown
        self.model_dropdown = widgets.Dropdown(
            options=[('Select a model...', '')],
            description='Base Model:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        self.model_refresh_btn = widgets.Button(description="🔄 Refresh", button_style='info', layout=widgets.Layout(width='100px'))
        self.model_path = widgets.Text(description="Or custom path:", placeholder="Full path to model file", layout=widgets.Layout(width='99%'))

        # Auto-populate models on widget creation
        self._refresh_model_list()

        # Connect refresh button
        self.model_refresh_btn.on_click(lambda b: self._refresh_model_list())

        # Update model_path when dropdown changes
        def on_model_selected(change):
            if change['new']:
                self.model_path.value = change['new']
        self.model_dropdown.observe(on_model_selected, names='value')

        # Flux/SD3 specific widgets
        self.clip_l_path = widgets.Text(description="CLIP-L Path:", placeholder="Path to clip_l.safetensors", layout=widgets.Layout(width='99%'))
        self.clip_g_path = widgets.Text(description="CLIP-G Path:", placeholder="Path to clip_g.safetensors (for SD3)", layout=widgets.Layout(width='99%'))
        self.t5xxl_path = widgets.Text(description="T5-XXL Path:", placeholder="Path to t5xxl.safetensors", layout=widgets.Layout(width='99%'))
        self.flux_sd3_widgets = widgets.VBox([self.clip_l_path, self.clip_g_path, self.t5xxl_path])
        self.flux_sd3_widgets.layout.display = 'none' # Initially hidden

        self.dataset_dir = widgets.Text(description="Dataset Dir:", placeholder="Absolute path to your dataset directory (e.g., /path/to/my_dataset)", layout=widgets.Layout(width='99%'))
        self.continue_from_lora = widgets.Text(description="Continue from LoRA:", placeholder="Absolute path to an existing LoRA to continue training (optional)", layout=widgets.Layout(width='99%'))
        self.wandb_key = widgets.Password(description="WandB API Key:", placeholder="Your key will be hidden (e.g., ••••••••••••••••••••••••)", layout=widgets.Layout(width='99%'))

        def _on_model_type_change(change):
            if change['new'] in ['Flux', 'SD3']:
                self.flux_sd3_widgets.layout.display = 'block'
            else:
                self.flux_sd3_widgets.layout.display = 'none'

        self.model_type.observe(_on_model_type_change, names='value')

        # Model selection layout
        model_selection_box = widgets.HBox([self.model_dropdown, self.model_refresh_btn])

        project_box = widgets.VBox([project_desc, self.project_name, self.model_type, model_selection_box, self.model_path, self.flux_sd3_widgets, self.dataset_dir, self.continue_from_lora, self.wandb_key])

        # --- Training Configuration (merged: Basic Settings + Learning Rate + Training Options) ---
        training_config_desc = widgets.HTML("""<h3>▶️ Training Configuration</h3>
        <p>Configure all core training parameters including dataset settings, learning rates, optimizer, and training options. Total steps calculated automatically based on your dataset.</p>""")
        self.resolution = widgets.IntText(value=1024, description='Resolution:', style={'description_width': 'initial'})
        self.num_repeats = widgets.IntText(value=10, description='Num Repeats:', style={'description_width': 'initial'})
        self.epochs = widgets.IntText(value=10, description='Epochs:', style={'description_width': 'initial'})
        self.train_batch_size = widgets.IntText(value=4, description='Train Batch Size:', style={'description_width': 'initial'})
        self.flip_aug = widgets.Checkbox(value=False, description="Flip Augmentation (data augmentation)", indent=False)
        self.shuffle_caption = widgets.Checkbox(value=True, description="Shuffle Captions (improves variety, incompatible with text encoder caching)", indent=False)

        # Basic training options (moved from advanced)
        self.keep_tokens = widgets.IntText(value=0, description='Keep Tokens:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.clip_skip = widgets.IntText(value=2, description='Clip Skip:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))

        # Auto-detected dataset size (will be updated when dataset directory changes)
        self.dataset_size = widgets.IntText(value=0, description="Dataset Size:", style={'description_width': 'initial'}, disabled=True)
        self.step_calculator = widgets.HTML()

        # Auto-detect dataset size when dataset_dir changes
        def update_dataset_size(*args):
            dataset_path = self.dataset_dir.value.strip()
            if dataset_path:
                try:
                    from core.image_utils import count_images_in_directory
                    image_count = count_images_in_directory(dataset_path)
                    self.dataset_size.value = image_count
                    update_step_calculation()
                except Exception:
                    self.dataset_size.value = 0
            else:
                self.dataset_size.value = 0
            update_step_calculation()

        self.dataset_dir.observe(update_dataset_size, names='value')

        def update_step_calculation(*args):
            images = self.dataset_size.value
            repeats = self.num_repeats.value
            epochs = self.epochs.value
            batch_size = self.train_batch_size.value

            if batch_size > 0 and images > 0:
                total_steps = (images * repeats * epochs) // batch_size

                # Neutral color scheme
                color = "#17a2b8"  # Blue-teal, neutral

                self.step_calculator.value = f"""
                <div style='background: {color}20; padding: 10px; border-left: 4px solid {color}; margin: 5px 0;'>
                <strong>📊 Total Steps: {total_steps}</strong><br>
                {images} images × {repeats} repeats × {epochs} epochs ÷ {batch_size} batch = {total_steps} steps
                </div>
                """
            else:
                # No dataset or invalid parameters
                self.step_calculator.value = """
                <div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 5px 0;'>
                <strong>📊 Total Steps: Pending</strong><br>
                <em>Select a dataset directory to calculate training steps</em>
                </div>
                """

        # Attach observers to update calculation
        self.dataset_size.observe(update_step_calculation, names='value')
        self.num_repeats.observe(update_step_calculation, names='value')
        self.epochs.observe(update_step_calculation, names='value')
        self.train_batch_size.observe(update_step_calculation, names='value')

        # Initial calculation
        update_step_calculation()

        # Configuration warnings
        self.config_warnings = widgets.HTML()

        def check_config_conflicts(*args):
            warnings = []

            # Check text encoder caching vs shuffle caption conflict
            if self.cache_text_encoder_outputs.value and self.shuffle_caption.value:
                warnings.append("⚠️ Cannot use Caption Shuffling with Text Encoder Caching")

            # Check text encoder caching vs text encoder training conflict
            if self.cache_text_encoder_outputs.value and float(self.text_encoder_lr.value) > 0:
                warnings.append("⚠️ Cannot cache Text Encoder while training it (set Text LR to 0)")
                self.text_encoder_lr.value = '0'

            # Check random crop vs latent caching conflict
            if self.random_crop.value and self.cache_latents.value:
                warnings.append("⚠️ Cannot use Random Crop with Latent Caching - choose one or the other")

            if warnings:
                warning_html = "<div style='padding: 10px; border: 1px solid #856404; border-radius: 5px; margin: 5px 0;'>"
                warning_html += "<br>".join(warnings)
                warning_html += "<br><em>💡 Fix these conflicts to enable training</em></div>"
                self.config_warnings.value = warning_html
            else:
                self.config_warnings.value = ""

        # Note: Observers will be attached after all widgets are created

        # Learning rate widgets (no separate section header)
        self.unet_lr = widgets.Text(value='5e-4', placeholder='e.g., 5e-4 or 0.0005', description='🧠 UNet LR:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.text_encoder_lr = widgets.Text(value='1e-4', placeholder='e.g., 1e-4 or 0.0001', description='📝 Text LR:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.lr_scheduler = widgets.Dropdown(options=['cosine', 'cosine_with_restarts', 'constant', 'linear', 'polynomial', 'rex'], value='cosine', description='Scheduler:', style={'description_width': 'initial'})
        self.lr_scheduler_number = widgets.IntSlider(value=3, min=1, max=10, description='Scheduler Num (for restarts/polynomial):', style={'description_width': 'initial'}, continuous_update=False)
        self.lr_warmup_ratio = widgets.Text(value='0.05', description='Warmup Ratio:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.min_snr_gamma_enabled = widgets.Checkbox(value=True, description="Enable Min SNR Gamma (recommended for better results)", indent=False)
        self.min_snr_gamma = widgets.Text(value='5.0', description='Min SNR Gamma:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.ip_noise_gamma_enabled = widgets.Checkbox(value=False, description="Enable IP Noise Gamma", indent=False)
        self.ip_noise_gamma = widgets.FloatSlider(value=0.05, min=0.0, max=0.1, step=0.01, description='IP Noise Gamma:', style={'description_width': 'initial'}, continuous_update=False)
        self.multinoise = widgets.Checkbox(value=False, description="Multi-noise (can help with color balance)", indent=False)
        self.noise_offset = widgets.Text(value='0.0', description='Noise Offset:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        # Training options widgets (no separate section header)
        self.zero_terminal_snr = widgets.Checkbox(value=False, description="Zero Terminal SNR (recommended for SDXL)", indent=False)
        self.enable_bucket = widgets.Checkbox(value=True, description="Enable Bucket (resolution bucketing)", indent=False)
        self.gradient_checkpointing = widgets.Checkbox(value=True, description="Gradient Checkpointing (saves memory - REQUIRED for 4090)", indent=False)
        self.gradient_accumulation_steps = widgets.IntText(value=1, description='Gradient Accumulation Steps:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.max_grad_norm = widgets.Text(value='1.0', description='Max Grad Norm:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.full_fp16 = widgets.Checkbox(value=False, description="Full FP16 (more aggressive mixed precision)", indent=False)
        self.random_crop = widgets.Checkbox(value=False, description="Random Crop (data augmentation)", indent=False)

        # Bucketing settings (moved from "advanced")
        self.sdxl_bucket_optimization = widgets.Checkbox(value=False, description="📐 SDXL Bucket Optimization (32 steps vs 64 standard)", indent=False)
        self.min_bucket_reso = widgets.IntSlider(value=256, min=128, max=512, step=64, description='Min Bucket Resolution:', style={'description_width': 'initial'}, continuous_update=False)
        self.max_bucket_reso = widgets.IntSlider(value=2048, min=1024, max=4096, step=512, description='Max Bucket Resolution:', style={'description_width': 'initial'}, continuous_update=False)
        self.bucket_no_upscale = widgets.Checkbox(value=False, description="No Bucket Upscale (prevent upscaling small images)", indent=False)

        # VAE settings (moved from "advanced")
        self.vae_batch_size = widgets.IntSlider(value=1, min=1, max=8, step=1, description='VAE Batch Size:', style={'description_width': 'initial'}, continuous_update=False)
        self.no_half_vae = widgets.Checkbox(value=False, description="No Half VAE (fixes some VAE issues, uses more VRAM)", indent=False)

        # --- LoRA Structure ---
        lora_struct_desc = widgets.HTML("""
        <h3>▶️ LoRA Structure</h3>
        <p>Choose your LoRA type and define its dimensions. <strong>16 dim/8 alpha balances capacity and stability</strong> for most use cases (~100MB). LyCORIS methods can handle higher dimensions than regular LoRA.</p>
        
        <div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 10px 0;'>
        <h4>🧩 Conv Layers Explained:</h4>
        <p><strong>Conv Dim/Alpha</strong> control additional convolutional learning layers (for LoCon, LoKR, DyLoRA, etc.):</p>
        <ul>
        <li><strong>Conv layers help with:</strong> Fine details, textures, spatial features, artistic styles</li>
        <li><strong>When to use:</strong> Style LoRAs, complex characters, detailed concepts</li>
        <li><strong>Recommended:</strong> Start with same values as Network Dim/Alpha (16/8)</li>
        <li><strong>Higher values:</strong> More detail capture but larger file size and slower training</li>
        <li><strong>Skip for:</strong> Simple character LoRAs where standard LoRA works fine</li>
        </ul>
        </div>
        """)
        self.lora_type = widgets.Dropdown(
            options=[
                'LoRA', 'LoCon', 'LoKR', 'DyLoRA',
                'DoRA (Weight Decomposition)',
                'LoHa (Hadamard Product)',
                '(IA)³ (Few Parameters)',
                'GLoRA (Generalized LoRA)',
                'BOFT (Butterfly Transform)'
            ],
            value='LoRA',
            description='LoRA Type:',
            style={'description_width': 'initial'}
        )
        self.network_dim = widgets.IntText(value=16, description='Network Dim:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.network_alpha = widgets.IntText(value=8, description='Network Alpha:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.conv_dim = widgets.IntText(value=16, description='🧩 Conv Dim (for textures/details):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.conv_alpha = widgets.IntText(value=8, description='🧩 Conv Alpha (conv learning rate):', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        lora_box = widgets.VBox([lora_struct_desc, self.lora_type, self.network_dim, self.network_alpha, self.conv_dim, self.conv_alpha])

        # --- Sample Generation Settings ---
        sample_gen_desc = widgets.HTML("""
        <h3>▶️ Sample Generation Settings</h3>
        <p>Configure settings for generating sample images during training. These images will be generated at the end of each epoch to help you monitor your LoRA's progress.</p>
        """)
        self.sample_prompt = widgets.Textarea(
            value='masterpiece, best quality, 1girl, solo, <lora_name>, in a garden, sunny day',
            description='Sample Prompt:',
            layout=widgets.Layout(width='90%', height='80px'),
            style={'description_width': 'initial'}
        )
        self.sample_num_images = widgets.IntSlider(
            value=3,
            min=0,
            max=10,
            step=1,
            description='Number of Samples (per epoch):',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style={'description_width': 'initial'}
        )
        self.sample_resolution = widgets.Dropdown(
            options=[(f'{res}x{res}', res) for res in [512, 768, 1024]],
            value=512,
            description='Sample Resolution:',
            style={'description_width': 'initial'}
        )
        self.sample_seed = widgets.IntText(
            value=42,
            description='Sample Seed:',
            style={'description_width': 'initial'}
        )
        sample_gen_box = widgets.VBox([
            sample_gen_desc,
            self.sample_prompt,
            self.sample_num_images,
            self.sample_resolution,
            self.sample_seed
        ])

        # --- Training Options ---
        train_opt_desc = widgets.HTML("""<h3>▶️ Training Options</h3>
        <p>Select your optimizer, cross-attention mechanism, and precision. Caching latents can save memory. Enable V-Parameterization for SDXL v-pred models. Configure saving frequency and retention.</p>
        
        <div style='padding: 10px; border: 1px solid #856404; border-radius: 5px; margin: 10px 0;'>
        <strong>🧪 Optimizer Guide:</strong><br>
        • <strong>AdamW:</strong> Safe, reliable, works everywhere (recommended for beginners)<br>
        • <strong>AdamW8bit:</strong> Memory efficient, should work with proper CUDA setup<br>
        • <strong>CAME:</strong> Memory efficient, great for large models and low VRAM<br>
        • <strong>Prodigy:</strong> Adaptive learning rate, excellent results<br>
        • <strong>Lion:</strong> Fast and memory efficient<br>
        <em>Our environment fixes should resolve most compatibility issues! Try CAME or AdamW8bit for memory savings.</em>
        </div>""")
        self.optimizer = widgets.Dropdown(options=['AdamW', 'AdamW8bit', 'Prodigy', 'DAdaptation', 'DadaptAdam', 'DadaptLion', 'Lion', 'SGDNesterov', 'SGDNesterov8bit', 'AdaFactor', 'CAME'], value='AdamW', description='Optimizer:', style={'description_width': 'initial'})
        self.cross_attention = widgets.Dropdown(options=['sdpa', 'xformers'], value='sdpa', description='Cross Attention:', style={'description_width': 'initial'})
        self.precision = widgets.Dropdown(options=['fp16', 'bf16', 'float'], value='fp16', description='Precision:', style={'description_width': 'initial'})
        self.fp8_base = widgets.Checkbox(value=False, description="FP8 Base (experimental, requires PyTorch 2.1+)", indent=False)
        self.cache_latents = widgets.Checkbox(value=True, description="Cache Latents (saves memory)", indent=False)
        self.cache_latents_to_disk = widgets.Checkbox(value=True, description="Cache Latents to Disk (uses disk space, saves more memory)", indent=False)
        self.cache_text_encoder_outputs = widgets.Checkbox(value=False, description="Cache Text Encoder Outputs (disables text encoder training)", indent=False)
        self.v2 = widgets.Checkbox(value=False, description="SD 2.x Base Model (enable for SD 2.0/2.1 base models)", indent=False)
        self.v_parameterization = widgets.Checkbox(value=False, description="V-Parameterization (enable for SDXL v-pred models or SD 2.x 768px models)", indent=False)

        # SDXL-specific optimizations (highly recommended by Kohya SS docs)
        self.network_train_unet_only = widgets.Checkbox(value=False, description="🎯 Train U-Net Only (highly recommended for SDXL LoRA)", indent=False)

        # Saving options (moved from separate section)
        self.save_every_n_epochs = widgets.IntText(value=1, description='Save Every N Epochs:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))
        self.keep_only_last_n_epochs = widgets.IntText(value=5, description='Keep Last N Epochs:', style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))

        # Create unified training configuration box (combines basic, learning rate, and training options)
        training_config_box = widgets.VBox([
            training_config_desc,
            # Dataset and basic settings
            self.dataset_size, self.resolution, self.num_repeats, self.epochs, self.train_batch_size, self.step_calculator,
            self.flip_aug, self.shuffle_caption, self.keep_tokens, self.clip_skip,
            # Learning rate settings
            self.unet_lr, self.text_encoder_lr, self.lr_scheduler, self.lr_scheduler_number, self.lr_warmup_ratio,
            self.min_snr_gamma_enabled, self.min_snr_gamma, self.ip_noise_gamma_enabled, self.ip_noise_gamma,
            self.multinoise, self.noise_offset,
            # Training options
            self.optimizer, self.cross_attention, self.precision, self.fp8_base,
            self.cache_latents, self.cache_latents_to_disk, self.cache_text_encoder_outputs,
            self.v2, self.v_parameterization, self.network_train_unet_only, self.zero_terminal_snr, self.enable_bucket,
            self.gradient_checkpointing, self.gradient_accumulation_steps, self.max_grad_norm, self.full_fp16, self.random_crop,
            # Bucketing and VAE settings
            self.sdxl_bucket_optimization, self.min_bucket_reso, self.max_bucket_reso, self.bucket_no_upscale,
            self.vae_batch_size, self.no_half_vae,
            self.save_every_n_epochs, self.keep_only_last_n_epochs,
            # Config warnings at the end
            self.config_warnings
        ])

        # --- Advanced Training Options ---
        advanced_train_desc = widgets.HTML("""
        <h3>▶️ Advanced Training Options</h3>
        <p>Fine-tune caption handling, noise settings, and training stability options. These control how the model learns from your dataset.</p>
        
        <div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 10px 0;'>
        <h4>📚 Caption Controls Explained:</h4>
        <ul>
        <li><strong>Caption Dropout:</strong> Randomly removes entire captions to improve unconditional generation</li>
        <li><strong>Tag Dropout:</strong> Randomly removes individual tags to prevent overfitting to specific combinations</li>
        <li><strong>Keep Tokens:</strong> Always keeps the first N tokens (useful for trigger words)</li>
        </ul>
        
        <h4>🔊 Noise & Stability:</h4>
        <ul>
        <li><strong>Noise Offset:</strong> Adds brightness variation to improve dark/light image generation</li>
        <li><strong>Zero Terminal SNR:</strong> Removes bias in noise scheduler (recommended for SDXL)</li>
        <li><strong>Clip Skip:</strong> How many CLIP layers to skip (1-2 for anime, 2 for realistic)</li>
        </ul>
        </div>
        """)

        # Caption dropout controls
        self.caption_dropout_rate = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.5, step=0.05,
            description='Caption Dropout Rate:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        self.caption_tag_dropout_rate = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.5, step=0.05,
            description='Tag Dropout Rate:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        # Noise and stability controls (keep tokens and noise offset moved to basic settings)

        self.adaptive_noise_scale = widgets.FloatSlider(
            value=0.0, min=0.0, max=0.02, step=0.001,
            description='Adaptive Noise Scale:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        # Zero Terminal SNR moved to basic training options

        # Clip skip moved to basic settings

        # VAE and performance options
        self.vae_batch_size = widgets.IntSlider(
            value=1, min=1, max=8, step=1,
            description='VAE Batch Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        self.no_half_vae = widgets.Checkbox(
            value=False,
            description="No Half VAE (fixes some VAE issues, uses more VRAM)",
            indent=False
        )

        # Dataset bucketing controls
        self.bucket_reso_steps = widgets.IntSlider(
            value=64, min=32, max=128, step=32,
            description='Bucket Resolution Steps:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        self.min_bucket_reso = widgets.IntSlider(
            value=256, min=128, max=512, step=64,
            description='Min Bucket Resolution:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        self.max_bucket_reso = widgets.IntSlider(
            value=2048, min=1024, max=4096, step=512,
            description='Max Bucket Resolution:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        self.bucket_no_upscale = widgets.Checkbox(
            value=False,
            description="No Bucket Upscale (prevent upscaling small images)",
            indent=False
        )

        advanced_training_box = widgets.VBox([
            advanced_train_desc,
            widgets.HTML("<h4>📚 Caption Controls</h4>"),
            self.caption_dropout_rate,
            self.caption_tag_dropout_rate,
            widgets.HTML("<h4>🔊 Noise & Stability</h4>"),
            self.adaptive_noise_scale,
        ])

        # Saving options moved to Training Options section

        # --- Advanced Options (merged: Advanced Training + Advanced Mode) ---
        advanced_combined_box = self._create_combined_advanced_section(advanced_training_box)

        # --- Accordion ---
        accordion = widgets.Accordion(children=[
            project_box,
            training_config_box,
            lora_box,
            sample_gen_box,
            advanced_combined_box
        ])
        accordion.set_title(0, "▶️ Project Settings")
        accordion.set_title(1, "▶️ Training Configuration")
        accordion.set_title(2, "▶️ LoRA Structure")
        accordion.set_title(3, "▶️ Sample Generation Settings")
        accordion.set_title(4, "🚀 Additional Options")

        # Attach observers for real-time validation (after all widgets are created)
        self.cache_text_encoder_outputs.observe(check_config_conflicts, names='value')
        self.shuffle_caption.observe(check_config_conflicts, names='value')
        self.text_encoder_lr.observe(check_config_conflicts, names='value')
        self.random_crop.observe(check_config_conflicts, names='value')
        self.cache_latents.observe(check_config_conflicts, names='value')

        # New button to prepare/commit configuration
        self.prepare_config_button = widgets.Button(
            description="✅ Prepare Training Configuration",
            button_style='primary',
            layout=widgets.Layout(width='auto', height='40px')
        )
        self.prepare_config_button.on_click(self.run_training) # This will now trigger config collection

        # Progress description (was missing!)
        progress_desc = widgets.HTML("""<h3>📊 Training Progress</h3>
        <p>Monitor your training progress and status below. The system will automatically update with real-time information.</p>""")

        # Status bar widget (was also missing!)
        self.status_bar = widgets.HTML(value="<div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px;'><strong>📊 Status:</strong> Ready to configure training</div>")

        # Training output widget
        self.training_output = widgets.Output()

        self.widget_box = widgets.VBox([
            header_main,
            accordion,
            self.prepare_config_button, # Add the new button here
            progress_desc,
            self.status_bar,
            self.training_output
        ])

    def _update_status(self, message, status_type="info"):
        """Update the status bar with current training progress"""
        status_colors = {
            "info": "#007acc",      # Blue
            "success": "#28a745",   # Green
            "warning": "#ffc107",   # Yellow
            "error": "#dc3545",     # Red
            "progress": "#17a2b8"   # Teal
        }
        color = status_colors.get(status_type, "#007acc")

        self.status_bar.value = f"<div style='padding: 10px; border: 1px solid {color}; border-radius: 5px;'><strong>📊 Status:</strong> {message}</div>"

    def run_training(self, b):
        # Actually generate the TOML files when user clicks "Prepare"!
        config = self._get_training_config()
        try:
            self.manager.prepare_config_only(config)  # Generate TOML files only
            self._update_status("✅ Configuration TOML files generated! Now click 'Start LoRA Training' below!", "success")
        except Exception as e:
            self._update_status(f"❌ Failed to generate config: {str(e)}", "error")
            return

        # Get the existing monitor if available
        try:
            import __main__
            if hasattr(__main__, 'training_monitor'):
                monitor = __main__.training_monitor
            else:
                # Fallback: create and display monitor
                # The TrainingMonitorWidget now needs the TrainingManager instance
                monitor = TrainingMonitorWidget(training_manager_instance=self.manager)
                __main__.training_monitor = monitor  # Store globally
                display(monitor.widget_box)
        except Exception as e:
            # Fallback: create and display monitor
            print(f"Error getting/creating monitor: {e}")
            monitor = TrainingMonitorWidget(training_manager_instance=self.manager)
            display(monitor.widget_box)

        # Pass the training configuration to the monitor
        config = self._get_training_config()
        monitor.set_training_config(config)

        with self.training_output:
            self.training_output.clear_output()
        # The monitor's start button will now trigger the training via its own manager reference

    def _get_training_config(self):
        """Helper method to gather all config settings from widget values"""
        return {
            'project_name': self.project_name.value,
            'model_type': self.model_type.value.lower().replace('/', '_').replace('.', ''), # e.g. sd1.5/2.0 -> sd1_5_2_0
            'model_path': self.model_path.value,
            'clip_l_path': self.clip_l_path.value,
            'clip_g_path': self.clip_g_path.value,
            't5xxl_path': self.t5xxl_path.value,
            'dataset_path': self.dataset_dir.value,
            'continue_from_lora': self.continue_from_lora.value,
            'wandb_key': self.wandb_key.value,
            'resolution': self.resolution.value,
            'num_repeats': self.num_repeats.value,
            'epochs': self.epochs.value,
            'train_batch_size': self.train_batch_size.value,
            'flip_aug': self.flip_aug.value,
            'unet_lr': self._parse_learning_rate(self.unet_lr.value),
            'text_encoder_lr': self._parse_learning_rate(self.text_encoder_lr.value),
            'lr_scheduler': self.lr_scheduler.value,
            'lr_scheduler_number': self.lr_scheduler_number.value,
            'lr_warmup_ratio': self.lr_warmup_ratio.value,
            'min_snr_gamma_enabled': self.min_snr_gamma_enabled.value,
            'min_snr_gamma': self.min_snr_gamma.value,
            'ip_noise_gamma_enabled': self.ip_noise_gamma_enabled.value,
            'ip_noise_gamma': self.ip_noise_gamma.value,
            'multinoise': self.multinoise.value,
            'lora_type': self.lora_type.value,
            'network_dim': self.network_dim.value,
            'network_alpha': self.network_alpha.value,
            'conv_dim': self.conv_dim.value,
            'conv_alpha': self.conv_alpha.value,
            'sample_prompt': self.sample_prompt.value,
            'sample_num_images': self.sample_num_images.value,
            'sample_resolution': self.sample_resolution.value,
            'sample_seed': self.sample_seed.value,
            'optimizer': self.optimizer.value,
            'cross_attention': self.cross_attention.value,
            'precision': self.precision.value,
            'fp8_base': self.fp8_base.value,
            'cache_latents': self.cache_latents.value,
            'cache_latents_to_disk': self.cache_latents_to_disk.value,
            'cache_text_encoder_outputs': self.cache_text_encoder_outputs.value,
            'shuffle_caption': self.shuffle_caption.value,
            'v2': self.v2.value,
            'v_parameterization': self.v_parameterization.value,
            'network_train_unet_only': self.network_train_unet_only.value,
            'save_every_n_epochs': self.save_every_n_epochs.value,
            'keep_only_last_n_epochs': self.keep_only_last_n_epochs.value,
            # Advanced training options
            'caption_dropout_rate': self.caption_dropout_rate.value,
            'caption_tag_dropout_rate': self.caption_tag_dropout_rate.value,
            'keep_tokens': self.keep_tokens.value,
            'noise_offset': self.noise_offset.value,
            'adaptive_noise_scale': self.adaptive_noise_scale.value,
            'zero_terminal_snr': self.zero_terminal_snr.value,
            'clip_skip': self.clip_skip.value,
            'enable_bucket': self.enable_bucket.value,
            'gradient_checkpointing': self.gradient_checkpointing.value,
            'gradient_accumulation_steps': self.gradient_accumulation_steps.value,
            'max_grad_norm': self.max_grad_norm.value,
            'full_fp16': self.full_fp16.value,
            'random_crop': self.random_crop.value,
            'bucket_reso_steps': 32 if self.sdxl_bucket_optimization.value else 64,
            'min_bucket_reso': self.min_bucket_reso.value,
            'max_bucket_reso': self.max_bucket_reso.value,
            'bucket_no_upscale': self.bucket_no_upscale.value,
            'vae_batch_size': self.vae_batch_size.value,
            'no_half_vae': self.no_half_vae.value,
            # Advanced options
            'advanced_mode_enabled': getattr(self, 'advanced_mode', widgets.Checkbox(value=False)).value,
            'advanced_optimizer': getattr(self, 'advanced_optimizer', type('obj', (object,), {'value': 'standard'})).value,
            'advanced_scheduler': getattr(self, 'advanced_scheduler', type('obj', (object,), {'value': 'auto'})).value,
            'fused_back_pass': getattr(self, 'fused_back_pass', widgets.Checkbox(value=False)).value,
            'lycoris_method': getattr(self, 'lycoris_method', type('obj', (object,), {'value': 'none'})).value,
            'experimental_features': self._get_experimental_features(),
        }

    def _convert_to_structured_config(self, flat_config):
        """Convert flat widget config to structured TOML format"""
        
        # 🔍 DEBUG: What's actually in the flat config?
        print("🚨 === FLAT CONFIG DEBUG (BEFORE CONVERSION) ===")
        print(f"📊 All flat config keys: {list(flat_config.keys())}")
        print(f"📊 network_dim: {repr(flat_config.get('network_dim'))}")
        print(f"📊 network_alpha: {repr(flat_config.get('network_alpha'))}")
        print(f"📊 unet_lr: {repr(flat_config.get('unet_lr'))}")
        print(f"📊 model_path: {repr(flat_config.get('model_path'))}")
        print(f"📊 epochs: {repr(flat_config.get('epochs'))}")
        print(f"📊 optimizer: {repr(flat_config.get('optimizer'))}")
        print("🚨 === END FLAT CONFIG DEBUG ===")
        
        # Handle resolution formatting like our KohyaTrainingManager does
        resolution = flat_config.get('resolution')
        if isinstance(resolution, (int, str)):
            formatted_resolution = f"{resolution},{resolution}"
        else:
            formatted_resolution = "512,512"

        structured_config = {
            "network_arguments": {
                "network_dim": flat_config.get('network_dim'),
                "network_alpha": flat_config.get('network_alpha'),
                "network_module": "networks.lora",
            },
            "optimizer_arguments": {
                "learning_rate": flat_config.get('unet_lr'),
                "text_encoder_lr": flat_config.get('text_encoder_lr'),
                "lr_scheduler": flat_config.get('lr_scheduler'),
                "optimizer_type": flat_config.get('optimizer'),
            },
            "training_arguments": {
                "pretrained_model_name_or_path": flat_config.get('model_path'),
                "max_train_epochs": flat_config.get('epochs'),
                "train_batch_size": flat_config.get('train_batch_size'),
                "save_every_n_epochs": flat_config.get('save_every_n_epochs'),
                "mixed_precision": flat_config.get('precision'),
                "output_dir": "output",
                "output_name": flat_config.get('project_name', 'lora'),
                "clip_skip": flat_config.get('clip_skip', 2),
                "save_model_as": "safetensors",
                "seed": 42,
            },
            "datasets": [{
                "subsets": [{
                    "image_dir": flat_config.get('dataset_path'),
                    "num_repeats": flat_config.get('num_repeats'),
                }]
            }],
            "general": {
                "resolution": formatted_resolution,
                "shuffle_caption": flat_config.get('shuffle_caption'),
                "flip_aug": flat_config.get('flip_aug'),
            },
            # Include widget-only fields for monitor widget compatibility
            "sample_prompt": flat_config.get('sample_prompt'),
            "sample_num_images": flat_config.get('sample_num_images'),
            "sample_resolution": flat_config.get('sample_resolution'),
            "sample_seed": flat_config.get('sample_seed'),
            "dataset_size": flat_config.get('dataset_size'),  # For step calculation
            "epochs": flat_config.get('epochs'),  # For total epochs calculation
        }
        return structured_config

    def _create_combined_advanced_section(self, advanced_training_box):
        """Create combined advanced section merging Advanced Training Options with Advanced Mode"""
        combined_desc = widgets.HTML("""<h3>🧪 Advanced Options</h3>
        <p>Advanced training controls, experimental features, and optimization settings. Use with caution - these can significantly impact training behavior.</p>""")

        # Get the original advanced mode content
        original_advanced = self._create_advanced_section()

        # Combine with advanced training options
        return widgets.VBox([combined_desc, advanced_training_box, original_advanced])

    def _create_advanced_section(self):
        """Creates the Advanced Mode section with educational explanations"""

        # Advanced Mode Toggle
        advanced_header = widgets.HTML("""
        <h3>🧪 Advanced Training Mode</h3>
        <p><strong>⚠️ For experienced users only!</strong> These features are experimental and may require VastAI or high-end hardware.</p>
        """)

        self.advanced_mode = widgets.Checkbox(
            value=False,
            description="🧪 Show More Training Options",
            style={'description_width': 'initial'}
        )

        # Advanced options container (initially hidden)
        self.advanced_container = widgets.VBox([
            self._create_advanced_optimizer_section(),
            self._create_memory_optimization_section(),
            self._create_lycoris_advanced_section(),
            self._create_experimental_section()
        ])

        # Initially hide advanced options
        self.advanced_container.layout.display = 'none'

        # Show/hide based on toggle
        def toggle_advanced_mode(change):
            if change['new']:
                self.advanced_container.layout.display = 'block'
                self._show_advanced_warning()
            else:
                self.advanced_container.layout.display = 'none'

        self.advanced_mode.observe(toggle_advanced_mode, names='value')

        return widgets.VBox([
            advanced_header,
            self.advanced_mode,
            self.advanced_container
        ])

    def _create_advanced_optimizer_section(self):
        """Advanced optimizers with educational explanations"""

        optimizer_info = widgets.HTML("""
        <h4>🚀 Advanced Optimizers</h4>
        <p><strong>Choose your optimization algorithm:</strong></p>
        """)

        self.advanced_optimizer = widgets.Dropdown(
            options=[
                ('Standard (Use basic options)', 'standard'),
                ('CAME - Memory Efficient', 'came'),
                ('Prodigy Plus - Schedule Free', 'prodigy_plus'),
                ('StableAdamW - Experimental', 'stable_adamw'),
                ('ADOPT - Research Grade', 'adopt')
            ],
            value='standard',
            description='Optimizer:',
            style={'description_width': 'initial'}
        )

        # Dynamic explanation based on selection
        self.optimizer_explanation = widgets.HTML()

        def update_optimizer_explanation(change):
            explanations = {
                'standard': """
                <div style='padding: 10px; border: 1px solid #007acc; border-radius: 5px;'>
                <strong>Standard Mode:</strong> Uses your basic optimizer selection above.<br>
                ✅ Safe and well-tested<br>
                ✅ Good for beginners
                </div>
                """,
                'came': """
                <div style='padding: 10px; border: 1px solid #28a745; border-radius: 5px;'>
                <strong>CAME (Derrian's Advanced):</strong> Memory-efficient optimizer from Derrian Distro.<br>
                ✅ Uses 30-40% less VRAM than AdamW<br>
                ✅ Often produces high-quality results<br>
                ✅ Auto-pairs with REX scheduler + Huber loss<br>
                ❌ Newer, less community testing<br>
                <em>🎯 Best for: VRAM-constrained training (8GB cards)</em>
                </div>
                """,
                'prodigy_plus': """
                <div style='padding: 10px; border: 1px solid #ffc107; border-radius: 5px;'>
                <strong>Prodigy Plus (OneTrainer):</strong> Learning rate AND schedule free!<br>
                ✅ No learning rate tuning needed<br>
                ✅ No scheduler needed<br>
                ✅ Memory optimizations included<br>
                ❌ Very new, experimental<br>
                <em>🎯 Best for: Users who hate hyperparameter tuning</em>
                </div>
                """,
                'stable_adamw': """
                <div style='padding: 10px; border: 1px solid #856404; border-radius: 5px;'>
                <strong>StableAdamW (Experimental):</strong> Research-grade stability improvements.<br>
                ✅ Better convergence stability<br>
                ✅ Handles difficult datasets better<br>
                ❌ Very experimental<br>
                ❌ May not work with all models<br>
                <em>⚠️ For research and experimentation only</em>
                </div>
                """,
                'adopt': """
                <div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px;'>
                <strong>ADOPT (Bleeding Edge):</strong> Adaptive gradient clipping research.<br>
                ✅ Potential for breakthrough results<br>
                ❌ Highly experimental<br>
                ❌ May crash or fail<br>
                ❌ No guarantees<br>
                <em>🔬 For AI researchers and risk-takers only!</em>
                </div>
                """
            }
            self.optimizer_explanation.value = explanations.get(change['new'], '')

            # Auto-update scheduler recommendations
            self._update_scheduler_recommendations(change['new'])

        self.advanced_optimizer.observe(update_optimizer_explanation, names='value')

        return widgets.VBox([
            optimizer_info,
            self.advanced_optimizer,
            self.optimizer_explanation
        ])

    def _create_memory_optimization_section(self):
        """Memory optimization techniques"""

        memory_info = widgets.HTML("""
        <h4>💾 Memory Wizardry</h4>
        <p><strong>Advanced VRAM reduction techniques:</strong></p>
        """)

        self.fused_back_pass = widgets.Checkbox(
            value=False,
            description="🚧 Fused Back Pass (Requires OneTrainer - Coming Soon)",
            style={'description_width': 'initial'},
            disabled=True  # Disable until OneTrainer integration
        )

        fused_explanation = widgets.HTML("""
        <div style='padding: 10px; border: 1px solid #856404; border-radius: 5px;'>
        <strong>🚧 Fused Back Pass - OneTrainer Integration Required</strong><br><br>
        <strong>What it would do:</strong><br>
        📊 Calculate gradient → ⚡ Update immediately → 🗑️ Free VRAM → 🔄 Next layer<br><br>
        <strong>Why it's disabled:</strong><br>
        ❌ Requires OneTrainer's custom training loop implementation<br>
        ❌ Cannot be added as simple config flag to SD scripts<br>
        ❌ Needs fundamental changes to gradient handling<br><br>
        <strong>🔮 Future Plans:</strong><br>
        • Integrate OneTrainer as optional backend<br>
        • Add backend switcher (SD Scripts vs OneTrainer)<br>
        • Enable advanced memory optimizations<br><br>
        <em>🎯 For now: Use gradient checkpointing + cache settings for VRAM optimization</em>
        </div>
        """)

        return widgets.VBox([
            memory_info,
            self.fused_back_pass,
            fused_explanation
        ])

    def _create_lycoris_advanced_section(self):
        """Advanced LyCORIS methods"""

        lycoris_info = widgets.HTML("""
        <h4>🦄 LyCORIS Advanced Methods</h4>
        <p><strong>Beyond standard LoRA - cutting-edge adaptation techniques:</strong></p>
        """)

        self.lycoris_method = widgets.Dropdown(
            options=[
                ('None (Use Main LoRA Type)', 'none'),
                ('BOFT - Butterfly Transform', 'boft')
            ],
            value='none',
            description='Advanced LyCORIS:',
            style={'description_width': 'initial'}
        )

        self.lycoris_explanation = widgets.HTML()

        def update_lycoris_explanation(change):
            explanations = {
                'none': """
                <div style='padding: 10px; border: 1px solid #007acc; border-radius: 5px;'>
                <strong>Standard LoRA:</strong> The classic, reliable choice.<br>
                ✅ Well-tested and stable<br>
                ✅ Fast training<br>
                ✅ Universal compatibility
                </div>
                """,
                'dora': """
                <div style='padding: 10px; border: 1px solid #28a745; border-radius: 5px;'>
                <strong>DoRA (Weight Decomposition):</strong> Trains like full fine-tune!<br>
                ✅ Much higher quality than standard LoRA<br>
                ✅ Better coherency and detail preservation<br>
                ✅ Especially good for faces and complex scenes<br>
                ❌ 2-3x slower training<br>
                ❌ More complex to tune<br>
                <em>🎯 Worth it for: High-quality character/style LoRAs</em>
                </div>
                """,
                'lokr': """
                <div style='padding: 10px; border: 1px solid #ffc107; border-radius: 5px;'>
                <strong>LoKr (Kronecker Product):</strong> Mathematical efficiency master.<br>
                ✅ Better parameter efficiency than standard LoRA<br>
                ✅ Can achieve same quality with smaller file sizes<br>
                ✅ Good for concept learning<br>
                ❌ More sensitive to hyperparameters<br>
                <em>🎯 Best for: Concept LoRAs and style transfer</em>
                </div>
                """,
                'ia3': """
                <div style='padding: 10px; border: 1px solid #856404; border-radius: 5px;'>
                <strong>(IA)³ (Implicit Attention):</strong> Attention-focused adaptation.<br>
                ✅ Very parameter efficient<br>
                ✅ Good for style and lighting changes<br>
                ✅ Fast training<br>
                ❌ Limited for complex content changes<br>
                <em>🎯 Perfect for: Style LoRAs and lighting adjustments</em>
                </div>
                """
            }
            self.lycoris_explanation.value = explanations.get(change['new'], '')

        self.lycoris_method.observe(update_lycoris_explanation, names='value')

        return widgets.VBox([
            lycoris_info,
            self.lycoris_method,
            self.lycoris_explanation
        ])

    def _create_experimental_section(self):
        """Experimental features section"""

        experimental_info = widgets.HTML("""
        <h4>🔬 Experimental Lab</h4>
        <p><strong>⚠️ Dragons be here! Use at your own risk:</strong></p>
        """)

        self.experimental_options = widgets.VBox([
            widgets.Checkbox(
                value=False,
                description="⚡ Adversarial Loss (Research)",
                style={'description_width': 'initial'},
                disabled=True  # Not implemented yet
            ),
            widgets.Checkbox(
                value=False,
                description="🌊 Multi-Resolution Training",
                style={'description_width': 'initial'}
            ),
            widgets.HTML("""
            <div style='padding: 8px; border: 1px solid #856404; border-radius: 5px;'>
            <strong>🚧 Work in Progress:</strong><br>
            • Adversarial Loss: GAN-style training improvements<br>
            • Multi-Res: Train on multiple resolutions simultaneously<br><br>
            <em>These will be enabled as they become stable!</em>
            </div>
            """)
        ])

        return widgets.VBox([
            experimental_info,
            self.experimental_options
        ])

    def _update_scheduler_recommendations(self, optimizer):
        """Update scheduler recommendations based on optimizer choice"""
        if hasattr(self, 'advanced_scheduler'):
            recommendations = {
                'came': 'rex',
                'prodigy_plus': 'constant',  # Schedule-free
                'standard': 'cosine'
            }

            recommended = recommendations.get(optimizer, 'cosine')
            if recommended in [option[1] for option in self.lr_scheduler.options]:
                self.lr_scheduler.value = recommended

    def _show_advanced_warning(self):
        """Show warning when advanced mode is enabled"""
        warning = widgets.HTML("""
        <div style='padding: 15px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'>
        <strong>⚠️ ADVANCED MODE ACTIVATED!</strong><br><br>
        You've entered the experimental zone! These features are:
        <ul>
        <li>🔬 <strong>Cutting-edge:</strong> May be unstable or break</li>
        <li>🚀 <strong>VastAI optimized:</strong> Some need powerful hardware</li>
        <li>🧠 <strong>Research-grade:</strong> Results may vary wildly</li>
        <li>💀 <strong>No guarantees:</strong> Backup your work!</li>
        </ul>
        <em>"Either gonna work or blow up!" - You asked for it! 😄</em>
        </div>
        """)

        # Add warning to the container
        if len(self.advanced_container.children) == 4:  # Only add once
            self.advanced_container.children = [warning] + list(self.advanced_container.children)

    def _get_experimental_features(self):
        """Collect experimental feature settings"""
        if hasattr(self, 'experimental_options'):
            # Extract checkbox values from experimental options
            features = {}
            for i, child in enumerate(self.experimental_options.children):
                if hasattr(child, 'value') and hasattr(child, 'description'):
                    features[f'experimental_{i}'] = child.value
            return features
        return {}

    def _refresh_model_list(self):
        """Scan pretrained_model directory and populate dropdown"""
        try:
            import glob

            # Look for models in common locations
            search_paths = [
                "pretrained_model/*.safetensors",
                "pretrained_model/*.ckpt",
                "pretrained_model/*.pth",
                "models/*.safetensors",
                "models/*.ckpt",
                "*/pretrained_model/*.safetensors",  # Check subdirectories
            ]

            found_models = []
            for pattern in search_paths:
                found_models.extend(glob.glob(pattern))

            # Remove duplicates and sort
            found_models = sorted(list(set(found_models)))

            if found_models:
                # Create dropdown options with friendly names
                options = [('Select a model...', '')]
                for model_path in found_models:
                    model_name = os.path.basename(model_path)
                    # Truncate long names for dropdown display
                    display_name = model_name if len(model_name) <= 50 else model_name[:47] + "..."
                    options.append((display_name, model_path))

                self.model_dropdown.options = options
                print(f"✅ Found {len(found_models)} models in pretrained_model directory")
            else:
                self.model_dropdown.options = [('No models found - use custom path below', '')]
                print("📁 No models found in pretrained_model/ directory")
                print("💡 Place your .safetensors/.ckpt files in pretrained_model/ folder or use custom path")

        except Exception as e:
            print(f"⚠️ Error scanning for models: {e}")
            self.model_dropdown.options = [('Error scanning - use custom path', '')]

    def display(self):
        display(self.widget_box)

        # Make this training widget globally available for the monitor widget
        import __main__
        __main__.training_widget = self
