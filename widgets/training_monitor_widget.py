# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/training_monitor_widget.py
import os
import re
import threading
import time

import ipywidgets as widgets
from IPython.display import Image as IPImage
from IPython.display import display

# TEMPORARILY DISABLED: Inference functionality
# from core.inference_utils import generate_sample_images
from shared_managers import get_config_manager


class TrainingMonitorWidget:
    def __init__(self, training_manager_instance):
        self.training_manager = training_manager_instance
        self.training_config = None # To store the config passed from TrainingWidget

        # TEMPORARILY DISABLED: Sidecar functionality for inference
        # Initialize sidecars for different purposes
        # try:
        #     from sidecar import Sidecar
        #     self.sample_sidecar = Sidecar(title='🎨 Training Sample Images', anchor='split-right')
        #     self.inference_sidecar = Sidecar(title='🎨 LoRA Test Inference', anchor='split-bottom')
        # except ImportError:
        #     print("⚠️ Sidecar not available. Images will display in main notebook.")
        self.sample_sidecar = None
        self.inference_sidecar = None

        self.create_widgets()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.training_phase = "Initializing..."
        self._last_checkpoint_time = None
        self._completion_check_timer = None
        self._is_final_epoch = False

        # Inference parameters
        self.sample_prompt = ""
        self.sample_num_images = 3  # Generate 3 images every epoch
        self.sample_resolution = 512  # Will be auto-detected based on model type
        self.sample_seed = 42
        self.base_model_path = ""
        self.output_dir = ""

    def create_widgets(self):
        """Create the training monitor interface with accordion structure"""

        # Header for the entire widget
        main_header = widgets.HTML("<h2>📈 Training Progress & Control</h2>")

        # Create accordion sections
        self.create_training_control_section()
        self.create_progress_monitoring_section()
        # TEMPORARILY DISABLED: Sample viewing/inference section
        # self.create_sample_viewing_section()

        # Create accordion (INFERENCE SECTION TEMPORARILY REMOVED)
        self.accordion = widgets.Accordion(children=[
            self.training_control_box,
            self.progress_monitoring_box,
            # self.sample_viewing_box  # TEMPORARILY DISABLED
        ])
        self.accordion.set_title(0, "🚀 Start Training")
        self.accordion.set_title(1, "📊 Live Progress Monitor")
        # self.accordion.set_title(2, "🖼️ View Training Samples")  # TEMPORARILY DISABLED

        # Main widget container
        self.widget_box = widgets.VBox([main_header, self.accordion])

    def create_training_control_section(self):
        """Create the training control section with start button"""
        control_desc = widgets.HTML("""<h3>🚀 Training Control</h3>
        <p>Start your LoRA training here. Configure your settings in the Training Configuration section above first!</p>
        <div style='padding: 10px; border: 1px solid #007acc; border-radius: 5px; margin: 10px 0;'>
        <strong>📋 Before Starting:</strong><br>
        • Configure your training parameters in the <strong>Training Configuration</strong> section above<br>
        • Verify your dataset directory and model paths are correct<br>
        • Check your learning rates and training steps look reasonable<br>
        • Make sure you have enough disk space and VRAM
        </div>""")

        # Start training button
        self.start_training_button = widgets.Button(
            description="🚀 Start LoRA Training",
            button_style='success',
            layout=widgets.Layout(width='300px', height='50px')
        )

        self.stop_training_button = widgets.Button(
            description="🛑 Emergency Stop",
            button_style='danger',
            layout=widgets.Layout(width='300px', height='50px')
        )
        self.stop_training_button.on_click(self.stop_training_clicked)
        self.stop_training_button.disabled = True

        # Training status
        self.control_status = widgets.HTML(
            value="<div style='padding: 10px; border: 1px solid #6c757d; border-radius: 5px; margin: 10px 0;'>"
                  "<strong>Status:</strong> Ready to start training. Click the button above when you're ready!</div>"
        )

        # Hook up the button (we'll connect this to the training widget later)
        self.start_training_button.on_click(self.start_training_clicked)

        self.training_control_box = widgets.VBox([
            control_desc,
            widgets.HBox([self.start_training_button, self.stop_training_button]),
            self.control_status
        ])

    def create_progress_monitoring_section(self):
        """Create the progress monitoring section"""

        # Progress header
        progress_desc = widgets.HTML("<h3>📊 Live Training Progress</h3><p>Real-time monitoring will appear here when training starts.</p>")

        # Training Phase Status (fix theme colors)
        self.phase_status = widgets.HTML(
            value="<div style='padding: 15px; border: 1px solid #007acc; border-radius: 8px;'>"
                  "<strong>📊 Phase:</strong> Waiting to start...</div>"
        )

        # Progress Bars
        self.epoch_progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Epoch:',
            bar_style='info',
            style={'bar_color': '#007acc', 'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )

        self.step_progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Step:',
            bar_style='success',
            style={'bar_color': '#28a745', 'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )

        # Progress Labels
        self.epoch_label = widgets.HTML(value="<strong>Epoch:</strong> Not started")
        self.step_label = widgets.HTML(value="<strong>Step:</strong> Not started")

        # Resource Monitoring (optional placeholders)
        self.resource_info = widgets.HTML(
            value="<p><strong>💾 Resources:</strong> Monitoring will begin when training starts</p>"
        )

        # Training Log Output
        self.training_log = widgets.Output(
            layout=widgets.Layout(
                height='400px',
                overflow='scroll',
                border='1px solid #ddd',
                margin='10px 0'
            )
        )

        # Sample generation button - REMOVED (inference disabled)

        # Auto-save status (fix theme colors)
        self.autosave_status = widgets.HTML(
            value="<div style='padding: 8px; border: 1px solid #28a745; border-radius: 5px;'>"
                  "<strong>💾 Auto-save:</strong> Enabled - checkpoints saved each epoch</div>"
        )

        # Create the progress monitoring box
        self.progress_monitoring_box = widgets.VBox([
            progress_desc,
            self.phase_status,
            self.epoch_label,
            self.epoch_progress,
            self.step_label,
            self.step_progress,
            self.resource_info,
            widgets.HTML("<h4>📋 Training Log</h4>"),
            self.training_log,
            self.autosave_status
        ])

    def create_sample_viewing_section(self):
        """Create automatic training sample viewing section"""
        viewing_desc = widgets.HTML("""<h3>🖼️ View Training Samples</h3>
        <p>View automatically generated sample images from each training epoch. Samples appear in the sidecar panel.</p>
        <div style='padding: 10px; border: 1px solid #6f42c1; border-radius: 5px; margin: 10px 0;'>
        <strong>📊 Sample Information:</strong><br>
        • Samples are generated automatically at the end of each epoch<br>
        • Uses the sample prompt configured in training settings<br>
        • Images are saved to the output directory<br>
        • Click buttons below to view samples in sidecar
        </div>""")

        # Current sample info display
        self.current_sample_info = widgets.HTML(
            value="<p><em>No samples generated yet. Start training to see automatic samples.</em></p>"
        )

        # View buttons
        self.view_latest_samples_button = widgets.Button(
            description="👁️ View Latest Epoch Samples",
            button_style='info',
            layout=widgets.Layout(width='220px', height='40px'),
            tooltip="View samples from the most recent completed epoch"
        )
        self.view_latest_samples_button.on_click(self._on_view_latest_samples)

        self.view_all_samples_button = widgets.Button(
            description="📁 View All Training Samples",
            button_style='',
            layout=widgets.Layout(width='220px', height='40px'),
            tooltip="Browse all generated samples from all epochs"
        )
        self.view_all_samples_button.on_click(self._on_view_all_samples)

        self.refresh_samples_button = widgets.Button(
            description="🔄 Refresh Sample Display",
            button_style='warning',
            layout=widgets.Layout(width='220px', height='40px'),
            tooltip="Refresh the sidecar display with latest samples"
        )
        self.refresh_samples_button.on_click(self._on_refresh_samples)

        button_row = widgets.HBox([
            self.view_latest_samples_button,
            self.view_all_samples_button,
            self.refresh_samples_button
        ])

        # Sample viewing output
        self.sample_viewing_output = widgets.Output()

        self.sample_viewing_box = widgets.VBox([
            viewing_desc,
            self.current_sample_info,
            button_row,
            self.sample_viewing_output
        ])

    def set_training_config(self, config):
        """Set the training configuration received from TrainingWidget (flat or structured format)"""
        self.training_config = config

        # Handle both flat widget config and structured TOML config format
        if self._is_structured_config(config):
            # Extract from structured TOML format
            training_args = config.get('training_arguments', {})
            datasets = config.get('datasets', [{}])[0].get('subsets', [{}])[0] if config.get('datasets') else {}
            general = config.get('general', {})
            
            # Store inference parameters from structured config
            self.sample_prompt = config.get('sample_prompt', '')  # This comes from widget, not TOML
            self.sample_num_images = config.get('sample_num_images', 0)
            self.sample_resolution = config.get('sample_resolution', 512)
            self.sample_seed = config.get('sample_seed', 42)
            self.base_model_path = training_args.get('pretrained_model_name_or_path', '')
            self.output_dir = training_args.get('output_dir', '')
            
            # Extract dataset info from structured format
            num_repeats = datasets.get('num_repeats', 1)
            batch_size = training_args.get('train_batch_size', 1)
        else:
            # Legacy flat format support
            self.sample_prompt = config.get('sample_prompt', '')
            self.sample_num_images = config.get('sample_num_images', 0)
            self.sample_resolution = config.get('sample_resolution', 512)
            self.sample_seed = config.get('sample_seed', 42)
            self.base_model_path = config.get('model_path', '')
            self.output_dir = config.get('output_dir', '')
            
            # Legacy dataset info extraction
            num_repeats = config.get('num_repeats', 1)
            batch_size = config.get('train_batch_size', 1)

        # Calculate steps per epoch for accurate progress tracking
        if config:
            try:
                # Get dataset info
                dataset_size = config.get('dataset_size', 100)  # fallback (not in TOML, comes from widget)

                # Calculate steps per epoch: (images * repeats) / batch_size
                self.steps_per_epoch = max(1, (dataset_size * num_repeats) // batch_size)
                
                # Get total epochs from structured or flat config
                if self._is_structured_config(config):
                    self.total_epochs = config.get('training_arguments', {}).get('max_train_epochs', 1)
                else:
                    self.total_epochs = config.get('epochs', 1)

                print(f"📊 Calculated: {self.steps_per_epoch} steps per epoch for {dataset_size} images")
            except Exception as e:
                print(f"⚠️ Could not calculate steps per epoch: {e}")
                self.steps_per_epoch = 100  # Safe fallback

    def _is_structured_config(self, config):
        """Check if config is structured TOML format"""
        return any(section in config for section in ['network_arguments', 'optimizer_arguments', 'training_arguments'])

    def start_training_clicked(self, b):
        """Handle start training button click - DEAD SIMPLE FILE HUNTING APPROACH"""
        try:
            # Create our brilliant file hunter
            config_mgr = get_config_manager()

            # Step 1: Hunt for TOML files like a boss
            if not config_mgr.files_ready():
                self.control_status.value = "<div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'><strong>❌ Error:</strong> Config files not ready! Click 'Prepare Training Configuration' first!</div>"
                return

            # Step 2: Files found? LET'S FUCKING GOOOOO! 🚀
            self.control_status.value = "<div style='padding: 10px; border: 1px solid #28a745; border-radius: 5px; margin: 10px 0;'><strong>Status:</strong> ✅ Config files found! Starting training! 🚀</div>"
            self.accordion.selected_index = 1  # Switch to progress tab

            # Step 3: Get the file paths and launch
            config_paths = config_mgr.get_config_paths()
            print(f"🔍 Found config files: {list(config_paths.keys())}")

            # Launch training using the file paths
            self.training_manager.launch_from_files(config_paths, monitor_widget=self)
            self.start_training_button.disabled = True
            self.stop_training_button.disabled = False

        except Exception as e:
            self.control_status.value = f"<div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'><strong>💥 Error:</strong> {str(e)}</div>"
            print(f"💥 Training start error: {e}")

    def stop_training_clicked(self, b):
        """Handle stop training button click"""
        self.control_status.value = "<div style='padding: 10px; border: 1px solid #dc3545; border-radius: 5px; margin: 10px 0;'><strong>Status:</strong> 🛑 Terminating training...</div>"
        self.training_manager.stop_training()
        self.start_training_button.disabled = False
        self.stop_training_button.disabled = True

    # Removed duplicate display method - main one is at end of class

    def update_phase(self, phase_text, phase_type="info"):
        """Update the current training phase"""
        phase_colors = {
            "info": "#007acc",
            "warning": "#ffc107",
            "success": "#28a745",
            "error": "#dc3545"
        }
        color = phase_colors.get(phase_type, "#007acc")

        self.phase_status.value = (
            f"<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid {color};'>"
            f"<strong>📊 Phase:</strong> {phase_text}</div>"
        )

    def update_epoch_progress(self, current_epoch, total_epochs):
        """Update epoch progress"""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        if total_epochs > 0:
            progress_pct = int((current_epoch / total_epochs) * 100)
            self.epoch_progress.value = progress_pct
            self.epoch_label.value = (
                f"<strong>Epoch:</strong> {current_epoch}/{total_epochs} "
                f"({progress_pct}%)"
            )

    def update_step_progress(self, current_step, total_steps_in_epoch):
        """Update step progress within current epoch"""
        self.current_step = current_step
        self.total_steps = total_steps_in_epoch

        if total_steps_in_epoch > 0:
            progress_pct = int((current_step / total_steps_in_epoch) * 100)
            self.step_progress.value = progress_pct
            self.step_label.value = (
                f"<strong>Step:</strong> {current_step}/{total_steps_in_epoch} "
                f"({progress_pct}%)"
            )

    def update_resources(self, gpu_usage=None, ram_usage=None, gpu_temp=None):
        """Update resource monitoring info"""
        resource_parts = []

        if gpu_usage:
            resource_parts.append(f"🎮 GPU: {gpu_usage}")
        if ram_usage:
            resource_parts.append(f"💾 RAM: {ram_usage}")
        if gpu_temp:
            resource_parts.append(f"🌡️ Temp: {gpu_temp}")

        if resource_parts:
            resource_text = " | ".join(resource_parts)
            self.resource_info.value = (
                f"<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace;'>"
                f"<strong>💾 Resources:</strong> {resource_text}</div>"
            )

    def parse_training_output(self, line):
        """Parse training output and update progress accordingly"""
        line = line.strip()

        with self.training_log:
            print(line)

        # Phase detection
        if "prepare optimizer, data loader etc." in line.lower():
            self.update_phase("Setting up optimizer and data loader...", "info")
        elif "caching latents" in line.lower():
            self.update_phase("Caching latents to disk...", "info")
        elif "enable bucket" in line.lower() or "make buckets" in line.lower():
            self.update_phase("Creating resolution buckets...", "info")
        elif "start training" in line.lower():
            self.update_phase("Training started!", "success")
        elif "epoch" in line.lower() and ("step" in line.lower() or "epoch is incremented" in line.lower()):
            # PRIORITY: Epoch detection ALWAYS overrides other phases (like "Saving checkpoint")
            # Parse Kohya's dual-epoch format: "current_epoch: 0, epoch: 3"
            current_epoch_match = re.search(r'current_epoch:\s*(\d+)', line)
            epoch_match = re.search(r'epoch:\s*(\d+)', line)
            step_match = re.search(r'step[:\s]+(\d+)', line.lower())
            
            # Use the more reliable epoch number (the one starting)
            if epoch_match:
                epoch_starting = int(epoch_match.group(1))
                
                # Check if we're starting the final epoch
                if epoch_starting >= self.total_epochs and self.total_epochs > 0:
                    self.update_phase(f"Starting final epoch ({epoch_starting}/{self.total_epochs})!", "warning")
                    self._is_final_epoch = True
                elif self.total_epochs > 0:
                    self.update_phase(f"Training - Epoch {epoch_starting}/{self.total_epochs}", "success")
                    self._is_final_epoch = False
                
                # Update epoch progress using the starting epoch number
                if self.total_epochs > 0:
                    self.update_epoch_progress(epoch_starting, self.total_epochs)
                    
            # Handle step progress
            if step_match:
                current_step = int(step_match.group(1))
                # Calculate total steps from training config if available
                if self.total_epochs > 0 and hasattr(self, 'steps_per_epoch'):
                    total_steps = self.total_epochs * self.steps_per_epoch
                else:
                    # Try to extract from line or fall back to current_step * 2 as minimum
                    total_steps = max(current_step * 2, 100)
                self.update_step_progress(current_step, total_steps)

        elif "saving checkpoint" in line.lower() or "saved" in line.lower():
            # Check if this is the final checkpoint save
            if self._is_final_epoch:
                self.update_phase("Saving final checkpoint - Training completing! 🎉", "success")
                # Cancel any existing timer since we know this is the end
                if self._completion_check_timer:
                    self._completion_check_timer.cancel()
                self._training_completed()
            else:
                self.update_phase("Checkpoint saved ✅ - Continuing training...", "success")  # Brief success message that doesn't get stuck
            
        elif ("training complete" in line.lower() or "finished" in line.lower() or 
              "training finished" in line.lower() or "done" in line.lower() or
              "training done" in line.lower() or "completed" in line.lower() or
              "end of training" in line.lower() or "training ended" in line.lower()):
            # Cancel completion timer since we got explicit completion
            if self._completion_check_timer:
                self._completion_check_timer.cancel()
            self.update_phase("Training completed successfully! 🎉", "success")
            self._training_completed()
            
        # Better error vs warning detection
        elif any(phrase in line.lower() for phrase in [
            "training failed", "fatal error", "cuda out of memory", "error:", "exception:", 
            "traceback", "failed to start", "cannot load", "file not found", "permission denied"
        ]):
            # These are actual errors that should stop training
            if self._completion_check_timer:
                self._completion_check_timer.cancel()
            self.update_phase("Training error occurred", "error")
        elif any(phrase in line.lower() for phrase in [
            "warning:", "deprecated", "futurewarning", "userwarning", "ignore", 
            "fallback", "not found (ignoring)"
        ]):
            # These are warnings - don't change phase, just note them
            pass  # Let training continue, warnings are normal

    def _check_training_completion(self):
        """Check if training has completed based on time since last checkpoint"""
        try:
            # This runs 30 seconds after the last checkpoint save
            # If no new logs have appeared, assume training completed
            self.update_phase("Training completed successfully! 🎉", "success")
            self._training_completed()
        except Exception as e:
            print(f"⚠️ Error in completion check: {e}")

    def _training_completed(self):
        """Handle training completion - re-enable buttons"""
        try:
            self.start_training_button.disabled = False
            self.stop_training_button.disabled = True
            # Update control status
            self.control_status.value = (
                "<div style='padding: 10px; border: 1px solid #28a745; border-radius: 5px; margin: 10px 0;'>"
                "<strong>Status:</strong> 🎉 Training completed! You can start a new training session.</div>"
            )
        except Exception as e:
            print(f"⚠️ Error in training completion handler: {e}")

    def clear_log(self):
        """Clear the training log"""
        with self.training_log:
            self.training_log.clear_output()

    def _get_resolution_for_model_type(self, model_path):
        """Auto-detect resolution based on model type"""
        if not model_path:
            return 512  # Safe default

        model_name = os.path.basename(model_path).lower()

        # SDXL detection
        if 'sdxl' in model_name or 'xl' in model_name:
            return 1024

        # SD 2.x detection
        if 'sd2' in model_name or 'stable-diffusion-2' in model_name or 'v2-' in model_name:
            return 768  # Whatever the fuck for 2.x 😂

        # Flux detection
        if 'flux' in model_name:
            return 1024  # Whatever the fuck for Flux

        # SD3 detection
        if 'sd3' in model_name:
            return 1024  # Whatever the fuck for SD3

        # Default to SD 1.5
        return 512

    def generate_and_display_samples(self, callback=None):
        """Generate sample images and display them in sidecar"""
        try:
            if not self.sample_prompt.strip():
                print("⚠️ No sample prompt set. Skipping image generation.")
                if callback:
                    callback()
                return

            # Auto-detect resolution
            resolution = self._get_resolution_for_model_type(self.base_model_path)

            print(f"🎨 Generating {self.sample_num_images} sample images at {resolution}x{resolution}...")

            # Generate sample images (using existing inference_utils)
            sample_images = generate_sample_images(
                prompt=self.sample_prompt,
                model_path=self.base_model_path,
                lora_path=self.output_dir,
                num_images=self.sample_num_images,
                resolution=(resolution, resolution),
                seed=self.sample_seed
            )

            if sample_images and len(sample_images) > 0:
                # Display in sidecar if available
                if self.sample_sidecar:
                    with self.sample_sidecar:
                        print(f"🎨 Epoch {self.current_epoch} Sample Images")
                        print(f"📝 Prompt: {self.sample_prompt}")
                        print(f"📐 Resolution: {resolution}x{resolution}")
                        print()

                        for i, img_path in enumerate(sample_images):
                            if os.path.exists(img_path):
                                display(IPImage(img_path, width=300))
                                print(f"Image {i+1}")
                            else:
                                print(f"❌ Image {i+1} not found: {img_path}")
                else:
                    # Fallback to main notebook
                    print("🎨 Sample Images Generated:")
                    for i, img_path in enumerate(sample_images):
                        if os.path.exists(img_path):
                            display(IPImage(img_path, width=300))
                        else:
                            print(f"❌ Image {i+1} not found: {img_path}")
            else:
                print("❌ No sample images were generated")

        except Exception as e:
            print(f"❌ Error generating sample images: {e}")
        finally:
            if callback:
                callback()

    def _on_view_latest_samples(self, button):
        """View samples from the most recent epoch in sidecar"""
        with self.sample_viewing_output:
            try:
                if not self.output_dir:
                    print("❌ No training output directory set")
                    return

                # Look for sample images in output directory
                sample_dir = os.path.join(self.output_dir, "sample")
                if not os.path.exists(sample_dir):
                    print("📂 No sample directory found yet. Complete an epoch first to generate samples.")
                    return

                # Find the latest epoch samples
                sample_files = []
                for file in os.listdir(sample_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_files.append(os.path.join(sample_dir, file))

                if not sample_files:
                    print("📷 No sample images found yet. Complete an epoch to generate samples.")
                    return

                # Sort by modification time to get latest
                sample_files.sort(key=os.path.getmtime, reverse=True)
                latest_samples = sample_files[:self.sample_num_images]  # Show latest N samples

                print(f"👁️ Displaying {len(latest_samples)} latest training samples...")

                # Display in sidecar
                if self.sample_sidecar:
                    with self.sample_sidecar:
                        print(f"🖼️ Latest Training Samples (Epoch {self.current_epoch})")
                        print(f"📝 Sample Prompt: {self.sample_prompt}")
                        print("-" * 50)

                        for i, img_path in enumerate(latest_samples):
                            if os.path.exists(img_path):
                                display(IPImage(img_path, width=300))
                                print(f"Sample {i+1} - {os.path.basename(img_path)}")
                                print()
                else:
                    # Fallback to main output
                    print("🖼️ Latest Training Samples:")
                    for i, img_path in enumerate(latest_samples):
                        if os.path.exists(img_path):
                            display(IPImage(img_path, width=300))

            except Exception as e:
                print(f"❌ Error viewing latest samples: {e}")

    def _on_view_all_samples(self, button):
        """View all training samples organized by epoch"""
        with self.sample_viewing_output:
            try:
                if not self.output_dir:
                    print("❌ No training output directory set")
                    return

                sample_dir = os.path.join(self.output_dir, "sample")
                if not os.path.exists(sample_dir):
                    print("📂 No sample directory found yet.")
                    return

                # Organize samples by epoch
                sample_files = []
                for file in os.listdir(sample_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_files.append(os.path.join(sample_dir, file))

                if not sample_files:
                    print("📷 No sample images found yet.")
                    return

                # Sort by name (which should include epoch info)
                sample_files.sort()

                print(f"📁 Displaying all {len(sample_files)} training samples...")

                # Display in sidecar
                if self.sample_sidecar:
                    with self.sample_sidecar:
                        print("🗂️ All Training Samples")
                        print(f"📝 Sample Prompt: {self.sample_prompt}")
                        print("=" * 50)

                        for img_path in sample_files:
                            if os.path.exists(img_path):
                                display(IPImage(img_path, width=250))
                                print(f"📷 {os.path.basename(img_path)}")
                                print()
                else:
                    # Fallback to main output
                    for img_path in sample_files:
                        if os.path.exists(img_path):
                            display(IPImage(img_path, width=250))

            except Exception as e:
                print(f"❌ Error viewing all samples: {e}")

    def _on_refresh_samples(self, button):
        """Refresh the sample display"""
        with self.sample_viewing_output:
            print("🔄 Refreshing sample display...")
            # Clear and refresh latest samples
            self._on_view_latest_samples(button)

    def display(self):
        """Display the widget"""
        display(self.widget_box)
