# widgets/training_monitor_widget.py
import ipywidgets as widgets
from IPython.display import display
import re
import time
import threading
from shared_managers import get_config_manager

class TrainingMonitorWidget:
    def __init__(self, training_manager_instance):
        self.training_manager = training_manager_instance
        self.training_config = None # To store the config passed from TrainingWidget
        self.create_widgets()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.training_phase = "Initializing..."
        
    def create_widgets(self):
        """Create the training monitor interface with accordion structure"""
        
        # Header for the entire widget
        main_header = widgets.HTML("<h2>📈 Training Progress & Control</h2>")
        
        # Create accordion sections
        self.create_training_control_section()
        self.create_progress_monitoring_section()
        
        # Create accordion
        self.accordion = widgets.Accordion(children=[
            self.training_control_box,
            self.progress_monitoring_box
        ])
        self.accordion.set_title(0, "🚀 Start Training")
        self.accordion.set_title(1, "📊 Live Progress Monitor")
        
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
            value="<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace;'>"
                  "<strong>💾 Resources:</strong> Monitoring will begin when training starts</div>"
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
    
    def set_training_config(self, config):
        """Set the training configuration received from TrainingWidget"""
        self.training_config = config
        
        # Calculate steps per epoch for accurate progress tracking
        if config:
            try:
                # Get dataset info
                dataset_size = config.get('dataset_size', 100)  # fallback
                num_repeats = config.get('num_repeats', 1)
                batch_size = config.get('train_batch_size', 1)
                
                # Calculate steps per epoch: (images * repeats) / batch_size
                self.steps_per_epoch = max(1, (dataset_size * num_repeats) // batch_size)
                self.total_epochs = config.get('epochs', 1)
                
                print(f"📊 Calculated: {self.steps_per_epoch} steps per epoch for {dataset_size} images")
            except Exception as e:
                print(f"⚠️ Could not calculate steps per epoch: {e}")
                self.steps_per_epoch = 100  # Safe fallback

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
        elif "epoch" in line.lower() and "step" in line.lower():
            # Try to parse epoch/step from training output
            epoch_match = re.search(r'epoch[:\s]+(\d+)', line.lower())
            step_match = re.search(r'step[:\s]+(\d+)', line.lower()) 
            
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                # Try to extract total epochs from line or use stored value
                total_match = re.search(r'epoch[:\s]+\d+[/\s]+(\d+)', line.lower())
                if total_match:
                    total_epochs = int(total_match.group(1))
                    self.update_epoch_progress(current_epoch, total_epochs)
                elif self.total_epochs > 0:
                    self.update_epoch_progress(current_epoch, self.total_epochs)
                    
                self.update_phase(f"Training - Epoch {current_epoch}", "success")
            
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
            self.update_phase("Saving checkpoint...", "warning")
        elif "training complete" in line.lower() or "finished" in line.lower():
            self.update_phase("Training completed successfully! 🎉", "success")
        elif "error" in line.lower() or "failed" in line.lower():
            self.update_phase("Training error occurred", "error")
    
    def clear_log(self):
        """Clear the training log"""
        with self.training_log:
            self.training_log.clear_output()
    
    def display(self):
        """Display the widget"""
        display(self.widget_box)