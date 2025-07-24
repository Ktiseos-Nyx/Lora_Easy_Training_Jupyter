# widgets/training_monitor_widget.py
import ipywidgets as widgets
from IPython.display import display
import re
import time
import threading

class TrainingMonitorWidget:
    def __init__(self):
        self.create_widgets()
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.training_phase = "Initializing..."
        
    def create_widgets(self):
        """Create the training monitor interface"""
        
        # Header
        header = widgets.HTML("<h2>🎛️ Training Monitor</h2>")
        
        # Training Phase Status
        self.phase_status = widgets.HTML(
            value="<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #007acc;'>"
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
        
        # Auto-save status
        self.autosave_status = widgets.HTML(
            value="<div style='background: #d4edda; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'>"
                  "<strong>💾 Auto-save:</strong> Enabled - checkpoints saved each epoch</div>"
        )
        
        # Progress section
        progress_section = widgets.VBox([
            widgets.HTML("<h3>📊 Training Progress</h3>"),
            self.epoch_label,
            self.epoch_progress,
            self.step_label, 
            self.step_progress,
            self.resource_info
        ])
        
        # Log section
        log_section = widgets.VBox([
            widgets.HTML("<h3>📋 Training Log</h3>"),
            self.training_log
        ])
        
        # Main widget container
        self.widget_box = widgets.VBox([
            header,
            self.phase_status,
            progress_section,
            self.autosave_status,
            log_section
        ])
    
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
                # Could extract total steps if available in output
                self.update_step_progress(current_step, 100)  # Placeholder
        
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