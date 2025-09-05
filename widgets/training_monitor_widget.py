# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/training_monitor_widget.py
import os
import re
import threading
import time

import ipywidgets as widgets
from IPython.display import display

# Inference functionality removed
from shared_managers import get_config_manager


class TrainingMonitorWidget:
    def __init__(self, training_manager_instance):
        self.training_manager = training_manager_instance
        self.training_config = None # To store the config passed from TrainingWidget

# Sidecar functionality removed

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
# Sample viewing/inference section removed

        # Create accordion
        self.accordion = widgets.Accordion(children=[
            self.training_control_box,
            self.progress_monitoring_box,
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

    def display(self):
        """Display the widget"""
        display(self.widget_box)
