# widgets/utilities_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.utilities_manager import UtilitiesManager

class UtilitiesWidget:
    def __init__(self, utilities_manager=None):
        # Use dependency injection - accept manager instance or create default
        if utilities_manager is None:
            utilities_manager = UtilitiesManager()
            
        self.manager = utilities_manager
        self.create_widgets()

    def create_widgets(self):
        header_icon = "🔧"
        header_main = widgets.HTML(f"<h2>{header_icon} 4. Utilities</h2>")

        # --- Hugging Face Upload ---
        self.hf_token = widgets.Password(description="HF Write Token:", placeholder="HuggingFace Write API Token", layout=widgets.Layout(width='99%'))
        self.model_path_to_upload = widgets.Text(description="Model Path:", placeholder="/path/to/your/lora.safetensors", layout=widgets.Layout(width='99%'))
        self.hf_repo_name = widgets.Text(description="Repo Name:", placeholder="e.g., my-awesome-loras", layout=widgets.Layout(width='99%'))
        self.upload_button = widgets.Button(description="Upload to Hugging Face", button_style='primary')
        self.upload_status = widgets.HTML("<p><strong>📊 Status:</strong> Ready to upload</p>")
        self.upload_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll'))
        hf_upload_box = widgets.VBox([self.hf_token, self.model_path_to_upload, self.hf_repo_name, self.upload_button, self.upload_status, self.upload_output])

        # --- LoRA Resizing ---
        self.lora_input_path = widgets.Text(description="Input LoRA Path:", placeholder="/path/to/input_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_output_path = widgets.Text(description="Output LoRA Path:", placeholder="/path/to/output_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_new_dim = widgets.IntSlider(value=4, min=1, max=128, step=1, description='New Dim:', style={'description_width': 'initial'})
        self.lora_new_alpha = widgets.IntSlider(value=1, min=1, max=128, step=1, description='New Alpha:', style={'description_width': 'initial'})
        self.resize_button = widgets.Button(description="Resize LoRA", button_style='primary')
        self.resize_status = widgets.HTML("<p><strong>📊 Status:</strong> Ready to resize</p>")
        self.resize_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll'))
        lora_resize_box = widgets.VBox([
            self.lora_input_path,
            self.lora_output_path,
            self.lora_new_dim,
            self.lora_new_alpha,
            self.resize_button,
            self.resize_status,
            self.resize_output
        ])

        # --- Dataset Image Optimization ---
        optimization_desc = widgets.HTML("""<h4>🖼️ Dataset Image Optimization</h4>
        <p><strong>📋 What this does:</strong><br>
        • Convert PNG/BMP/TIFF images to JPEG or WebP for smaller file sizes<br>
        • Standardize image formats across your dataset<br>
        • Handle transparency properly (white background for JPEG)<br>
        • Keep original image dimensions - no resizing!</p>
        <p><strong>⚠️ Important:</strong> This modifies your images permanently. Make a backup first if needed!</p>""")
        
        self.optimize_dataset_path = widgets.Text(
            description="Dataset Path:",
            placeholder="datasets/your_dataset_folder",
            layout=widgets.Layout(width='99%')
        )
        
        self.optimize_format = widgets.Dropdown(
            options=[('WebP (Best compression)', 'webp'), ('JPEG (Wide compatibility)', 'jpeg')],
            value='webp',
            description='Target Format:',
            style={'description_width': 'initial'}
        )
        
        self.optimize_quality = widgets.IntSlider(
            value=95,
            min=60,
            max=100,
            step=5,
            description='Quality:',
            style={'description_width': 'initial'}
        )
        
        optimization_help = widgets.HTML("""<p><strong>💡 Tips:</strong><br>
        • <strong>Format conversion only:</strong> Images keep their original dimensions<br>
        • <strong>Quality 95:</strong> Excellent quality, good compression<br>
        • <strong>Quality 85:</strong> Good quality, better compression<br>
        • <strong>WebP:</strong> Best compression, smaller files<br>
        • <strong>JPEG:</strong> Wide compatibility, good compression</p>""")
        
        self.optimize_button = widgets.Button(
            description="🖼️ Optimize Dataset Images",
            button_style='warning',
            layout=widgets.Layout(width='99%')
        )
        
        self.optimize_status = widgets.HTML("<p><strong>📊 Status:</strong> Ready to optimize</p>")
        self.optimize_output = widgets.Output(layout=widgets.Layout(height='400px', overflow='scroll'))
        
        optimization_box = widgets.VBox([
            optimization_desc,
            self.optimize_dataset_path,
            self.optimize_format,
            self.optimize_quality,
            optimization_help,
            self.optimize_button,
            self.optimize_status,
            self.optimize_output
        ])

        # Dataset counting has been moved to the Dataset Widget for better organization

        # --- Accordion ---
        accordion = widgets.Accordion(children=[
            hf_upload_box,
            lora_resize_box,
            optimization_box
        ])
        accordion.set_title(0, "▶️ Upload to Hugging Face")
        accordion.set_title(1, "▶️ LoRA Resizing")
        accordion.set_title(2, "🖼️ Dataset Image Optimization")

        self.widget_box = widgets.VBox([header_main, accordion])

        # --- Button Events ---
        self.upload_button.on_click(self.run_upload_to_hf)
        self.resize_button.on_click(self.run_resize_lora)
        self.optimize_button.on_click(self.run_optimize_dataset)

    def run_upload_to_hf(self, b):
        self.upload_output.clear_output()
        self.upload_status.value = "<p><strong>⚙️ Status:</strong> Uploading to Hugging Face...</p>"
        with self.upload_output:
            success = self.manager.upload_to_huggingface(
                self.hf_token.value,
                self.model_path_to_upload.value,
                self.hf_repo_name.value
            )
            if success:
                self.upload_status.value = "<p><strong>✅ Status:</strong> Upload complete!</p>"
            else:
                self.upload_status.value = "<p><strong>❌ Status:</strong> Upload failed. Check logs.</p>"

    def run_resize_lora(self, b):
        self.resize_output.clear_output()
        self.resize_status.value = "<p><strong>⚙️ Status:</strong> Resizing LoRA...</p>"
        with self.resize_output:
            success = self.manager.resize_lora(
                self.lora_input_path.value,
                self.lora_output_path.value,
                self.lora_new_dim.value,
                self.lora_new_alpha.value
            )
            if success:
                self.resize_status.value = "<p><strong>✅ Status:</strong> LoRA resized successfully!</p>"
            else:
                self.resize_status.value = "<p><strong>❌ Status:</strong> LoRA resize failed. Check logs.</p>"

    def run_optimize_dataset(self, b):
        """Handle dataset image optimization"""
        self.optimize_output.clear_output()
        self.optimize_status.value = "<p><strong>⚙️ Status:</strong> Optimizing dataset images...</p>"
        
        with self.optimize_output:
            dataset_path = self.optimize_dataset_path.value.strip()
            target_format = self.optimize_format.value
            quality = self.optimize_quality.value
            
            if not dataset_path:
                self.optimize_status.value = "<p><strong>❌ Status:</strong> Please enter a dataset path.</p>"
                print("❌ Please enter a dataset path.")
                return
            
            print(f"🖼️ Starting image optimization...")
            print(f"📁 Dataset: {dataset_path}")
            print(f"🎯 Target format: {target_format.upper()}")
            print(f"⚙️ Quality: {quality}")
            print("📐 No resizing - keeping original dimensions")
            
            # Call the optimization method with no resizing
            success = self.manager.optimize_dataset_images(
                dataset_path=dataset_path,
                target_format=target_format,
                max_file_size_mb=999,  # High value to disable size-based resizing
                quality=quality,
                max_dimension=None  # No dimension-based resizing
            )
            
            if success:
                self.optimize_status.value = "<p><strong>✅ Status:</strong> Dataset optimization complete!</p>"
            else:
                self.optimize_status.value = "<p><strong>❌ Status:</strong> Dataset optimization failed. Check logs.</p>"

    # Dataset counting has been moved to the Dataset Widget for better organization

    def display(self):
        display(self.widget_box)
