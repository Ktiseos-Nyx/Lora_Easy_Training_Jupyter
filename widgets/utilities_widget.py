# widgets/utilities_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.utilities_manager import UtilitiesManager

class UtilitiesWidget:
    def __init__(self):
        self.manager = UtilitiesManager()
        self.create_widgets()

    def create_widgets(self):
        header_icon = "🔧"
        header_main = widgets.HTML(f"<h2>{header_icon} 4. Utilities</h2>")

        # --- Hugging Face Upload ---
        self.hf_token = widgets.Password(description="HF Write Token:", placeholder="HuggingFace Write API Token", layout=widgets.Layout(width='99%'))
        self.model_path_to_upload = widgets.Text(description="Model Path:", placeholder="/path/to/your/lora.safetensors", layout=widgets.Layout(width='99%'))
        self.hf_repo_name = widgets.Text(description="Repo Name:", placeholder="e.g., my-awesome-loras", layout=widgets.Layout(width='99%'))
        self.upload_button = widgets.Button(description="Upload to Hugging Face", button_style='primary')
        self.upload_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>📊 Status:</strong> Ready to upload</div>")
        self.upload_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        hf_upload_box = widgets.VBox([self.hf_token, self.model_path_to_upload, self.hf_repo_name, self.upload_button, self.upload_status, self.upload_output])

        # --- LoRA Resizing ---
        self.lora_input_path = widgets.Text(description="Input LoRA Path:", placeholder="/path/to/input_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_output_path = widgets.Text(description="Output LoRA Path:", placeholder="/path/to/output_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_new_dim = widgets.IntSlider(value=4, min=1, max=128, step=1, description='New Dim:', style={'description_width': 'initial'})
        self.lora_new_alpha = widgets.IntSlider(value=1, min=1, max=128, step=1, description='New Alpha:', style={'description_width': 'initial'})
        self.resize_button = widgets.Button(description="Resize LoRA", button_style='primary')
        self.resize_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>📊 Status:</strong> Ready to resize</div>")
        self.resize_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        lora_resize_box = widgets.VBox([
            self.lora_input_path,
            self.lora_output_path,
            self.lora_new_dim,
            self.lora_new_alpha,
            self.resize_button,
            self.resize_status,
            self.resize_output
        ])

        # --- Dataset Counting ---
        self.dataset_count_path = widgets.Text(description="Dataset Path:", placeholder="/path/to/your/dataset", layout=widgets.Layout(width='99%'))
        self.count_button = widgets.Button(description="Count Dataset Files", button_style='primary')
        self.count_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>📊 Status:</strong> Ready to count</div>")
        self.count_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        dataset_count_box = widgets.VBox([self.dataset_count_path, self.count_button, self.count_status, self.count_output])

        # --- Accordion ---
        accordion = widgets.Accordion(children=[
            hf_upload_box,
            lora_resize_box,
            dataset_count_box
        ])
        accordion.set_title(0, "▶️ Upload to Hugging Face")
        accordion.set_title(1, "▶️ LoRA Resizing")
        accordion.set_title(2, "▶️ Dataset Counting")

        self.widget_box = widgets.VBox([header_main, accordion])

        # --- Button Events ---
        self.upload_button.on_click(self.run_upload_to_hf)
        self.resize_button.on_click(self.run_resize_lora)
        self.count_button.on_click(self.run_count_dataset_files)

    def run_upload_to_hf(self, b):
        self.upload_output.clear_output()
        self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Uploading to Hugging Face...</div>"
        with self.upload_output:
            success = self.manager.upload_to_huggingface(
                self.hf_token.value,
                self.model_path_to_upload.value,
                self.hf_repo_name.value
            )
            if success:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Upload complete!</div>"
            else:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload failed. Check logs.</div>"

    def run_resize_lora(self, b):
        self.resize_output.clear_output()
        self.resize_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Resizing LoRA...</div>"
        with self.resize_output:
            success = self.manager.resize_lora(
                self.lora_input_path.value,
                self.lora_output_path.value,
                self.lora_new_dim.value,
                self.lora_new_alpha.value
            )
            if success:
                self.resize_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> LoRA resized successfully!</div>"
            else:
                self.resize_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> LoRA resize failed. Check logs.</div>"

    def run_count_dataset_files(self, b):
        self.count_output.clear_output()
        self.count_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Counting files...</div>"
        with self.count_output:
            result = self.manager.count_dataset_files(self.dataset_count_path.value)
            if result:
                self.count_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> File count complete!</div>"
            else:
                self.count_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> File count failed. Check logs.</div>"

    def display(self):
        display(self.widget_box)
