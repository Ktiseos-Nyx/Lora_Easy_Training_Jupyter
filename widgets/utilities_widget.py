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
        self.upload_output = widgets.Output()
        hf_upload_box = widgets.VBox([self.hf_token, self.model_path_to_upload, self.hf_repo_name, self.upload_button, self.upload_output])

        # --- LoRA Resizing ---
        self.lora_input_path = widgets.Text(description="Input LoRA Path:", placeholder="/path/to/input_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_output_path = widgets.Text(description="Output LoRA Path:", placeholder="/path/to/output_lora.safetensors", layout=widgets.Layout(width='99%'))
        self.lora_new_dim = widgets.IntSlider(value=4, min=1, max=128, step=1, description='New Dim:', style={'description_width': 'initial'})
        self.lora_new_alpha = widgets.IntSlider(value=1, min=1, max=128, step=1, description='New Alpha:', style={'description_width': 'initial'})
        self.resize_button = widgets.Button(description="Resize LoRA", button_style='primary')
        self.resize_output = widgets.Output()
        lora_resize_box = widgets.VBox([
            self.lora_input_path,
            self.lora_output_path,
            self.lora_new_dim,
            self.lora_new_alpha,
            self.resize_button,
            self.resize_output
        ])

        # --- Dataset Counting ---
        self.dataset_count_path = widgets.Text(description="Dataset Path:", placeholder="/path/to/your/dataset", layout=widgets.Layout(width='99%'))
        self.count_button = widgets.Button(description="Count Dataset Files", button_style='primary')
        self.count_output = widgets.Output()
        dataset_count_box = widgets.VBox([self.dataset_count_path, self.count_button, self.count_output])

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
        with self.upload_output:
            self.upload_output.clear_output()
            self.manager.upload_to_huggingface(
                self.hf_token.value,
                self.model_path_to_upload.value,
                self.hf_repo_name.value
            )

    def run_resize_lora(self, b):
        with self.resize_output:
            self.resize_output.clear_output()
            self.manager.resize_lora(
                self.lora_input_path.value,
                self.lora_output_path.value,
                self.lora_new_dim.value,
                self.lora_new_alpha.value
            )

    def run_count_dataset_files(self, b):
        with self.count_output:
            self.count_output.clear_output()
            self.manager.count_dataset_files(self.dataset_count_path.value)

    def display(self):
        display(self.widget_box)
