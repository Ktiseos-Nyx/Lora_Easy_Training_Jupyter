# widgets/dataset_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.dataset_manager import DatasetManager
from core.managers import ModelManager

class DatasetWidget:
    def __init__(self):
        # Note: This is not ideal, but for now we'll instantiate a ModelManager
        # to pass to the DatasetManager. A better approach would be to use a single
        # manager instance for the whole application.
        self.manager = DatasetManager(ModelManager())
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the Dataset Manager."""
        header_icon = "📊"
        header_main = widgets.HTML(f"<h2>{header_icon} 2. Dataset Manager</h2>")

        # --- Upload Section ---
        self.upload_path = widgets.Text(description="Zip Path:", placeholder="/path/to/dataset.zip or HuggingFace URL", layout=widgets.Layout(width='99%'))
        self.extract_dir = widgets.Text(description="Extract to:", placeholder="e.g., my_dataset_folder", layout=widgets.Layout(width='99%'))
        self.upload_button = widgets.Button(description="Upload & Extract", button_style='primary')
        self.upload_output = widgets.Output()
        upload_box = widgets.VBox([self.upload_path, self.extract_dir, self.upload_button, self.upload_output])

        # --- Tagging Section ---
        tagging_desc = widgets.HTML("<h3>▶️ Image Tagging</h3><p>Automatically generate captions for your images using AI taggers. Anime method uses specialized taggers, Photo method uses BLIP captioning.</p>")
        
        self.tagging_dataset_dir = widgets.Text(
            description="Dataset Dir:", 
            placeholder="e.g., my_dataset_folder", 
            layout=widgets.Layout(width='99%')
        )
        
        self.tagging_method = widgets.Dropdown(
            options=['anime', 'photo'], 
            description='Method:',
            style={'description_width': 'initial'}
        )
        
        # Enhanced tagger models with descriptions
        tagger_models = {
            "SmilingWolf/wd-eva02-large-tagger-v3": "EVA02 Large v3 (Best Quality, Slower)",
            "SmilingWolf/wd-vit-large-tagger-v3": "ViT Large v3 (High Quality)", 
            "SmilingWolf/wd-swinv2-tagger-v3": "SwinV2 v3 (Balanced)",
            "SmilingWolf/wd-convnext-tagger-v3": "ConvNeXT v3 (Fast)",
            "SmilingWolf/wd-vit-tagger-v3": "ViT v3 (Fastest)"
        }
        
        self.tagger_model = widgets.Dropdown(
            options=list(tagger_models.keys()),
            value="SmilingWolf/wd-eva02-large-tagger-v3",
            description='Tagger Model:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )
        
        # Show model description
        self.tagger_desc = widgets.HTML()
        def update_tagger_desc(change):
            if change['new'] in tagger_models:
                self.tagger_desc.value = f"<small><i>{tagger_models[change['new']]}</i></small>"
        self.tagger_model.observe(update_tagger_desc, names='value')
        update_tagger_desc({'new': self.tagger_model.value})  # Initial desc
        
        self.tagging_threshold = widgets.FloatSlider(
            value=0.25, min=0.1, max=1.0, step=0.05, 
            description='Threshold:', 
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.blacklist_tags = widgets.Text(
            description="Blacklist Tags:",
            placeholder="e.g., 1girl,solo,standing (comma separated)",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )
        
        self.caption_extension = widgets.Dropdown(
            options=['.txt', '.caption'],
            value='.txt',
            description='Caption Extension:',
            style={'description_width': 'initial'}
        )
        
        self.tagging_button = widgets.Button(description="🏷️ Start Tagging", button_style='primary')
        self.tagging_output = widgets.Output()
        
        tagging_box = widgets.VBox([
            tagging_desc,
            self.tagging_dataset_dir, 
            self.tagging_method, 
            self.tagger_model, 
            self.tagger_desc,
            self.tagging_threshold, 
            self.blacklist_tags,
            self.caption_extension,
            self.tagging_button, 
            self.tagging_output
        ])

        # --- Caption Management ---
        caption_desc = widgets.HTML("<h3>▶️ Caption Management</h3><p>Add trigger words to activate your LoRA, or clean up captions by removing unwanted tags.</p>")
        
        self.caption_dataset_dir = widgets.Text(
            description="Dataset Dir:", 
            placeholder="e.g., my_dataset_folder", 
            layout=widgets.Layout(width='99%')
        )
        
        # Trigger word management
        self.trigger_word = widgets.Text(
            description="Trigger Word:", 
            placeholder="e.g., my_character, myart_style", 
            layout=widgets.Layout(width='99%')
        )
        
        self.add_trigger_button = widgets.Button(description="➕ Add Trigger Word", button_style='success')
        
        # Tag removal
        self.remove_tags = widgets.Text(
            description="Remove Tags:",
            placeholder="e.g., 1girl,solo (comma separated)",
            layout=widgets.Layout(width='99%')
        )
        
        self.remove_tags_button = widgets.Button(description="➖ Remove Tags", button_style='warning')
        
        self.caption_output = widgets.Output()
        
        caption_box = widgets.VBox([
            caption_desc,
            self.caption_dataset_dir, 
            self.trigger_word, 
            self.add_trigger_button,
            self.remove_tags,
            self.remove_tags_button,
            self.caption_output
        ])

        # --- Accordion ---
        self.accordion = widgets.Accordion(children=[
            upload_box,
            tagging_box,
            caption_box
        ])
        self.accordion.set_title(0, "▶️ Upload & Extract")
        self.accordion.set_title(1, "▶️ Image Tagging")
        self.accordion.set_title(2, "▶️ Caption Management")

        self.widget_box = widgets.VBox([header_main, self.accordion])

        # --- Button Events ---
        self.upload_button.on_click(self.run_upload)
        self.tagging_button.on_click(self.run_tagging)
        self.add_trigger_button.on_click(self.run_add_trigger)
        self.remove_tags_button.on_click(self.run_remove_tags)

    def run_upload(self, b):
        with self.upload_output:
            self.upload_output.clear_output()
            self.manager.extract_dataset(self.upload_path.value, self.extract_dir.value)

    def run_tagging(self, b):
        with self.tagging_output:
            self.tagging_output.clear_output()
            
            if not self.tagging_dataset_dir.value:
                print("❌ Please specify a dataset directory.")
                return
                
            print(f"🏷️ Starting {self.tagging_method.value} tagging with {self.tagger_model.value.split('/')[-1]}...")
            
            # Enhanced tagging with more options
            self.manager.tag_images(
                self.tagging_dataset_dir.value,
                self.tagging_method.value,
                self.tagger_model.value,
                self.tagging_threshold.value,
                blacklist_tags=self.blacklist_tags.value,
                caption_extension=self.caption_extension.value
            )

    def run_add_trigger(self, b):
        with self.caption_output:
            self.caption_output.clear_output()
            
            if not self.caption_dataset_dir.value:
                print("❌ Please specify a dataset directory.")
                return
                
            if not self.trigger_word.value:
                print("❌ Please specify a trigger word.")
                return
                
            print(f"➕ Adding trigger word '{self.trigger_word.value}' to captions...")
            self.manager.add_trigger_word(self.caption_dataset_dir.value, self.trigger_word.value)
    
    def run_remove_tags(self, b):
        """Remove specified tags from all caption files"""
        with self.caption_output:
            self.caption_output.clear_output()
            
            if not self.caption_dataset_dir.value:
                print("❌ Please specify a dataset directory.")
                return
                
            if not self.remove_tags.value:
                print("❌ Please specify tags to remove.")
                return
                
            print(f"➖ Removing tags '{self.remove_tags.value}' from captions...")
            self.manager.remove_tags(self.caption_dataset_dir.value, self.remove_tags.value)

    def display(self):
        display(self.widget_box)
