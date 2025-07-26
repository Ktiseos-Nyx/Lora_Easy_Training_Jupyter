# widgets/dataset_widget.py
import ipywidgets as widgets
from IPython.display import display
from core.dataset_manager import DatasetManager
from core.managers import ModelManager

class DatasetWidget:
    def __init__(self, dataset_manager=None):
        # Use dependency injection - accept manager instance or create default
        if dataset_manager is None:
            from core.managers import ModelManager
            dataset_manager = DatasetManager(ModelManager())
        
        self.manager = dataset_manager
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the Dataset Manager."""
        header_icon = "📊"
        header_main = widgets.HTML(f"<h2>{header_icon} 2. Dataset Manager</h2>")

        # --- Unified Dataset Setup Section ---
        dataset_setup_desc = widgets.HTML("""<h3>📁 Dataset Setup</h3>
        <p><strong>🎯 Choose your preferred method to set up your training dataset.</strong></p>
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>📋 Available Methods:</strong><br>
        • <strong>URL/ZIP Download:</strong> Download and extract from URLs or file paths<br>
        • <strong>Direct Image Upload:</strong> Upload individual images directly (perfect for small datasets)<br>
        • <strong>Gelbooru Scraper:</strong> Download images from Gelbooru using tags (anime/character datasets)<br>
        </div>""")
        
        # Dataset method selection
        self.dataset_method = widgets.RadioButtons(
            options=[('📥 URL/ZIP Download', 'url'), ('📁 Direct Image Upload', 'upload'), ('🔍 Gelbooru Scraper', 'gelbooru')],
            description='Method:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )
        
        # Shared dataset directory (will be auto-populated)
        self.dataset_directory = widgets.Text(
            description="Dataset Directory:",
            placeholder="Will be set automatically or enter manually",
            layout=widgets.Layout(width='99%')
        )
        
        # URL/ZIP Download Section
        url_method_desc = widgets.HTML("<h4>📥 URL/ZIP Download Setup</h4>")
        
        self.project_name = widgets.Text(
            description="Project Name:", 
            placeholder="e.g., my_awesome_character (no spaces or special chars)", 
            layout=widgets.Layout(width='99%')
        )
        
        self.dataset_url = widgets.Text(
            description="Dataset URL/Path:", 
            placeholder="/path/to/dataset.zip or HuggingFace URL", 
            layout=widgets.Layout(width='99%')
        )
        
        self.url_download_button = widgets.Button(description="🚀 Download & Extract Dataset", button_style='success')
        
        self.url_method_box = widgets.VBox([
            url_method_desc,
            self.project_name,
            self.dataset_url,
            self.url_download_button
        ])
        
        # Direct Upload Section  
        upload_method_desc = widgets.HTML("<h4>📁 Direct Image Upload Setup</h4>")
        
        self.folder_name = widgets.Text(
            description="Folder Name:",
            placeholder="e.g., my_character_dataset",
            layout=widgets.Layout(width='70%')
        )
        
        self.create_folder_button = widgets.Button(
            description="📁 Create Folder", 
            button_style='info',
            layout=widgets.Layout(width='25%')
        )
        
        folder_creation_box = widgets.HBox([self.folder_name, self.create_folder_button])
        
        self.file_upload = widgets.FileUpload(
            accept='.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.tif',
            multiple=True,
            description='Select Images:',
            layout=widgets.Layout(width='99%')
        )
        
        self.upload_images_button = widgets.Button(
            description="🚀 Upload Images", 
            button_style='success',
            disabled=True
        )
        
        self.upload_method_box = widgets.VBox([
            upload_method_desc,
            folder_creation_box,
            self.file_upload,
            self.upload_images_button
        ])
        
        # Gelbooru Scraper Section
        gelbooru_method_desc = widgets.HTML("""<h4>🔍 Gelbooru Image Scraper</h4>
        <div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #856404;'>
        <strong>⚠️ Important Notes:</strong><br>
        • Use appropriate tags like "1girl, character_name, blue_hair" - check Gelbooru first<br>
        • Use minus tags to exclude content: "-nsfw, -explicit"<br>
        • Downloads may take time depending on number of images<br>
        • Images are filtered to common formats (jpg, png, webp)<br>
        </div>""")
        
        self.gelbooru_tags = widgets.Text(
            description="Gelbooru Tags:",
            placeholder="e.g., 1girl, blue_hair, long_hair, -solo",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )
        
        self.gelbooru_limit = widgets.IntSlider(
            value=100,
            min=10,
            max=1000,
            step=10,
            description='Max Images:',
            style={'description_width': 'initial'}
        )
        
        self.gelbooru_folder = widgets.Text(
            description="Folder Name:",
            placeholder="e.g., character_name_dataset",
            layout=widgets.Layout(width='70%')
        )
        
        self.gelbooru_button = widgets.Button(
            description="🔍 Scrape from Gelbooru",
            button_style='info',
            layout=widgets.Layout(width='25%')
        )
        
        gelbooru_action_box = widgets.HBox([self.gelbooru_folder, self.gelbooru_button])
        
        self.gelbooru_method_box = widgets.VBox([
            gelbooru_method_desc,
            self.gelbooru_tags,
            self.gelbooru_limit,
            gelbooru_action_box
        ])
        
        # Status and output (shared)
        self.dataset_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>Status:</strong> Select a method to begin</div>")
        self.dataset_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        # Show/hide method boxes based on selection
        def on_method_change(change):
            if change['new'] == 'url':
                self.url_method_box.layout.display = 'block'
                self.upload_method_box.layout.display = 'none'
                self.gelbooru_method_box.layout.display = 'none'
            elif change['new'] == 'upload':
                self.url_method_box.layout.display = 'none'
                self.upload_method_box.layout.display = 'block'
                self.gelbooru_method_box.layout.display = 'none'
            else:  # gelbooru
                self.url_method_box.layout.display = 'none'
                self.upload_method_box.layout.display = 'none'
                self.gelbooru_method_box.layout.display = 'block'
        
        self.dataset_method.observe(on_method_change, names='value')
        # Initialize with URL method visible
        self.upload_method_box.layout.display = 'none'
        self.gelbooru_method_box.layout.display = 'none'
        
        dataset_setup_box = widgets.VBox([
            dataset_setup_desc,
            self.dataset_method,
            self.dataset_directory,
            self.url_method_box,
            self.upload_method_box,
            self.gelbooru_method_box,
            self.dataset_status,
            self.dataset_output
        ])

        # --- Tagging Section ---
        tagging_desc = widgets.HTML("""<h3>▶️ Image Tagging</h3>
        <p>Automatically generate captions for your images using AI taggers. <strong>Anime method uses SmilingWolf's WD14 taggers</strong>, Photo method uses BLIP captioning.</p>
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>📋 Available SmilingWolf WD14 Models:</strong><br>
        <strong>V3 Models (Latest, Recommended):</strong><br>
        • <strong>EVA02 Large v3:</strong> Best quality, 315M params (default)<br>
        • <strong>ViT Large v3:</strong> High quality, updated training<br>
        • <strong>SwinV2 v3:</strong> Balanced performance, latest<br>
        • <strong>ConvNeXT v3:</strong> Good speed, updated<br>
        • <strong>ViT v3:</strong> Fast tagging, latest<br><br>
        <strong>V2 Models (Stable):</strong><br>
        • <strong>SwinV2 v2, MoAT v2, ConvNeXT v2:</strong> Proven stable options<br><br>
        <em>🔄 Models auto-download from HuggingFace on first use!</em>
        </div>""")
        
        # Dataset directory will be auto-populated from setup section
        
        self.tagging_method = widgets.Dropdown(
            options=['anime', 'photo'], 
            description='Method:',
            style={'description_width': 'initial'}
        )
        
        # Enhanced tagger models with descriptions (mix of v2 and v3 models that actually exist)
        tagger_models = {
            "SmilingWolf/wd-eva02-large-tagger-v3": "EVA02 Large v3 (Best Quality, Newer)",
            "SmilingWolf/wd-vit-large-tagger-v3": "ViT Large v3 (High Quality, Updated)", 
            "SmilingWolf/wd-swinv2-tagger-v3": "SwinV2 v3 (Balanced, Latest)",
            "SmilingWolf/wd-convnext-tagger-v3": "ConvNeXT v3 (Good Speed, Updated)",
            "SmilingWolf/wd-vit-tagger-v3": "ViT v3 (Fast, Latest)",
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2": "SwinV2 v2 (Stable)",
            "SmilingWolf/wd-v1-4-moat-tagger-v2": "MoAT v2 (Alternative)", 
            "SmilingWolf/wd-v1-4-convnext-tagger-v2": "ConvNeXT v2 (Stable)"
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
        self.tagging_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007acc;'><strong>Status:</strong> Ready</div>")
        self.tagging_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        tagging_box = widgets.VBox([
            tagging_desc,
            self.tagging_method, 
            self.tagger_model, 
            self.tagger_desc,
            self.tagging_threshold, 
            self.blacklist_tags,
            self.caption_extension,
            self.tagging_button, 
            self.tagging_status,
            self.tagging_output
        ])

        # --- Dataset Cleanup ---
        cleanup_desc = widgets.HTML("""<h3>▶️ Dataset Cleanup</h3>
        <p>Clean up old files when re-tagging datasets or starting fresh.</p>
        
        <div style='background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #856404;'>
        <strong>⚠️ What gets cleaned:</strong><br>
        • <strong>.txt files:</strong> Caption files from previous tagging<br>
        • <strong>.npz files:</strong> Cached latents from previous training<br>
        • <strong>.caption files:</strong> Alternative caption format<br>
        • <strong>Non-image files:</strong> Model files (.safetensors, .ckpt), configs (.json, .yaml), etc.<br>
        <em>🎯 Use this when you want to re-tag a dataset or clean up accidentally extracted files</em>
        </div>""")
        
        # Dataset directory will be auto-populated from setup section
        
        # Cleanup options
        self.cleanup_text_files = widgets.Checkbox(
            value=True,
            description="🗑️ Remove .txt caption files",
            indent=False
        )
        
        self.cleanup_npz_files = widgets.Checkbox(
            value=True,
            description="🗑️ Remove .npz cached latents",
            indent=False
        )
        
        self.cleanup_caption_files = widgets.Checkbox(
            value=True,
            description="🗑️ Remove .caption files",
            indent=False
        )
        
        self.cleanup_non_images = widgets.Checkbox(
            value=False,
            description="🗑️ Remove non-image files (.safetensors, .ckpt, .bin, .json, .yaml, etc.)",
            indent=False
        )
        
        self.cleanup_preview = widgets.Checkbox(
            value=True,
            description="👀 Preview files before deletion (recommended)",
            indent=False
        )
        
        self.cleanup_button = widgets.Button(description="🧹 Clean Dataset", button_style='warning')
        self.cleanup_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>Status:</strong> Ready</div>")
        self.cleanup_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        cleanup_box = widgets.VBox([
            cleanup_desc,
            self.cleanup_text_files,
            self.cleanup_npz_files, 
            self.cleanup_caption_files,
            self.cleanup_non_images,
            self.cleanup_preview,
            self.cleanup_button,
            self.cleanup_status,
            self.cleanup_output
        ])

        # --- Advanced Caption Management ---
        caption_desc = widgets.HTML("""<h3>▶️ Advanced Caption Management</h3>
        <p>Professional tag curation tools for cleaning and organizing your caption files.</p>
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>🎯 Available Tools:</strong><br>
        • <strong>Trigger Words:</strong> Add activation tags to all captions<br>
        • <strong>Tag Removal:</strong> Remove unwanted tags from all captions<br>
        • <strong>Search & Replace:</strong> Advanced bulk tag replacement with AND/OR logic<br>
        • <strong>Sort & Deduplicate:</strong> Organize tags alphabetically and remove duplicates<br>
        </div>""")
        
        # Basic trigger word management
        basic_management_desc = widgets.HTML("<h4>🎯 Basic Tag Management</h4>")
        
        self.trigger_word = widgets.Text(
            description="Trigger Word:", 
            placeholder="e.g., my_character, myart_style", 
            layout=widgets.Layout(width='99%')
        )
        
        self.add_trigger_button = widgets.Button(description="➕ Add Trigger Word", button_style='success')
        
        self.remove_tags = widgets.Text(
            description="Remove Tags:",
            placeholder="e.g., 1girl,solo (comma separated)",
            layout=widgets.Layout(width='99%')
        )
        
        self.remove_tags_button = widgets.Button(description="➖ Remove Tags", button_style='warning')
        
        # Advanced search and replace
        advanced_management_desc = widgets.HTML("<h4>🔍 Advanced Search & Replace</h4>")
        
        self.search_tags = widgets.Text(
            description="Search Tags:",
            placeholder="e.g., 1girl solo standing (space or comma separated)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )
        
        self.replace_with = widgets.Text(
            description="Replace With:",
            placeholder="e.g., woman alone (leave empty to remove)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )
        
        self.search_mode = widgets.Dropdown(
            options=[('AND - all search tags must match', 'AND'), ('OR - any search tag matches', 'OR')],
            value='OR',
            description='Search Mode:',
            style={'description_width': 'initial'}
        )
        
        self.search_replace_button = widgets.Button(description="🔄 Search & Replace", button_style='info')
        
        # Sort and deduplicate
        organization_desc = widgets.HTML("<h4>📝 Tag Organization</h4>")
        
        self.sort_alphabetically = widgets.Checkbox(
            value=True,
            description="Sort tags alphabetically",
            indent=False
        )
        
        self.remove_duplicates = widgets.Checkbox(
            value=True,
            description="Remove duplicate tags",
            indent=False
        )
        
        self.organize_tags_button = widgets.Button(description="📋 Organize Tags", button_style='info')
        
        self.caption_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>Status:</strong> Ready</div>")
        self.caption_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))
        
        caption_box = widgets.VBox([
            caption_desc,
            basic_management_desc,
            self.trigger_word, 
            self.add_trigger_button,
            self.remove_tags,
            self.remove_tags_button,
            advanced_management_desc,
            self.search_tags,
            self.replace_with,
            self.search_mode,
            self.search_replace_button,
            organization_desc,
            self.sort_alphabetically,
            self.remove_duplicates,
            self.organize_tags_button,
            self.caption_status,
            self.caption_output
        ])

        # --- Accordion ---
        self.accordion = widgets.Accordion(children=[
            dataset_setup_box,
            tagging_box,
            caption_box,
            cleanup_box
        ])
        self.accordion.set_title(0, "📁 Dataset Setup")
        self.accordion.set_title(1, "🏷️ Image Tagging")
        self.accordion.set_title(2, "📝 Advanced Caption Management")
        self.accordion.set_title(3, "🧹 Dataset Cleanup")

        self.widget_box = widgets.VBox([header_main, self.accordion])

        # --- Button Events ---
        self.url_download_button.on_click(self.run_url_download)
        self.create_folder_button.on_click(self.run_create_folder)
        self.upload_images_button.on_click(self.run_upload_images)
        self.gelbooru_button.on_click(self.run_gelbooru_scraper)
        self.tagging_button.on_click(self.run_tagging)
        self.cleanup_button.on_click(self.run_cleanup)
        self.add_trigger_button.on_click(self.run_add_trigger)
        self.remove_tags_button.on_click(self.run_remove_tags)
        self.search_replace_button.on_click(self.run_search_replace)
        self.organize_tags_button.on_click(self.run_organize_tags)

    def run_url_download(self, b):
        """Handle URL/ZIP download method"""
        self.dataset_output.clear_output()
        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Downloading and extracting...</div>"
        with self.dataset_output:
            project_name = self.project_name.value.strip()
            dataset_url = self.dataset_url.value.strip()
            
            if not project_name:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a project name.</div>"
                print("❌ Please enter a project name.")
                return
                
            if not dataset_url:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a dataset URL.</div>"
                print("❌ Please enter a dataset URL.")
                return
            
            # Sanitize project name
            import re
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)
            if clean_name != project_name:
                print(f"📝 Cleaned project name: '{project_name}' → '{clean_name}'")
                project_name = clean_name
            
            project_dir = f"datasets/{project_name}"
            
            # Create project directory
            import os
            os.makedirs(project_dir, exist_ok=True)
            
            success = self.manager.extract_dataset(dataset_url, project_dir)
            if success:
                # Update shared dataset directory
                self.dataset_directory.value = project_dir
                
                # Count images
                try:
                    from core.image_utils import count_images_in_directory
                    image_count = count_images_in_directory(project_dir)
                    print(f"📁 Found {image_count} images in {project_dir}")
                    print(f"📝 Dataset directory set: {project_dir}")
                    
                    self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Downloaded {image_count} images to {project_dir}</div>"
                except Exception as e:
                    print(f"⚠️ Image counting error: {e}")
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Download complete.</div>"
            else:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Download and extraction failed. Check logs.</div>"

    def run_create_folder(self, b):
        """Create a new folder for image upload"""
        self.dataset_output.clear_output()
        folder_name = self.folder_name.value.strip()
        
        if not folder_name:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a folder name.</div>"
            return
        
        # Sanitize folder name
        import re
        import os
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
        if clean_name != folder_name:
            folder_name = clean_name
        
        # Create folder in datasets directory
        folder_path = f"datasets/{folder_name}"
        
        with self.dataset_output:
            try:
                os.makedirs(folder_path, exist_ok=True)
                print(f"✅ Created folder: {folder_path}")
                
                # Update shared dataset directory
                self.dataset_directory.value = folder_path
                
                # Enable upload button
                self.upload_images_button.disabled = False
                
                self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder created! Now select images to upload.</div>"
                
            except Exception as e:
                print(f"❌ Failed to create folder: {e}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to create folder.</div>"

    def run_upload_images(self, b):
        """Upload multiple images to the created folder"""
        self.dataset_output.clear_output()
        
        if not self.dataset_directory.value:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please create a folder first.</div>"
            return
        
        if not self.file_upload.value:
            self.dataset_status.value = "❌ Status: Please select images to upload."
            return
        
        upload_folder = self.dataset_directory.value
        uploaded_files = self.file_upload.value
        
        self.dataset_status.value = f"⚙️ Status: Uploading {len(uploaded_files)} images..."
        
        with self.dataset_output:
            import os
            uploaded_count = 0
            total_size = 0
            
            print(f"📁 Uploading {len(uploaded_files)} images to: {upload_folder}")
            print("="*60)
            
            for file_info in uploaded_files:
                try:
                    filename = file_info['name']
                    content_memview = file_info['content']
                    
                    # Convert memory view to bytes
                    content = content_memview.tobytes()
                    
                    if not content:
                        print(f"⚠️ Warning: {filename} has no content, skipping")
                        continue
                    
                    file_path = os.path.join(upload_folder, filename)
                    
                    # Write file content
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    
                    file_size = len(content)
                    total_size += file_size
                    uploaded_count += 1
                    
                    print(f"✅ {filename} ({file_size/1024:.1f} KB)")
                    
                except Exception as e:
                    print(f"❌ Failed to upload {filename if 'filename' in locals() else 'unknown file'}: {e}")
            
            total_size_mb = total_size / (1024 * 1024)
            print(f"\n🎉 Upload complete!")
            print(f"📊 Uploaded: {uploaded_count}/{len(uploaded_files)} images")
            print(f"💾 Total size: {total_size_mb:.2f} MB")
            print(f"📁 Location: {upload_folder}")
            
            if uploaded_count > 0:
                self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Uploaded {uploaded_count} images ({total_size_mb:.1f} MB)</div>"
                
                # Clear the file upload widget for next use
                self.file_upload.value = ()
            else:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> No images were uploaded successfully.</div>"

    def run_gelbooru_scraper(self, b):
        """Handle Gelbooru image scraping"""
        self.dataset_output.clear_output()
        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Preparing Gelbooru scraper...</div>"
        
        with self.dataset_output:
            tags = self.gelbooru_tags.value.strip()
            folder_name = self.gelbooru_folder.value.strip()
            limit = self.gelbooru_limit.value
            
            if not tags:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter Gelbooru tags.</div>"
                print("❌ Please enter tags to search for on Gelbooru.")
                return
                
            if not folder_name:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a folder name.</div>"
                print("❌ Please enter a folder name.")
                return
            
            # Sanitize folder name
            import re
            import os
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
            if clean_name != folder_name:
                print(f"📝 Cleaned folder name: '{folder_name}' → '{clean_name}'")
                folder_name = clean_name
            
            # Create folder in datasets directory
            dataset_dir = f"datasets/{folder_name}"
            
            # Ask for user confirmation
            def confirm_download():
                print(f"\n🔍 About to search Gelbooru for: {tags}")
                print(f"📁 Images will be saved to: {dataset_dir}")
                print(f"📊 Maximum images: {limit}")
                print(f"\n⚠️ This will download images from the internet.")
                
                # For now, we'll proceed automatically in the widget
                # In a real implementation, you might want a confirmation dialog
                return True
            
            try:
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Update shared dataset directory immediately
                self.dataset_directory.value = dataset_dir
                
                self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Scraping Gelbooru... ({limit} max images)</div>"
                
                # Run the scraper
                success = self.manager.scrape_from_gelbooru(
                    tags=tags,
                    dataset_dir=dataset_dir,
                    limit=limit,
                    confirm_callback=confirm_download
                )
                
                if success:
                    # Count images after download
                    try:
                        from core.image_utils import count_images_in_directory
                        image_count = count_images_in_directory(dataset_dir)
                        print(f"\n📁 Total images in dataset: {image_count}")
                        print(f"📝 Dataset directory set: {dataset_dir}")
                        
                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Downloaded {image_count} images from Gelbooru</div>"
                    except Exception as e:
                        print(f"⚠️ Image counting error: {e}")
                        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Gelbooru scraping complete.</div>"
                else:
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Gelbooru scraping failed. Check logs.</div>"
                    
            except Exception as e:
                print(f"❌ Failed to create dataset directory: {e}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to create dataset directory.</div>"

    def run_tagging(self, b):
        self.tagging_output.clear_output()
        self.tagging_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Starting {self.tagging_method.value} tagging...</div>"
        with self.tagging_output:
            if not self.dataset_directory.value:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return
                
            print(f"🏷️ Starting {self.tagging_method.value} tagging with {self.tagger_model.value.split('/')[-1]}...")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            
            # Enhanced tagging with more options
            success = self.manager.tag_images(
                self.dataset_directory.value,
                self.tagging_method.value,
                self.tagger_model.value,
                self.tagging_threshold.value,
                blacklist_tags=self.blacklist_tags.value,
                caption_extension=self.caption_extension.value
            )
            if success:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tagging complete.</div>"
            else:
                self.tagging_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Tagging failed. Check logs.</div>"

    def run_cleanup(self, b):
        """Clean up old caption and cache files from dataset directory"""
        self.cleanup_output.clear_output()
        self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Scanning for files to clean...</div>"
        
        with self.cleanup_output:
            dataset_dir = self.dataset_directory.value.strip()
            
            if not dataset_dir:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return
                
            print(f"🧩 Cleaning dataset: {dataset_dir}")
            
            import os
            import glob
            
            if not os.path.exists(dataset_dir):
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Dataset directory does not exist.</div>"
                print(f"❌ Directory does not exist: {dataset_dir}")
                return
            
            # Find files to clean
            files_to_clean = []
            
            if self.cleanup_text_files.value:
                txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
                files_to_clean.extend([(f, "caption file") for f in txt_files])
            
            if self.cleanup_npz_files.value:
                npz_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
                files_to_clean.extend([(f, "cached latent") for f in npz_files])
            
            if self.cleanup_caption_files.value:
                caption_files = glob.glob(os.path.join(dataset_dir, "*.caption"))
                files_to_clean.extend([(f, "caption file") for f in caption_files])
            
            if self.cleanup_non_images.value:
                # Define common image extensions to preserve
                image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
                
                # Find all files in directory
                all_files = glob.glob(os.path.join(dataset_dir, "*"))
                
                for file_path in all_files:
                    if os.path.isfile(file_path):  # Skip directories
                        file_ext = os.path.splitext(file_path)[1].lower()
                        
                        # If it's not an image extension and not already in our cleanup list
                        if file_ext not in image_extensions:
                            # Skip files we're already cleaning (txt, npz, caption)
                            if file_ext not in {'.txt', '.npz', '.caption'}:
                                # Identify file type for better reporting
                                if file_ext in {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}:
                                    file_type = "model file"
                                elif file_ext in {'.json', '.yaml', '.yml', '.toml', '.ini'}:
                                    file_type = "config file"
                                elif file_ext in {'.py', '.sh', '.bat', '.cmd'}:
                                    file_type = "script file"
                                elif file_ext in {'.zip', '.rar', '.7z', '.tar', '.gz'}:
                                    file_type = "archive file"
                                elif file_ext in {'.log', '.txt'}:
                                    file_type = "log file"
                                else:
                                    file_type = f"non-image file ({file_ext})"
                                
                                files_to_clean.append((file_path, file_type))
            
            if not files_to_clean:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> No files found to clean. Directory is already clean!</div>"
                print("✅ No files found to clean. Directory is already clean!")
                return
            
            print(f"🧹 Found {len(files_to_clean)} files to clean in: {dataset_dir}")
            print("="*60)
            
            # Preview files
            if self.cleanup_preview.value:
                print("👀 PREVIEW - Files that will be deleted:")
                for file_path, file_type in files_to_clean:
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    size_mb = file_size / (1024 * 1024)
                    print(f"  🗑️ {file_name} ({file_type}, {size_mb:.2f} MB)")
                
                print("\n⚠️ PREVIEW MODE: No files were actually deleted.")
                print("💡 Uncheck 'Preview files before deletion' to actually delete these files.\n")
                
                self.cleanup_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>👀 Status:</strong> Preview complete - found {len(files_to_clean)} files to clean.</div>"
                return
            
            # Actually delete files
            deleted_count = 0
            total_size_freed = 0
            
            print("🗑️ DELETING FILES:")
            for file_path, file_type in files_to_clean:
                try:
                    if os.path.exists(file_path):
                        file_name = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_count += 1
                        total_size_freed += file_size
                        print(f"  ✅ Deleted: {file_name} ({file_type})")
                    else:
                        print(f"  ⚠️ File not found: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  ❌ Failed to delete {os.path.basename(file_path)}: {e}")
            
            size_freed_mb = total_size_freed / (1024 * 1024)
            print(f"\n🎉 Cleanup complete!")
            print(f"📊 Deleted {deleted_count} files")
            print(f"💾 Freed {size_freed_mb:.2f} MB of disk space")
            print(f"📁 Dataset directory: {dataset_dir}")
            
            if deleted_count > 0:
                self.cleanup_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Cleaned {deleted_count} files ({size_freed_mb:.1f} MB freed).</div>"
            else:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> No files were deleted (may have been errors).</div>"

    def run_add_trigger(self, b):
        self.caption_output.clear_output()
        self.caption_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Adding trigger word...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return
                
            if not self.trigger_word.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify a trigger word.</div>"
                print("❌ Please specify a trigger word.")
                return
                
            print(f"➕ Adding trigger word '{self.trigger_word.value}' to captions...")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            success = self.manager.add_trigger_word(self.dataset_directory.value, self.trigger_word.value)
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Trigger word added.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to add trigger word. Check logs.</div>"
    
    def run_remove_tags(self, b):
        """Remove specified tags from all caption files"""
        self.caption_output.clear_output()
        self.caption_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Removing tags...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return
                
            if not self.remove_tags.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify tags to remove.</div>"
                print("❌ Please specify tags to remove.")
                return
                
            print(f"➖ Removing tags '{self.remove_tags.value}' from captions...")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            success = self.manager.remove_tags(self.dataset_directory.value, self.remove_tags.value)
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tags removed.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to remove tags. Check logs.</div>"

    def run_search_replace(self, b):
        """Advanced search and replace functionality for tags"""
        self.caption_output.clear_output()
        self.caption_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Processing search and replace...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return
                
            if not self.search_tags.value.strip():
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify tags to search for.</div>"
                print("❌ Please specify tags to search for.")
                return
                
            print(f"🔍 Search and replace operation:")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            print(f"🔍 Search tags: {self.search_tags.value}")
            print(f"🔄 Replace with: '{self.replace_with.value}' (empty = remove)")
            print(f"🎯 Search mode: {self.search_mode.value}")
            
            success = self.manager.search_and_replace_tags(
                dataset_dir=self.dataset_directory.value,
                search_tags=self.search_tags.value,
                replace_with=self.replace_with.value,
                search_mode=self.search_mode.value
            )
            
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Search and replace complete.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Search and replace failed. Check logs.</div>"

    def run_organize_tags(self, b):
        """Sort tags alphabetically and remove duplicates"""
        self.caption_output.clear_output()
        self.caption_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Organizing tags...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return
                
            operations = []
            if self.sort_alphabetically.value:
                operations.append("sort alphabetically")
            if self.remove_duplicates.value:
                operations.append("remove duplicates")
                
            if not operations:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> No operations selected.</div>"
                print("⚠️ Please select at least one operation (sort or remove duplicates).")
                return
                
            print(f"📋 Tag organization:")
            print(f"📁 Dataset: {self.dataset_directory.value}")
            print(f"🔧 Operations: {', '.join(operations)}")
            
            success = self.manager.sort_and_deduplicate_tags(
                dataset_dir=self.dataset_directory.value,
                sort_alphabetically=self.sort_alphabetically.value,
                remove_duplicates=self.remove_duplicates.value
            )
            
            if success:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tag organization complete.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Tag organization failed. Check logs.</div>"

    def display(self):
        display(self.widget_box)
