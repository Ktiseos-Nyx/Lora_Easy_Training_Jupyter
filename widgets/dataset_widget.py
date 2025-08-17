# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# widgets/dataset_widget.py
import asyncio
import glob
import logging
import os
import re
import zipfile
from datetime import datetime

import ipywidgets as widgets
from IPython.display import display

from core.dataset_manager import DatasetManager
from core.logging_config import setup_file_logging

# Setup external logging for widget debugging
setup_file_logging()
logger = logging.getLogger(__name__)


class DatasetWidget:
    def __init__(self, dataset_manager=None):
        # Use dependency injection - accept manager instance or create default
        if dataset_manager is None:
            dataset_manager = DatasetManager()  # Uses lazy ModelManager loading now!

        self.manager = dataset_manager
        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components for the Dataset Manager."""
        header_icon = "📊"
        header_main = widgets.HTML(f"<h2>{header_icon} 2. Dataset Manager</h2>")

        # --- Unified Dataset Setup Section ---
        dataset_setup_desc = widgets.HTML("""<h3>📁 Dataset Setup</h3>
        <p><strong>🎯 Choose your preferred method to set up your training dataset.</strong></p>
        <p><strong>📋 Available Methods:</strong></p>
        <ul>
        <li><strong>URL/ZIP Download:</strong> Download and extract from URLs or file paths</li>
        <li><strong>Direct Image Upload:</strong> Upload individual images directly (perfect for small datasets)</li>
        <li><strong>Gallery-DL Scraper:</strong> Download images from various sites using gallery-dl (anime/character datasets)</li>
        </ul>""")

        # Dataset method selection
        self.dataset_method = widgets.RadioButtons(
            options=[('📥 URL/ZIP Download', 'url'), ('📁 Direct Image Upload', 'upload'), ('🔍 Gallery-DL Scraper', 'gallery_dl')],
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
            placeholder="e.g., character_name (repeat count will be added automatically)",
            layout=widgets.Layout(width='70%')
        )

        self.folder_repeats = widgets.IntText(
            description="Repeat Count:",
            value=10,
            min=1,
            max=100,
            layout=widgets.Layout(width='25%'),
            style={'description_width': 'initial'}
        )

        self.create_folder_button = widgets.Button(
            description="📁 Create Folder",
            button_style='info',
            layout=widgets.Layout(width='25%')
        )

        folder_creation_box = widgets.VBox([
            widgets.HBox([self.folder_name, self.folder_repeats]),
            widgets.HTML("<small><i>💡 Final folder will be: <strong>{repeat_count}_{folder_name}</strong> (Kohya format)</i></small>"),
            self.create_folder_button
        ])

        self.file_upload = widgets.FileUpload(
            accept='.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.tif,.zip', # Added .zip
            multiple=True,
            description='Select Images/ZIP:', # Updated description
            layout=widgets.Layout(width='99%'),
            # Add max file size to prevent memory issues
            max_size=10*1024*1024  # 10MB per file limit
        )

        self.upload_images_button = widgets.Button(
            description="🚀 Upload Images",
            button_style='success',
            disabled=True
        )

        self.upload_zip_button = widgets.Button(
            description="📦 Upload & Extract ZIP",
            button_style='primary',
            disabled=True
        )

        self.reset_upload_button = widgets.Button(
            description="🔄 Reset Upload",
            button_style='warning',
            disabled=False
        )

        # Create button row for upload and reset
        upload_button_row = widgets.HBox([self.upload_images_button, self.upload_zip_button, self.reset_upload_button])

        self.upload_method_box = widgets.VBox([
            upload_method_desc,
            folder_creation_box,
            self.file_upload,
            upload_button_row
        ])

        # Gelbooru Scraper Section
        # Gallery-DL Scraper Section
        gallery_dl_method_desc = widgets.HTML("""<h4>🔍 Gallery-DL Scraper</h4>
        <div style='background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>🌐 Supported Sites:</strong> Gelbooru, Danbooru, Pixiv, Twitter, etc. (see <a href="https://github.com/mikf/gallery-dl/blob/master/docs/supportedsites.md" target="_blank">supported sites</a>)<br>
        <strong>💡 Tips:</strong> Use appropriate tags, specify limits, and consider subfolders for organization.
        </div>""")

        self.gallery_dl_site = widgets.Dropdown(
            options=[('Gelbooru', 'gelbooru'), ('Danbooru', 'danbooru'), ('Pixiv', 'pixiv'), ('Twitter', 'twitter'), ('Custom URL (enter below)', 'custom')],
            description='Site:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.gallery_dl_tags = widgets.Text(
            description="Tags:",
            placeholder="e.g., 1girl, blue_hair, long_hair (comma separated)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_limit_range = widgets.Text(
            value="1-100",
            description='Limit Range:',
            placeholder="e.g., 1-100 (downloads first 100 images)",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.gallery_dl_custom_url = widgets.Text(
            description="Custom URL:",
            placeholder="Full URL for custom site (if 'Custom URL' selected above)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_sub_folder = widgets.Text(
            description="Subfolder:",
            placeholder="e.g., my_character (creates a subfolder within dataset dir)",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_additional_args = widgets.Text(
            description="Additional Args:",
            placeholder="e.g., --no-part --write-info-json",
            layout=widgets.Layout(width='99%'),
            style={'description_width': 'initial'}
        )

        self.gallery_dl_write_tags = widgets.Checkbox(
            value=True,
            description="Write Tags (creates .txt files)",
            indent=False
        )

        self.gallery_dl_use_aria2c = widgets.Checkbox(
            value=True,
            description="Use aria2c (faster downloads)",
            indent=False
        )

        self.gallery_dl_button = widgets.Button(
            description="🔍 Start Gallery-DL Scrape",
            button_style='info',
            layout=widgets.Layout(width='25%')
        )

        self.gallery_dl_method_box = widgets.VBox([
            gallery_dl_method_desc,
            self.gallery_dl_site,
            self.gallery_dl_tags,
            self.gallery_dl_limit_range,
            self.gallery_dl_custom_url,
            self.gallery_dl_sub_folder,
            self.gallery_dl_additional_args,
            self.gallery_dl_write_tags,
            self.gallery_dl_use_aria2c,
            self.gallery_dl_button
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
                self.gallery_dl_method_box.layout.display = 'none'
            else:  # gallery_dl
                self.url_method_box.layout.display = 'none'
                self.upload_method_box.layout.display = 'none'
                self.gallery_dl_method_box.layout.display = 'block'

        self.dataset_method.observe(on_method_change, names='value')
        # Initialize with URL method visible
        self.upload_method_box.layout.display = 'none'
        self.gallery_dl_method_box.layout.display = 'none'

        dataset_setup_box = widgets.VBox([
            dataset_setup_desc,
            self.dataset_method,
            self.dataset_directory,
            self.url_method_box,
            self.upload_method_box,
            self.gallery_dl_method_box,
            self.dataset_status,
            self.dataset_output
        ])

        # Add FiftyOne integration button
        self.fiftyone_explorer_button = widgets.Button(
            description="🔍 Open Visual Explorer",
            button_style='info',
            layout=widgets.Layout(width='99%')
        )
        self.fiftyone_explorer_button.on_click(self.launch_fiftyone_explorer)

        # Add Apply Curation Changes button
        self.apply_curation_button = widgets.Button(
            description="💾 Apply Curation Changes",
            button_style='success',
            layout=widgets.Layout(width='99%')
        )
        self.apply_curation_button.on_click(self.apply_fiftyone_curation)

        dataset_setup_box.children = list(dataset_setup_box.children) + [self.fiftyone_explorer_button, self.apply_curation_button]

        # --- File Renaming Section ---
        rename_desc = widgets.HTML("""<h3>📝 File Renaming</h3>
        <p>Rename your dataset files to fix UTF-8 issues and create consistent naming. <strong>Caption files are automatically renamed too!</strong></p>
        <p><strong>📋 Naming Patterns:</strong></p>
        <ul>
        <li><strong>Simple Numbering:</strong> MyProject_001.jpg, MyProject_002.jpg</li>
        <li><strong>Sanitized Original:</strong> MyProject_CleanedName.jpg (removes special chars)</li>
        <li><strong>With Timestamp:</strong> MyProject_20240807_001.jpg</li>
        </ul>
        <p><em>💡 This fixes common training issues with special characters and UTF-8!</em></p>""")

        self.rename_project_name = widgets.Text(
            description="Project Name:",
            placeholder="e.g., MyCharacter (will become MyCharacter_001.jpg)",
            layout=widgets.Layout(width='99%')
        )

        self.rename_pattern = widgets.Dropdown(
            options=[
                ('Simple Numbering (MyProject_001.jpg)', 'numbered'),
                ('Sanitized Original (MyProject_CleanName.jpg)', 'sanitized'),
                ('With Timestamp (MyProject_20240807_001.jpg)', 'timestamp')
            ],
            value='numbered',
            description='Naming Pattern:'
        )

        self.rename_start_number = widgets.IntText(
            value=1,
            description="Start Number:",
            style={'description_width': 'initial'}
        )

        self.preview_rename_button = widgets.Button(
            description="👁️ Preview Changes",
            button_style='info'
        )

        self.rename_files_button = widgets.Button(
            description="📝 Rename Files",
            button_style='warning',
            disabled=True
        )

        self.rename_output = widgets.Output()

        rename_controls = widgets.HBox([
            self.rename_start_number,
            self.preview_rename_button,
            self.rename_files_button
        ])

        rename_box = widgets.VBox([
            rename_desc,
            self.rename_project_name,
            self.rename_pattern,
            rename_controls,
            self.rename_output
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
        
        <p><strong>⚠️ What gets cleaned:</strong></p>
        <ul>
        <li><strong>.txt files:</strong> Caption files from previous tagging</li>
        <li><strong>.npz files:</strong> Cached latents from previous training</li>
        <li><strong>.caption files:</strong> Alternative caption format</li>
        <li><strong>Non-image files:</strong> Model files (.safetensors, .ckpt), configs (.json, .yaml), etc.</li>
        </ul>
        <p><em>🎯 Use this when you want to re-tag a dataset or clean up accidentally extracted files</em></p>""")

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

        # --- HuggingFace Upload Section ---
        upload_desc = widgets.HTML("""<h3>🤗 HuggingFace Dataset Upload</h3>
        <div style='background: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>🚀 Share your dataset with the community!</strong><br>
        Upload your prepared dataset to HuggingFace Hub for sharing and backup.<br>
        <strong>📋 Requirements:</strong> HuggingFace account with WRITE token
        </div>""")

        self.upload_dataset_path = widgets.Text(
            description="Dataset Path:",
            placeholder="Path to your dataset folder (auto-filled from current dataset)",
            layout=widgets.Layout(width='99%')
        )

        self.upload_dataset_name = widgets.Text(
            description="Repository Name:",
            placeholder="e.g., my-character-dataset (lowercase, hyphens allowed)",
            layout=widgets.Layout(width='99%')
        )

        self.upload_hf_token = widgets.Password(
            description="HF Token:",
            placeholder="Your HuggingFace WRITE token",
            layout=widgets.Layout(width='99%')
        )

        token_help = widgets.HTML("""
        <div style='background: #fff3cd; padding: 8px; border-radius: 5px; margin: 5px 0;'>
        🔑 Get your <strong>WRITE</strong> token <a href='https://huggingface.co/settings/tokens' target='_blank'>here</a>
        </div>
        """)

        self.upload_orgs_name = widgets.Text(
            description="Organization:",
            placeholder="Leave empty to use personal account, or enter org name",
            layout=widgets.Layout(width='99%')
        )

        self.upload_description = widgets.Textarea(
            description="Description:",
            placeholder="Describe your dataset (optional but recommended)",
            layout=widgets.Layout(width='99%', height='80px')
        )

        self.upload_make_private = widgets.Checkbox(
            value=False,
            description="🔒 Make repository private",
            indent=False
        )

        self.upload_button = widgets.Button(description="🚀 Upload to HuggingFace", button_style='info')
        self.upload_status = widgets.HTML("<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007bff;'><strong>Status:</strong> Ready</div>")
        self.upload_output = widgets.Output(layout=widgets.Layout(height='300px', overflow='scroll', border='1px solid #ddd'))

        upload_box = widgets.VBox([
            upload_desc,
            self.upload_dataset_path,
            self.upload_dataset_name,
            self.upload_hf_token,
            token_help,
            self.upload_orgs_name,
            self.upload_description,
            self.upload_make_private,
            self.upload_button,
            self.upload_status,
            self.upload_output
        ])

        # --- Image Utilities Section ---
        image_utils_desc = widgets.HTML("""<h3>🖼️ Image Format Conversion</h3>
        <p>Convert image formats for better compatibility and storage optimization.</p>
        <p><strong>⚠️ Note:</strong> This only converts formats (WebP ↔ JPG ↔ PNG). No resizing or cropping is performed to preserve training data quality.</p>
        """)

        self.format_selector = widgets.Dropdown(
            options=[('Convert to JPG', 'jpg'), ('Convert to PNG', 'png'), ('Convert to WebP', 'webp')],
            value='jpg',
            description='Target Format:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        # Format conversion only - no resizing

        self.conversion_quality = widgets.IntSlider(
            value=95,
            min=85,
            max=100,
            step=5,
            description='Quality:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='99%')
        )

        self.convert_images_button = widgets.Button(
            description="🔄 Convert Image Formats",
            button_style='info'
        )
        self.convert_images_button.on_click(self.run_format_conversion)

        self.image_utils_output = widgets.Output()

        image_utils_box = widgets.VBox([
            image_utils_desc,
            self.format_selector,
            self.conversion_quality,
            self.convert_images_button,
            self.image_utils_output
        ])

        # Tag Curation functionality is integrated into Dataset Setup section

        # --- Advanced Caption Management ---
        caption_desc = widgets.HTML("""<h3>▶️ Advanced Caption Management</h3>
        <p>Professional tag curation tools for cleaning and organizing your caption files.</p>
        <p><strong>🎯 Available Tools:</strong></p>
        <ul>
        <li><strong>Trigger Words:</strong> Add activation tags to all captions</li>
        <li><strong>Tag Removal:</strong> Remove unwanted tags from all captions</li>
        <li><strong>Search & Replace:</strong> Advanced bulk tag replacement with AND/OR logic</li>
        <li><strong>Sort & Deduplicate:</strong> Organize tags alphabetically and remove duplicates</li>
        </ul>""")

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

        # Review/display tags
        self.review_tags_button = widgets.Button(description="👀 Review Tags", button_style='info')

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
            widgets.HBox([self.remove_tags_button, self.review_tags_button]),
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
            rename_box,
            tagging_box,
            caption_box,
            cleanup_box,
            upload_box,
            image_utils_box
        ])
        self.accordion.set_title(0, "📁 Dataset Setup")
        self.accordion.set_title(1, "📝 File Renaming")
        self.accordion.set_title(2, "🏷️ Image Tagging")
        self.accordion.set_title(3, "📝 Advanced Caption Management")
        self.accordion.set_title(4, "🧹 Dataset Cleanup")
        self.accordion.set_title(5, "🤗 HuggingFace Upload")
        self.accordion.set_title(6, "🖼️ Image Utilities")

        self.widget_box = widgets.VBox([header_main, self.accordion])

        # --- Button Events ---
        self.url_download_button.on_click(self.run_url_download)
        self.create_folder_button.on_click(self.run_create_folder)
        self.upload_images_button.on_click(self._handle_async_upload)
        self.upload_zip_button.on_click(self._handle_async_zip_upload)
        
        logger.info("🔗 BUTTON HANDLERS: Upload button click handlers attached")
        # Gallery-dl scraper functionality integrated into URL download
        # self.gelbooru_button.on_click(self.run_gallery_dl_scraper)  # Button removed, functionality integrated
        self.preview_rename_button.on_click(self.run_preview_rename)
        self.rename_files_button.on_click(self.run_rename_files)
        self.tagging_button.on_click(self.run_tagging)
        self.cleanup_button.on_click(self.run_cleanup)
        self.add_trigger_button.on_click(self.run_add_trigger)
        self.remove_tags_button.on_click(self.run_remove_tags)
        self.review_tags_button.on_click(self.run_review_tags)
        self.search_replace_button.on_click(self.run_search_replace)
        self.organize_tags_button.on_click(self.run_organize_tags)
        self.upload_button.on_click(self.run_upload_to_huggingface)

        # --- File Upload Observer ---
        logger.info(f"🔗 ATTACHING FILE UPLOAD OBSERVER")
        logger.info(f"🔗 FileUpload widget type: {type(self.file_upload)}")
        logger.info(f"🔗 FileUpload widget value: {self.file_upload.value}")
        logger.info(f"🔗 FileUpload widget description: {self.file_upload.description}")
        
        self.file_upload.observe(self.on_file_upload_change, names='value')
        
        logger.info(f"🔗 FILE UPLOAD OBSERVER ATTACHED SUCCESSFULLY")
        logger.info(f"🔗 Observer method: {self.on_file_upload_change}")
        
        # Test if we can manually trigger the observer
        logger.info(f"🧪 TESTING: Manual observer trigger...")
        try:
            test_change = {'new': [], 'old': [], 'type': 'change', 'name': 'value', 'owner': self.file_upload}
            self.on_file_upload_change(test_change)
            logger.info(f"🧪 TESTING: Manual trigger successful!")
        except Exception as e:
            logger.error(f"🧪 TESTING: Manual trigger failed: {e}")
            
        # WORKAROUND: Implement polling-based file detection since observer isn't working
        logger.info(f"🔄 WORKAROUND: Starting file upload polling system...")
        self._last_file_count = 0
        self._last_file_names = set()
        import threading
        import time
        
        def poll_file_upload():
            """Poll FileUpload widget for changes since observer isn't working"""
            while True:
                try:
                    current_files = self.file_upload.value
                    current_count = len(current_files)
                    current_names = {f['name'] for f in current_files} if current_files else set()
                    
                    # Check if files have changed
                    if current_count != self._last_file_count or current_names != self._last_file_names:
                        logger.info(f"🔄 POLLING DETECTED FILE CHANGE: {self._last_file_count} -> {current_count} files")
                        
                        # Simulate the observer event manually
                        change_event = {
                            'new': current_files,
                            'old': (),  # We don't track old state in polling
                            'type': 'change',
                            'name': 'value',
                            'owner': self.file_upload
                        }
                        
                        # Trigger our observer method manually
                        self.on_file_upload_change(change_event)
                        
                        # Update tracking
                        self._last_file_count = current_count
                        self._last_file_names = current_names
                        
                except Exception as e:
                    logger.error(f"🔄 POLLING ERROR: {e}")
                
                # Poll every 2 seconds
                time.sleep(2)
        
        # Start polling in background thread
        polling_thread = threading.Thread(target=poll_file_upload, daemon=True)
        polling_thread.start()
        logger.info(f"🔄 WORKAROUND: File upload polling thread started")

        # Reset upload button event handler (button created earlier)
        self.reset_upload_button.on_click(self.reset_upload_widget)

        # Auto-detect existing dataset directories on init
        self.auto_detect_existing_datasets()

    def on_file_upload_change(self, change):
        """Handle file upload selection changes"""
        try:
            logger.info(f"🔍 FILE UPLOAD CHANGE TRIGGERED!")
            logger.info(f"🔍 Change object type: {type(change)}")
            logger.info(f"🔍 Change object keys: {list(change.keys()) if hasattr(change, 'keys') else 'No keys'}")
            
            new_files = change.get('new', [])
            logger.info(f"🔍 Files count: {len(new_files)}")
            logger.info(f"🔍 Change type: {change.get('type', 'No type')}")
            logger.debug(f"🔍 Full change object: {change}")
            
            if new_files:  # Files have been selected
                file_count = len(new_files)
                if file_count > 0:
                    # Check if a folder is created AND exists on disk
                    folder_path = self.dataset_directory.value.strip()
                    folder_exists = bool(folder_path) and os.path.exists(folder_path)

                    # Determine if any selected file is a ZIP (case-insensitive)
                    is_zip_selected = any(f['name'].lower().endswith('.zip') for f in new_files)

                    if folder_exists:
                        self.upload_images_button.disabled = is_zip_selected # Disable image upload if zip is selected
                        self.upload_zip_button.disabled = not is_zip_selected # Enable zip upload only if zip is selected

                        if is_zip_selected:
                            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> {file_count} file(s) selected. Ready to upload and extract ZIP to '{folder_path}'.</div>"
                        else:
                            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> {file_count} file(s) selected. Ready to upload images to '{folder_path}'.</div>"
                    else:
                        # If no folder exists but files are selected, auto-suggest creating folder
                        self.upload_images_button.disabled = True
                        self.upload_zip_button.disabled = True
                        # Try to suggest a folder name based on the files
                        suggested_name = "uploaded_dataset"
                        if is_zip_selected:
                            zip_name = next(f['name'] for f in new_files if f['name'].lower().endswith('.zip'))
                            suggested_name = zip_name.replace('.zip', '_dataset')

                        if not self.folder_name.value.strip():
                            self.folder_name.value = suggested_name

                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>📁 Status:</strong> {file_count} file(s) selected. Click 'Create Folder' first, then upload.</div>"
                else:
                    self.upload_images_button.disabled = True
                    self.upload_zip_button.disabled = True
            else:
                # No files selected - reset everything
                self.upload_images_button.disabled = True
                self.upload_zip_button.disabled = True
                if not self.dataset_directory.value.strip():
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>Status:</strong> Select images/ZIP or create a folder to start uploading.</div>"
        except Exception as e:
            logger.error(f"🚨 ERROR in file upload observer: {e}")
            logger.exception("Full traceback:")
            # Ensure buttons are disabled if error occurs
            self.upload_images_button.disabled = True
            self.upload_zip_button.disabled = True

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
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name)
            if clean_name != project_name:
                print(f"📝 Cleaned project name: '{project_name}' → '{clean_name}'")
                project_name = clean_name

            project_dir = f"datasets/{project_name}"

            # Create project directory
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
        """Create a new folder for image upload using Kohya's repeat count format"""
        logger.info("🖱️ BUTTON CLICK: Create Folder button clicked!")
        logger.info(f"📁 Folder name: '{self.folder_name.value}'")
        logger.info(f"📁 Repeat count: {self.folder_repeats.value}")
        
        self.dataset_output.clear_output()
        folder_name = self.folder_name.value.strip()
        repeat_count = self.folder_repeats.value

        if not folder_name:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter a folder name.</div>"
            return

        # Sanitize folder name
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
        if clean_name != folder_name:
            folder_name = clean_name

        # Create folder in Kohya's expected format: {repeat_count}_{folder_name}
        kohya_folder_name = f"{repeat_count}_{folder_name}"
        folder_path = f"datasets/{kohya_folder_name}"

        with self.dataset_output:
            try:
                os.makedirs(folder_path, exist_ok=True)
                print(f"✅ Created Kohya-compatible folder: {folder_path}")
                print(f"📊 Repeat count: {repeat_count} (auto-detected by training)")

                # Update shared dataset directory
                self.dataset_directory.value = folder_path

                # Check if files are already selected and enable appropriate buttons
                if self.file_upload.value:
                    file_count = len(self.file_upload.value)
                    is_zip_selected = any(f['name'].lower().endswith('.zip') for f in self.file_upload.value)

                    # Enable the right button based on file type
                    self.upload_images_button.disabled = is_zip_selected
                    self.upload_zip_button.disabled = not is_zip_selected

                    if is_zip_selected:
                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder created! {file_count} file(s) ready to upload (ZIP detected).</div>"
                    else:
                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder created! {file_count} file(s) ready to upload.</div>"
                else:
                    self.upload_images_button.disabled = True
                    self.upload_zip_button.disabled = True
                    self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Folder '{folder_path}' created! Now select images/ZIP to upload.</div>"

            except Exception as e:
                print(f"❌ Failed to create folder: {e}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to create folder.</div>"

    def _handle_async_upload(self, b):
        """Wrapper to handle async upload function"""
        logger.info("🖱️ BUTTON CLICK: Upload Images button clicked!")
        logger.info(f"🖱️ Button state - disabled: {self.upload_images_button.disabled}")
        logger.info(f"🖱️ Files in widget: {len(self.file_upload.value)}")
        asyncio.create_task(self.run_upload_images(b))

    def _handle_async_zip_upload(self, b):
        """Wrapper to handle async ZIP upload function"""
        logger.info("🖱️ BUTTON CLICK: Upload ZIP button clicked!")
        logger.info(f"🖱️ Button state - disabled: {self.upload_zip_button.disabled}")
        logger.info(f"🖱️ Files in widget: {len(self.file_upload.value)}")
        asyncio.create_task(self.run_upload_zip(b))

    async def run_upload_images(self, b):
        """Upload multiple images to the created folder with async batch processing"""
        self.dataset_output.clear_output()

        if not self.dataset_directory.value:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please create a folder first.</div>"
            return

        if not self.file_upload.value:
            self.dataset_status.value = "❌ Status: Please select images to upload."
            return

        upload_folder = self.dataset_directory.value
        uploaded_files = self.file_upload.value
        total_files = len(uploaded_files)
        batch_size = 5  # Process 5 files at a time

        with self.dataset_output:
            uploaded_count = 0
            total_size = 0

            print(f"📁 Uploading {total_files} images to: {upload_folder}")
            print("="*60)

            # Process files in batches
            for i in range(0, total_files, batch_size):
                batch = uploaded_files[i:i + batch_size]
                batch_start = i + 1
                batch_end = min(i + batch_size, total_files)
                
                # Update status for current batch
                self.dataset_status.value = f"⚙️ Status: Processing batch {batch_start}-{batch_end} of {total_files} images..."
                
                for file_info in batch:
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

                # Yield control back to the event loop after each batch
                await asyncio.sleep(0)

            total_size_mb = total_size / (1024 * 1024)
            print("\n🎉 Upload complete!")
            print(f"📊 Uploaded: {uploaded_count}/{total_files} images")
            print(f"💾 Total size: {total_size_mb:.2f} MB")
            print(f"📁 Location: {upload_folder}")

            if uploaded_count > 0:
                self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Uploaded {uploaded_count} images ({total_size_mb:.1f} MB)</div>"

                # DISABLED: Auto-clear causes race condition where widget resets before user can upload
                # self.file_upload.value = ()  # Users can manually reset if needed
            else:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> No images were uploaded successfully.</div>"

    async def run_upload_zip(self, b):
        """Upload and extract a ZIP file to the created folder with async processing"""
        self.dataset_output.clear_output()

        if not self.dataset_directory.value:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please create a folder first.</div>"
            return

        if not self.file_upload.value:
            self.dataset_status.value = "❌ Status: Please select a ZIP file to upload."
            return

        upload_folder = self.dataset_directory.value
        uploaded_files = self.file_upload.value

        # Find the first ZIP file in the uploaded files
        zip_file_info = None
        for file_info in uploaded_files:
            if file_info['name'].lower().endswith('.zip'):
                zip_file_info = file_info
                break

        if not zip_file_info:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> No ZIP file selected.</div>"
            print("❌ No ZIP file selected for upload.")
            return

        zip_filename = zip_file_info['name']
        zip_content = zip_file_info['content'].tobytes()

        self.dataset_status.value = f"⚙️ Status: Uploading and extracting {zip_filename}..."

        with self.dataset_output:
            temp_zip_path = os.path.join(upload_folder, zip_filename)

            try:
                print(f"📦 Saving {zip_filename} to {temp_zip_path}")
                with open(temp_zip_path, 'wb') as f:
                    f.write(zip_content)

                print(f"✨ Extracting {zip_filename} to {upload_folder}")
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(upload_folder)
                # Yield control during extraction
                await asyncio.sleep(0)
                print("✅ Extraction complete.")

                # Clean up the temporary zip file
                os.remove(temp_zip_path)
                print(f"🗑️ Removed temporary zip file: {temp_zip_path}")

                # Count images after extraction
                try:
                    from core.image_utils import count_images_in_directory
                    image_count = count_images_in_directory(upload_folder)
                    print(f"📁 Found {image_count} images in {upload_folder}")
                    self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Uploaded and extracted {zip_filename}. Found {image_count} images.</div>"
                except Exception as e:
                    print(f"⚠️ Image counting error: {e}")
                    self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Uploaded and extracted {zip_filename}.</div>"

                # DISABLED: Auto-clear causes race condition where widget resets before user can upload
                # self.file_upload.value = ()  # Users can manually reset if needed
                self.upload_images_button.disabled = True
                self.upload_zip_button.disabled = True

            except zipfile.BadZipFile:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Invalid ZIP file.</div>"
                print(f"❌ Error: {zip_filename} is not a valid ZIP file.")
            except Exception as e:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to upload or extract ZIP.</div>"
                print(f"❌ An error occurred during ZIP upload/extraction: {e}")

    def reset_upload_widget(self, b):
        """Reset the file upload widget to clear any cached state"""
        logger.info("🔄 MANUAL RESET: User clicked reset upload button")
        # Clear the upload widget value
        self.file_upload.value = ()

        # Reset button states
        self.upload_images_button.disabled = True
        self.upload_zip_button.disabled = True

        # Clear folder name if it was auto-suggested
        if self.folder_name.value in ['uploaded_dataset'] or '_dataset' in self.folder_name.value:
            self.folder_name.value = ""

        # Update status
        if self.dataset_directory.value.strip():
            self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'><strong>🔄 Status:</strong> Upload reset. Folder '{self.dataset_directory.value}' ready for new files.</div>"
        else:
            self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'><strong>🔄 Status:</strong> Upload reset. Create a folder and select files to upload.</div>"

        print("🔄 Upload widget reset! Ready for new file selection.")

    def auto_detect_existing_datasets(self):
        """Auto-detect existing dataset directories and set the first one as default"""

        # Look for existing datasets directories
        datasets_pattern = "datasets/*"
        existing_dirs = [d for d in glob.glob(datasets_pattern) if os.path.isdir(d)]

        if existing_dirs:
            # Sort by modification time (most recent first)
            existing_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            most_recent = existing_dirs[0]

            # Auto-populate if dataset directory is empty
            if not self.dataset_directory.value.strip():
                self.dataset_directory.value = most_recent
                print(f"📁 Auto-detected existing dataset: {most_recent}")

                # Update status to show detection
                self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #17a2b8;'><strong>🔍 Status:</strong> Auto-detected dataset folder '{most_recent}'. Select files to upload here.</div>"

    def launch_fiftyone_explorer(self, b):
        """Launch FiftyOne dataset explorer in sidecar"""
        dataset_path = self.dataset_directory.value
        if not dataset_path or not os.path.exists(dataset_path):
            print("❌ Please set up a dataset first")
            return

        # Check for sidecar availability first
        try:
            from sidecar import Sidecar
            has_sidecar = True
        except ImportError:
            has_sidecar = False
            print("⚠️ Sidecar not available - FiftyOne will open in browser tab")
            print("   Install with: pip install sidecar")

        try:
            import fiftyone as fo

            # Create dataset
            print("📁 Loading dataset for FiftyOne...")
            dataset = self.manager.create_fiftyone_dataset(dataset_path)
            print(f"✅ Dataset loaded: {len(dataset)} images")

            if has_sidecar:
                # Use sidecar integration
                print("🔧 Opening FiftyOne in Jupyter sidecar...")

                # Create the sidecar
                self.fiftyone_sidecar = Sidecar(
                    title='🔍 Dataset Explorer - FiftyOne',
                    anchor='split-right'
                )

                # Launch FiftyOne session in sidecar context
                with self.fiftyone_sidecar:
                    session = fo.launch_app(dataset, auto=False, height=800)
                    self.current_fo_session = session  # Store for curation

                print("✅ FiftyOne launched in sidecar panel!")
                print("   Use the panel on the right to explore your dataset")

            else:
                # Fallback to browser-based launch
                print("🌐 Opening FiftyOne in browser...")
                session = fo.launch_app(dataset, auto=True)
                self.current_fo_session = session
                print("✅ FiftyOne launched in browser tab")

            # Add optional analysis (using real data structure)
            try:
                print("📊 Running dataset analysis...")

                # Basic stats
                print(f"   📁 Total images: {len(dataset)}")

                # Check for captions
                captioned_samples = 0
                total_tags = 0
                trigger_words = set()

                for sample in dataset:
                    if hasattr(sample, 'tag_count') and sample.tag_count > 0:
                        captioned_samples += 1
                        total_tags += sample.tag_count
                    if hasattr(sample, 'trigger_word') and sample.trigger_word:
                        trigger_words.add(sample.trigger_word)

                print(f"   📝 Captioned images: {captioned_samples}/{len(dataset)} ({captioned_samples/len(dataset)*100:.1f}%)")
                print(f"   🏷️ Total tags: {total_tags}")
                print(f"   🎯 Unique triggers: {len(trigger_words)}")

                # Show trigger words
                if trigger_words:
                    print("   🔥 Trigger words found:")
                    for trigger in sorted(trigger_words):
                        print(f"      - {trigger}")

                # Show top tags if available
                if len(dataset) > 0:
                    first_sample = dataset.first()
                    if hasattr(first_sample, 'wd14_tags') and first_sample.wd14_tags:
                        print("   ✅ WD14 tags loaded and available for editing")
                    else:
                        print("   ℹ️ No WD14 tags found - may need to run tagging first")

            except Exception as e:
                print(f"   ⚠️ Analysis failed: {e}")

        except ImportError:
            print("❌ FiftyOne not available. Install with:")
            print("   pip install fiftyone")
        except Exception as e:
            print(f"❌ Failed to launch FiftyOne: {e}")
            print(f"   Error details: {str(e)}")

    def apply_fiftyone_curation(self, b):
        """Apply changes made in FiftyOne back to the local dataset."""
        dataset_path = self.dataset_directory.value
        if not dataset_path or not os.path.exists(dataset_path):
            print("❌ Please set up a dataset first")
            return

        # Check if we have an active FiftyOne session
        if not hasattr(self, 'current_fo_session') or self.current_fo_session is None:
            print("❌ No active FiftyOne session found")
            print("   Please launch the FiftyOne Explorer first")
            return

        print("💾 Applying FiftyOne curation changes to local dataset...")
        try:
            # Get the current dataset from the session
            dataset = self.current_fo_session.dataset
            print(f"📊 Processing {len(dataset)} samples...")

            # This will call the new method in DatasetManager with FiftyOne dataset
            success = self.manager.apply_curation_to_dataset(dataset_path, fiftyone_dataset=dataset)
            if success:
                print("✅ Curation changes applied successfully!")
                print("🔄 Changes applied:")
                print("   - Updated captions from FiftyOne edits")
                print("   - Synced tag modifications")
                print("   - Applied any sample filtering")
            else:
                print("❌ Failed to apply curation changes.")
        except Exception as e:
            print(f"❌ Error applying curation: {e}")
            print("   Make sure FiftyOne Explorer is open and contains your dataset")

    def run_gallery_dl_scraper(self, b):
        """Handle image scraping using gallery-dl"""
        self.dataset_output.clear_output()
        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Preparing gallery-dl scrape...</div>"

        with self.dataset_output:
            site = self.gallery_dl_site.value
            tags = self.gallery_dl_tags.value.strip()
            limit_range = self.gallery_dl_limit_range.value.strip()
            custom_url = self.gallery_dl_custom_url.value.strip()
            sub_folder = self.gallery_dl_sub_folder.value.strip()
            additional_args = self.gallery_dl_additional_args.value.strip()
            write_tags = self.gallery_dl_write_tags.value
            use_aria2c = self.gallery_dl_use_aria2c.value

            # Determine the final dataset directory
            # Use the folder_name and repeat_count from the direct upload section
            # This ensures Kohya-compatible folder naming for scraped datasets
            folder_name_base = self.folder_name.value.strip() if self.folder_name.value.strip() else "scraped_dataset"
            repeat_count = self.folder_repeats.value
            kohya_folder_name = f"{repeat_count}_{folder_name_base}"
            dataset_dir = f"datasets/{kohya_folder_name}"

            # If custom URL is selected, ensure it's provided
            if site == 'custom' and not custom_url:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please provide a Custom URL when 'Custom URL' site is selected.</div>"
                print("❌ Please provide a Custom URL when 'Custom URL' site is selected.")
                return

            if not tags and not custom_url:
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please enter tags or a Custom URL.</div>"
                print("❌ Please enter tags or a Custom URL.")
                return

            try:
                os.makedirs(dataset_dir, exist_ok=True)
                self.dataset_directory.value = dataset_dir # Update shared dataset directory

                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Scraping with gallery-dl...</div>"

                success = self.manager.scrape_with_gallery_dl(
                    site=site,
                    tags=tags,
                    dataset_dir=dataset_dir,
                    limit_range=limit_range,
                    write_tags=write_tags,
                    use_aria2c=use_aria2c,
                    custom_url=custom_url,
                    sub_folder=sub_folder,
                    additional_args=additional_args
                )

                if success:
                    try:
                        from core.image_utils import count_images_in_directory
                        image_count = count_images_in_directory(dataset_dir)
                        print(f"\n📁 Total images in dataset: {image_count}")
                        print(f"📝 Dataset directory set: {dataset_dir}")

                        self.dataset_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Scrape complete. Found {image_count} images.</div>"
                    except Exception as e:
                        print(f"⚠️ Image counting error: {e}")
                        self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Scrape complete.</div>"
                else:
                    self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Gallery-DL scrape failed. Check logs.</div>"

            except Exception as e:
                print(f"❌ Failed to run gallery-dl scraper: {e}")
                self.dataset_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Scraper encountered an error.</div>"

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
            print("\n🎉 Cleanup complete!")
            print(f"📊 Deleted {deleted_count} files")
            print(f"💾 Freed {size_freed_mb:.2f} MB of disk space")
            print(f"📁 Dataset directory: {dataset_dir}")

            if deleted_count > 0:
                self.cleanup_status.value = f"<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Cleaned {deleted_count} files ({size_freed_mb:.1f} MB freed).</div>"
            else:
                self.cleanup_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #ffc107;'><strong>⚠️ Status:</strong> No files were deleted (may have been errors).</div>"

    def run_add_trigger(self, b):
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Adding trigger word...</div>"
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
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Removing tags...</div>"
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
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Processing search and replace...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please set up a dataset first.</div>"
                print("❌ Please set up a dataset first in the Dataset Setup section.")
                return

            if not self.search_tags.value.strip():
                self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Please specify tags to search for.</div>"
                print("❌ Please specify tags to search for.")
                return

            print("🔍 Search and replace operation:")
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
        self.caption_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>⚙️ Status:</strong> Organizing tags...</div>"
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

            print("📋 Tag organization:")
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

    def run_preview_rename(self, b):
        """Preview file renaming changes"""
        with self.rename_output:
            self.rename_output.clear_output()

            if not self.rename_project_name.value.strip():
                print("⚠️ Please enter a project name.")
                return

            if not self.dataset_directory.value.strip():
                print("⚠️ Please set up a dataset directory first.")
                return

            project_name = self.rename_project_name.value.strip()
            pattern = self.rename_pattern.value
            start_num = self.rename_start_number.value

            print(f"🔍 Preview renaming for: {self.dataset_directory.value}")
            print(f"📝 Pattern: {pattern}, Project: {project_name}, Start: {start_num}")

            preview_data = self.manager.preview_rename_files(
                self.dataset_directory.value,
                project_name,
                pattern,
                start_num
            )

            if preview_data:
                print(f"\n📋 Preview of {len(preview_data)} files to be renamed:")
                print("=" * 60)
                for i, item in enumerate(preview_data[:10]):  # Show first 10
                    status = "📸" if not item['has_caption'] else "📸📝"
                    print(f"{status} {item['old_name']} → {item['new_name']}")
                    if item['has_caption']:
                        print(f"    {item['old_caption']} → {item['new_caption']}")

                if len(preview_data) > 10:
                    print(f"... and {len(preview_data) - 10} more files")

                print("=" * 60)
                print("💡 If this looks good, click 'Rename Files' to proceed!")

                # Enable the rename button
                self.rename_files_button.disabled = False

                # Store preview data for actual rename
                self._current_preview = preview_data
            else:
                self.rename_files_button.disabled = True

    def run_rename_files(self, b):
        """Actually rename the files"""
        with self.rename_output:
            self.rename_output.clear_output()

            if not hasattr(self, '_current_preview') or not self._current_preview:
                print("⚠️ Please preview the changes first!")
                return

            project_name = self.rename_project_name.value.strip()
            pattern = self.rename_pattern.value

            print("📝 Renaming files...")
            print(f"🎯 Project: {project_name}, Pattern: {pattern}")

            success = self.manager.rename_dataset_files(
                self.dataset_directory.value,
                project_name,
                pattern,
                self.rename_start_number.value,
                self._current_preview
            )

            if success:
                print("✅ File renaming completed successfully!")
                print("💡 Caption files were automatically renamed to match!")
                # Clear preview data and disable button
                self._current_preview = None
                self.rename_files_button.disabled = True
            else:
                print("❌ File renaming failed. Check the logs above.")

    def run_upload_to_huggingface(self, b):
        """Upload dataset to HuggingFace Hub"""
        self.upload_output.clear_output()
        self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #007bff;'><strong>🚀 Status:</strong> Starting upload...</div>"

        with self.upload_output:
            # Auto-fill dataset path if empty
            if not self.upload_dataset_path.value and self.dataset_directory.value:
                self.upload_dataset_path.value = self.dataset_directory.value
                print(f"📁 Auto-filled dataset path: {self.dataset_directory.value}")

            # Validation
            if not self.upload_dataset_path.value:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Dataset path required.</div>"
                print("❌ Please specify the dataset path to upload.")
                return

            if not self.upload_dataset_name.value:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Repository name required.</div>"
                print("❌ Please specify a repository name.")
                return

            if not self.upload_hf_token.value:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> HuggingFace token required.</div>"
                print("❌ Please provide your HuggingFace WRITE token.")
                print("🔑 Get it here: https://huggingface.co/settings/tokens")
                return

            print("🤗 Starting HuggingFace dataset upload...")
            print("=" * 60)

            try:
                success = self.manager.upload_dataset_to_huggingface(
                    dataset_path=self.upload_dataset_path.value,
                    dataset_name=self.upload_dataset_name.value,
                    hf_token=self.upload_hf_token.value,
                    orgs_name=self.upload_orgs_name.value,
                    make_private=self.upload_make_private.value,
                    description=self.upload_description.value
                )

                if success:
                    self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Upload completed successfully!</div>"
                    print("\n🎉 Upload completed successfully!")
                    print("🤗 Your dataset is now available on HuggingFace Hub!")

                    # Create shareable info
                    repo_id = f"{self.upload_orgs_name.value.strip() or 'your-username'}/{self.upload_dataset_name.value}"
                    print("\n📋 Dataset Details:")
                    print(f"   🔗 URL: https://huggingface.co/datasets/{repo_id}")
                    print(f"   📝 Name: {self.upload_dataset_name.value}")
                    print(f"   🔒 Privacy: {'Private' if self.upload_make_private.value else 'Public'}")

                else:
                    self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload failed.</div>"
                    print("❌ Upload failed. Check the error messages above.")

            except Exception as e:
                self.upload_status.value = "<div style='background: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Upload error.</div>"
                print(f"💥 Unexpected error during upload: {str(e)}")
                print("💡 Please check your token permissions and try again.")

    def run_format_conversion(self, b):
        """Convert image formats in the current dataset directory."""
        self.image_utils_output.clear_output()
        dataset_path = self.dataset_directory.value.strip()

        target_format = self.format_selector.value
        quality = self.conversion_quality.value

        if not dataset_path:
            self.image_utils_output.append_stdout("❌ Please set up a dataset directory first.\n")
            return

        if not os.path.exists(dataset_path):
            self.image_utils_output.append_stdout(f"❌ Dataset directory not found: {dataset_path}\n")
            return

        self.image_utils_output.append_stdout(f"🔄 Starting format conversion to {target_format.upper()} (quality: {quality})...\n")
        try:
            # Use the existing image conversion functionality (need to implement in manager)
            success = self.manager.convert_image_formats(dataset_path, target_format, quality)
            if success:
                self.image_utils_output.append_stdout("✅ Format conversion complete.\n")
            else:
                self.image_utils_output.append_stdout("❌ Format conversion failed. Check logs.\n")
        except Exception as e:
            self.image_utils_output.append_stdout(f"❌ An error occurred during format conversion: {e}\n")

    def run_review_tags(self, b):
        """Display the generated tags for review"""
        self.caption_output.clear_output()
        self.caption_status.value = "<div style='background: #e2e3e5; padding: 8px; border-radius: 5px; border-left: 4px solid #6c757d;'><strong>📋 Status:</strong> Displaying tags...</div>"
        with self.caption_output:
            if not self.dataset_directory.value:
                print("❌ Please specify a dataset directory first.")
                self.caption_status.value = "<div style='background: #f8d7da; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> No dataset directory specified.</div>"
                return

            # Display the tags
            success = self.manager.display_dataset_tags(self.dataset_directory.value)
            if success:
                self.caption_status.value = "<div style='background: #d4edda; padding: 8px; border-radius: 5px; border-left: 4px solid #28a745;'><strong>✅ Status:</strong> Tags displayed successfully.</div>"
            else:
                self.caption_status.value = "<div style='background: #f8d7da; padding: 8px; border-radius: 5px; border-left: 4px solid #dc3545;'><strong>❌ Status:</strong> Failed to display tags.</div>"

    def display(self):
        display(self.widget_box)
