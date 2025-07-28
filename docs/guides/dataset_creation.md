# Guide: Creating and Managing Datasets

This guide covers all the methods available in the Dataset Widget for building high-quality datasets, from simple uploads to advanced scraping and curation.

## Getting Started

Open `Dataset_Maker_Widget.ipynb` and run Cell 2 to display the main dataset widget. This single interface handles all your dataset preparation needs.

## Section 1: Upload Methods

### ZIP File Upload (Recommended)

The most straightforward method for getting a large number of images into the project at once.

1. **Prepare Your ZIP File**: On your local machine, gather all your training images into a single folder. Compress this folder into a `.zip` file.
2. **Use the Upload Interface**: In the Dataset Widget, find the "Dataset Upload" section.
3. **Upload**: Click the upload button and select your ZIP file. The widget will automatically extract the contents into a new folder in your project directory.
4. **Set the Path**: The dataset path field will be automatically populated with the path to your extracted folder.

### Individual File Upload

Use this method if you want to add a few images to an existing dataset or build one from scratch.

1. **Select Target Directory**: Use the file browser in the widget to navigate to your desired dataset folder.
2. **Upload Files**: Use the individual file upload button to add one or more images.
3. **Supported Formats**: JPG, PNG, WebP are supported.

### HuggingFace Dataset Import

Download datasets directly from HuggingFace repositories.

1. **Enter HF URL**: In the "HuggingFace URL" field, paste the link to a HuggingFace dataset.
2. **Optional Token**: If the dataset is private, enter your HuggingFace token.
3. **Download**: The widget will handle the download and extraction automatically.

## Section 2: Auto-Tagging Systems

### WD14 v3 Tagger (Anime/Art Content)

The WD14 (Waifu Diffusion 1.4) tagger is specifically trained on anime and artistic content.

**Available Models:**
- **wd14-vit-v2**: General purpose, balanced accuracy and speed
- **wd14-convnext-v2**: Higher accuracy, slower processing  
- **wd14-swinv2-v2**: Best for complex scenes and compositions
- **wd14-convnext-v3**: Latest model with improved accuracy

**Configuration:**
1. **Select Model**: Choose the appropriate WD14 model for your content
2. **Set Threshold**: 0.35 is recommended for characters, 0.4 for complex art
3. **Blacklist Tags**: Optional - enter unwanted tags separated by commas
4. **Caption Extension**: Usually `.txt` (default)

**Best Practices:**
- Use higher thresholds (0.4-0.5) for cleaner tag sets
- Use lower thresholds (0.3-0.35) for more comprehensive tagging
- The convnext and swinv2 models are more accurate but slower

### BLIP Captioning (Realistic Photos)

BLIP generates natural language descriptions ideal for photographic content.

**Features:**
- **Natural Language**: Creates sentence-based descriptions
- **Scene Understanding**: Captures relationships and context
- **Quality Control**: Beam search for better generation

**Best For:**
- Real photographs and portraits
- Realistic landscapes and scenes
- When you want descriptive sentences rather than tags

**Configuration:**
1. **Select BLIP Method**: Choose BLIP from the tagging options
2. **Set Parameters**: The widget uses optimized settings automatically
3. **Caption Extension**: Usually `.txt` (default)

## Section 3: Caption Management

### Bulk Editing Tools

**Add Trigger Words:**
1. **Enter Trigger Word**: Type your unique trigger word (e.g., "saria_zelda")
2. **Choose Position**: Beginning, end, or random placement in captions
3. **Apply to All**: Adds the trigger word to every caption file

**Find and Replace:**
1. **Search Terms**: Enter tags or phrases to find
2. **Replace With**: Enter replacement text (leave empty to remove)
3. **Apply Changes**: Updates all matching instances across the dataset

**Remove Unwanted Tags:**
1. **Enter Tags to Remove**: List unwanted tags separated by commas
2. **Removal Mode**: Choose to remove all instances or just specific ones
3. **Clean Up**: Automatically fixes formatting after removal

### Advanced Curation Tools

**Sort and Deduplicate:**
- **Alphabetical Sorting**: Organizes tags in consistent order
- **Duplicate Removal**: Eliminates repeated tags within captions
- **Format Standardization**: Ensures consistent comma separation

**Tag Analysis:**
- **Frequency Analysis**: See most and least common tags
- **Tag Distribution**: Understand your dataset's vocabulary
- **Quality Metrics**: Check caption completeness and consistency

## Section 4: Content Scraping (Advanced)

### Gelbooru Scraper

The built-in Gelbooru scraper allows automatic download of anime/art images based on tags.

**⚠️ Important Notes:**
- Use responsibly and respect platform terms of service
- Always review downloaded content before using
- Be mindful of copyright and content policies

**How to Use:**
1. **Open Scraper Section**: Find the "Dataset Scraper" accordion in the widget
2. **Enter Tags**: Use Gelbooru tag syntax (e.g., "shirakami_fubuki solo high_quality")
3. **Set Limits**: Start with small numbers (50-100) to test quality
4. **Output Folder**: Choose destination folder name
5. **Start Scraping**: Click the button and monitor progress

**Tag Syntax Tips:**
- Use spaces to separate multiple tags
- Use `-` to exclude tags (e.g., "character_name -nsfw")
- Include quality tags like "high_quality" or "masterpiece"
- Be specific to get focused results

**Best Practices:**
- Start with small batches to check quality
- Use multiple quality filters
- Review and curate downloaded content
- Respect rate limits and platform policies

## Section 5: Quality Control

### Image Validation

The widget automatically checks for:
- **Supported Formats**: Ensures compatibility with training
- **File Integrity**: Detects corrupted images
- **Resolution**: Reports image dimensions
- **File Sizes**: Identifies unusually large or small files

### Caption Quality

**Length Optimization:**
- **Too Short**: Captions under 20 tokens may lack detail
- **Too Long**: Captions over 200 tokens may confuse training
- **Sweet Spot**: 50-150 tokens typically work best

**Consistency Checking:**
- **Tag Standardization**: Ensures same concepts use same terms
- **Trigger Word Verification**: Confirms all images have trigger words
- **Format Validation**: Checks for proper comma separation

### Dataset Statistics

The widget provides comprehensive statistics:
- **Image Count**: Total number of training images
- **Average Caption Length**: Helps optimize for training
- **Tag Frequency**: Most common tags in your dataset
- **Resolution Distribution**: Shows variety in image sizes

## Section 6: Best Practices by Content Type

### Character LoRAs

**Dataset Size**: 15-50 high-quality images
**Image Requirements:**
- Different poses and viewing angles
- Various expressions and emotions  
- Multiple outfits when relevant
- Consistent character design

**Tagging Strategy:**
- Focus on character-defining features
- Include consistent trigger word
- Tag distinctive clothing and accessories
- Note unique characteristics (eye color, hair style, etc.)

### Style LoRAs

**Dataset Size**: 50-200 images minimum
**Image Requirements:**
- Consistent artistic style and technique
- Similar color palette and rendering
- Unified aesthetic approach
- Various subjects in the same style

**Tagging Strategy:**
- Emphasize style descriptors
- Include medium and technique tags
- Note lighting and color characteristics
- Tag composition elements

### Concept LoRAs

**Dataset Size**: 20-100 images depending on complexity
**Image Requirements:**
- Clear representation of the concept
- Various contexts and applications
- High-quality examples
- Avoid conflicting interpretations

**Tagging Strategy:**
- Precise concept descriptions
- Include relevant context tags
- Note important attributes
- Use clear, unambiguous terms

## Section 7: Common Issues and Solutions

### Upload Problems

**"No images found in ZIP":**
- Check ZIP file structure (images should be in root or single subfolder)
- Verify file formats (only JPG, PNG, WebP supported)
- Ensure files aren't corrupted

**"Upload failed":**
- Check available disk space
- Verify file permissions
- Try smaller ZIP files

### Tagging Issues

**"Tagging failed":**
- Check internet connection for model downloads
- Ensure sufficient disk space (2-3GB for tagger models)
- Try a different tagger model
- Reduce batch size if memory issues occur

**"Tags too generic/specific":**
- Adjust threshold settings (lower = more tags, higher = fewer tags)
- Try different tagger models
- Use manual editing for fine-tuning

### Caption Problems

**"Captions too long/short":**
- Adjust tagging threshold
- Use tag filtering to remove excess tags
- Consider manual editing for important images

**"Inconsistent trigger words":**
- Use bulk edit tools to standardize
- Check that trigger words aren't being filtered out
- Verify trigger word placement settings

---

*Good datasets are the foundation of great LoRAs. Take time to prepare quality training data!*