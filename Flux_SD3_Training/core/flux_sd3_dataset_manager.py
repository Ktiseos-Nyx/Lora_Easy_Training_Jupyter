# Flux_SD3_Training/core/flux_sd3_dataset_manager.py
import sys
import os

# Add parent directory to path to import from main core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from dataset_manager import DatasetManager

class FluxSD3DatasetManager(DatasetManager):
    """🚀 Flux/SD3 Dataset Manager
    
    Inherits from the main DatasetManager since Flux/SD3 LoRA training uses:
    - Same image files
    - Same .txt caption files
    - Same tagging process (Danbooru/BLIP)
    
    Adds Flux/SD3-specific preprocessing and resolution handling.
    """
    
    def __init__(self):
        super().__init__()
        print("🚀 Flux/SD3 Dataset Manager initialized!")
        print("📸 Uses same image + caption dataset as standard LoRA training")
        print("🎯 Optimized for 1024x1024 resolution and advanced text encoders")
    
    def preprocess_captions_for_t5(self, caption_text: str, model_type: str = 'auraflow_t5') -> str:
        """T5-specific caption preprocessing"""
        
        if not caption_text.strip():
            return caption_text
        
        # T5 models sometimes benefit from specific formatting
        if model_type == 'auraflow_t5':
            # AuraFlow T5 might prefer more structured prompts
            if not caption_text.startswith(('A photo of', 'An image of', 'A picture of')):
                # Add context prefix for better T5 understanding
                caption_text = f"A detailed image showing: {caption_text}"
        
        elif model_type == 'hidream_t5':
            # HiDream might prefer more descriptive language
            # Keep tags but make them more natural
            if ',' in caption_text:  # Danbooru-style tags
                tags = [tag.strip() for tag in caption_text.split(',')]
                # Convert first few tags to natural language
                if len(tags) > 2:
                    main_subject = tags[0]
                    descriptors = ', '.join(tags[1:3])
                    rest_tags = ', '.join(tags[3:]) if len(tags) > 3 else ''
                    caption_text = f"{main_subject} with {descriptors}"
                    if rest_tags:
                        caption_text += f", {rest_tags}"
        
        return caption_text.strip()
    
    def process_dataset_for_t5(self, dataset_dir: str, model_type: str = 'auraflow_t5', 
                              max_caption_length: int = 256) -> bool:
        """Process existing dataset for T5 training with optional caption enhancement"""
        
        print(f"🔄 Processing dataset for T5 ({model_type})...")
        
        if not os.path.exists(dataset_dir):
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return False
        
        # Get all caption files
        caption_files = []
        for file in os.listdir(dataset_dir):
            if file.lower().endswith('.txt'):
                caption_files.append(os.path.join(dataset_dir, file))
        
        if not caption_files:
            print("❌ No caption files found in dataset")
            return False
        
        print(f"📝 Processing {len(caption_files)} caption files...")
        
        processed_count = 0
        for caption_file in caption_files:
            try:
                # Read original caption
                with open(caption_file, 'r', encoding='utf-8') as f:
                    original_caption = f.read().strip()
                
                if not original_caption:
                    continue
                
                # Apply T5-specific preprocessing
                processed_caption = self.preprocess_captions_for_t5(original_caption, model_type)
                
                # Truncate if too long (T5 has sequence limits)
                if len(processed_caption) > max_caption_length:
                    processed_caption = processed_caption[:max_caption_length].rsplit(' ', 1)[0]
                    if not processed_caption.endswith('.'):
                        processed_caption += '...'
                
                # Write back if changed
                if processed_caption != original_caption:
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(processed_caption)
                    processed_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {caption_file}: {e}")
                continue
        
        print(f"✅ T5 dataset processing complete!")
        print(f"📊 Processed {processed_count} captions for {model_type}")
        return True
    
    def validate_t5_dataset(self, dataset_dir: str) -> dict:
        """Validate dataset for T5 training and return statistics"""
        
        stats = {
            'total_images': 0,
            'total_captions': 0,
            'avg_caption_length': 0,
            'max_caption_length': 0,
            'missing_captions': 0,
            'valid_pairs': 0
        }
        
        if not os.path.exists(dataset_dir):
            print(f"❌ Dataset directory not found: {dataset_dir}")
            return stats
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []
        captions = []
        caption_lengths = []
        
        # Collect all files
        for file in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    images.append(file)
                elif ext == '.txt':
                    captions.append(file)
                    
                    # Check caption length
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            caption_text = f.read().strip()
                            if caption_text:
                                caption_lengths.append(len(caption_text))
                    except:
                        pass
        
        stats['total_images'] = len(images)
        stats['total_captions'] = len(captions)
        
        if caption_lengths:
            stats['avg_caption_length'] = sum(caption_lengths) / len(caption_lengths)
            stats['max_caption_length'] = max(caption_lengths)
        
        # Check for matching image-caption pairs
        for image in images:
            image_name = os.path.splitext(image)[0]
            caption_file = f"{image_name}.txt"
            
            if caption_file in captions:
                stats['valid_pairs'] += 1
            else:
                stats['missing_captions'] += 1
        
        # Print validation results
        print(f"📊 T5 Dataset Validation Results:")
        print(f"   📸 Images: {stats['total_images']}")
        print(f"   📝 Captions: {stats['total_captions']}")
        print(f"   ✅ Valid pairs: {stats['valid_pairs']}")
        print(f"   ❌ Missing captions: {stats['missing_captions']}")
        print(f"   📏 Avg caption length: {stats['avg_caption_length']:.1f} chars")
        print(f"   📏 Max caption length: {stats['max_caption_length']} chars")
        
        if stats['missing_captions'] > 0:
            print(f"⚠️ Warning: {stats['missing_captions']} images missing captions")
        
        if stats['max_caption_length'] > 512:
            print(f"⚠️ Warning: Some captions exceed recommended T5 length (512 chars)")
        
        return stats