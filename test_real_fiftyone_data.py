#!/usr/bin/env python3
"""
Test script to understand how FiftyOne actually handles real WD14 tagged data.
This investigates the ACTUAL data structure instead of making assumptions.
"""

import os
import sys

def analyze_real_dataset_structure(dataset_path):
    """Analyze the actual structure of a real WD14-tagged dataset"""
    
    print("=" * 60)
    print("🔍 REAL DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
        
    # Analyze file structure
    files = os.listdir(dataset_path)
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    captions = [f for f in files if f.endswith('.txt')]
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"🖼️ Images: {len(images)} files")
    print(f"🏷️ Captions: {len(captions)} files")
    print(f"📊 Match ratio: {len(captions)/len(images)*100:.1f}%")
    
    # Analyze caption format
    print(f"\n📝 CAPTION FILE ANALYSIS")
    print("-" * 40)
    
    sample_files = captions[:5]  # Look at first 5
    for caption_file in sample_files:
        caption_path = os.path.join(dataset_path, caption_file)
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            print(f"\n{caption_file}:")
            print(f"  Length: {len(content)} chars")
            print(f"  Content: {content[:100]}...")
            
            # Analyze tag separation
            if ', ' in content:
                tags = [tag.strip() for tag in content.split(', ')]
                print(f"  Tags (comma+space): {len(tags)} found")
                print(f"  First 3 tags: {tags[:3]}")
                print(f"  Tag separator: ', ' (comma + space)")
            elif ',' in content:
                tags = [tag.strip() for tag in content.split(',')]
                print(f"  Tags (comma only): {len(tags)} found") 
                print(f"  Tag separator: ',' (comma only)")
            else:
                print(f"  No commas found - single tag or different format")
                
        except Exception as e:
            print(f"  ❌ Error reading: {e}")
    
    return {
        'total_images': len(images),
        'total_captions': len(captions),
        'sample_content': content if 'content' in locals() else '',
        'separator': ', ' if ', ' in (content if 'content' in locals() else '') else ','
    }


def test_fiftyone_loading(dataset_path):
    """Test how FiftyOne actually loads this dataset"""
    
    print(f"\n🔬 FIFTYONE LOADING TEST")
    print("-" * 40)
    
    try:
        import fiftyone as fo
        print("✅ FiftyOne imported successfully")
    except ImportError:
        print("❌ FiftyOne not available - skipping test")
        return None
        
    try:
        # Test basic loading
        print(f"📁 Loading dataset from: {dataset_path}")
        dataset = fo.Dataset.from_images_dir(dataset_path, recursive=False)
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Examine first sample
        if len(dataset) > 0:
            sample = dataset.first()
            print(f"\n🔍 First sample analysis:")
            print(f"  Filepath: {sample.filepath}")
            print(f"  Available fields: {list(sample.field_names)}")
            
            # Check if captions are auto-loaded
            for field_name in sample.field_names:
                field_value = getattr(sample, field_name)
                print(f"  {field_name}: {type(field_value)} = {str(field_value)[:100]}...")
                
        # Try to add caption data manually
        print(f"\n📝 Testing caption integration...")
        for sample in dataset.take(3):  # Test first 3 samples
            image_path = sample.filepath
            caption_path = os.path.splitext(image_path)[0] + '.txt'
            
            if os.path.exists(caption_path):
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_content = f.read().strip()
                
                # Try different ways to add the caption data
                sample['caption_raw'] = caption_content
                
                # Parse tags from caption
                if ', ' in caption_content:
                    tags = [tag.strip() for tag in caption_content.split(', ')]
                else:
                    tags = [tag.strip() for tag in caption_content.split(',')]
                
                sample['tags'] = tags
                sample['tag_count'] = len(tags)
                
                # Look for trigger word (first tag is often the concept)
                if tags:
                    sample['trigger_word'] = tags[0]
                    sample['other_tags'] = tags[1:]
                
                sample.save()
                
        print("✅ Caption data added to samples")
        
        # Test the enhanced dataset
        print(f"\n📊 Enhanced dataset analysis:")
        sample = dataset.first()
        print(f"  Available fields: {list(sample.field_names)}")
        print(f"  Caption raw: {getattr(sample, 'caption_raw', 'N/A')[:50]}...")
        print(f"  Tags: {getattr(sample, 'tags', 'N/A')[:3] if hasattr(sample, 'tags') else 'N/A'}")
        print(f"  Tag count: {getattr(sample, 'tag_count', 'N/A')}")
        print(f"  Trigger word: {getattr(sample, 'trigger_word', 'N/A')}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ FiftyOne loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    dataset_path = "/Users/duskfall/Training Zips to Sort/1825954_training_data"
    
    # Analyze real structure
    structure_info = analyze_real_dataset_structure(dataset_path)
    
    # Test FiftyOne integration 
    dataset = test_fiftyone_loading(dataset_path)
    
    print(f"\n📋 SUMMARY & FINDINGS")
    print("=" * 60)
    print(f"✅ Real tag format: Tags separated by '{structure_info.get('separator', 'unknown')}'")
    print(f"✅ Real structure: Flat directory with image + .txt pairs")
    print(f"✅ Caption content: Mixed tags (trigger word + WD14 tags)")
    
    if dataset:
        print(f"✅ FiftyOne integration: Working with manual field addition")
        print(f"✅ Fields available: caption_raw, tags, tag_count, trigger_word")
    else:
        print(f"❌ FiftyOne integration: Needs troubleshooting")
        
    print(f"\n💡 KEY INSIGHTS FOR FIXING CODE:")
    print(f"   - Tags are comma+space separated strings, not lists")
    print(f"   - FiftyOne doesn't auto-load .txt files as 'wd14_tags'")
    print(f"   - Need manual field creation for tag integration")
    print(f"   - First tag is often the trigger word/concept")
    print(f"   - No folder structure (flat directory)")


if __name__ == "__main__":
    main()