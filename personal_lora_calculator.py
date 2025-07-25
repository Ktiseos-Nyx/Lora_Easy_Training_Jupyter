#!/usr/bin/env python3
"""
Personal LoRA Training Calculator
A pure, unbiased step calculator.
"""
import os
import glob
import sys
import subprocess

def count_images_in_directory(directory_path):
    """Count image files in a directory"""
    if not os.path.exists(directory_path):
        return 0
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
    image_count = 0
    
    # Use a set to store found file paths to avoid double counting
    found_files = set()

    for ext in image_extensions:
        # Search recursively
        pattern_recursive = os.path.join(directory_path, '**', ext)
        for file in glob.glob(pattern_recursive, recursive=True):
            found_files.add(file)
        
        # Search non-recursively
        pattern_non_recursive = os.path.join(directory_path, ext)
        for file in glob.glob(pattern_non_recursive):
            found_files.add(file)
            
    return len(found_files)

def main():
    print("🎯 Personal LoRA Training Calculator")
    print("=" * 40)
    
    try:
        # --- Step 1: Get Dataset Size ---
        print("How do you want to specify your dataset size?")
        print("1. Enter number of images manually")
        print("2. Point to a directory (I'll count for you!)")
        
        method = input("Choose method (1 or 2): ").strip()
        
        if method == "2":
            directory = input("Enter the path to your dataset directory: ").strip()
            images = count_images_in_directory(directory)
            if images == 0:
                print(f"❌ No images found in '{directory}'. Please check the path or try manual entry.")
                return
            print(f"📁 Found {images} images in the directory.")
        else:
            images = int(input("Enter the total number of images in your dataset: "))
        
        print("\n--- Step 2: Enter Training Parameters ---")
        
        # --- Get user input for calculation ---
        repeats = int(input("Enter the number of repeats: "))
        epochs = int(input("Enter the number of epochs: "))
        batch_size = int(input("Enter the training batch size: "))

        # --- Step 3: Calculate Total Steps ---
        if batch_size <= 0:
            print("❌ Batch size must be greater than zero.")
            return
            
        total_steps = (images * repeats * epochs) // batch_size

        # --- Step 4: Display Results ---
        print("\n--- Calculation Results ---")
        print(f"📸 Images:       {images}")
        print(f"🔄 Repeats:      {repeats}")
        print(f"📅 Epochs:       {epochs}")
        print(f"📦 Batch Size:   {batch_size}")
        print("=" * 27)
        print(f"⚡ Total Steps:  {total_steps})
        
        print("\nYour steps and epochs with that dataset is Doro.")

        # --- Step 5: Display Doro ---
        try:
            # Get the absolute path to the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the path to the image relative to the script
            image_path = os.path.join(script_dir, 'assets', 'doro.png')

            if os.path.exists(image_path):
                if sys.platform == "win32":
                    os.startfile(image_path)
                elif sys.platform == "darwin": # macOS
                    subprocess.run(["open", image_path], check=True)
                else: # linux
                    subprocess.run(["xdg-open", image_path], check=True)
            else:
                print("\n(Doro image not found at '{image_path}')")
        except Exception as e:
            print(f"\n(Could not open Doro image: {e})")
            
    except KeyboardInterrupt:
        print("\n\n👋 Bye! Go train some LoRAs!")
    except ValueError:
        print("❌ Invalid input. Please enter a valid number.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
