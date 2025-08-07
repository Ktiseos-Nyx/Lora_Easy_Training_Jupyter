# core/demo_colored_print.py
# Demonstration script for colored print functionality
# Inspired by Linaqruf's colablib

from .colored_print import (
    cprint, print_line, print_header, print_status,
    success, error, warning, info, debug, progress
)

def demo_colored_output():
    """Demonstrate all colored print capabilities"""
    
    # Header demonstration
    print_header("LoRA Easy Training - Colored Output Demo", color="bright_blue", style="bold")
    print()
    
    # Status messages
    success("Environment setup completed successfully")
    error("Failed to download model checkpoint")
    warning("GPU memory usage is at 85%")
    info("Starting image tagging process")
    debug("Tensor shape: torch.Size([1, 4, 64, 64])")
    progress("Training epoch 5/10 - Step 250/500")
    print()
    
    # Color demonstrations
    print_header("Available Colors", length=50, char="-", color="bright_green")
    
    colors = ["red", "green", "yellow", "blue", "purple", "cyan", "white"]
    for color in colors:
        cprint(f"This text is {color}", color=color, style="bold")
    
    print()
    flat_colors = ["flat_red", "flat_yellow", "flat_blue", "flat_purple", "flat_orange", "flat_green", "flat_cyan", "flat_pink"]
    for color in flat_colors:
        cprint(f"This text is {color}", color=color, style="italic")
    
    print()
    
    # Style demonstrations  
    print_header("Available Styles", length=50, char="-", color="bright_purple")
    
    styles = ["normal", "bold", "italic", "underline"]
    for style in styles:
        cprint(f"This text uses {style} style", color="white", style=style)
    
    print()
    
    # Training progress simulation
    print_header("Training Progress Simulation", length=60, char="=", color="flat_orange")
    
    # Simulate typical LoRA training output
    info("Initializing LoRA training session")
    progress("Loading base model: SD1.5")
    success("Model loaded successfully (2.3GB)")
    progress("Processing dataset: 150 images")
    info("Using AdamW optimizer with learning rate 1e-4")
    
    for epoch in range(1, 4):
        progress(f"Epoch {epoch}/3 starting")
        for step in [50, 100, 150]:
            if step == 100:
                info(f"Epoch {epoch} - Step {step}/150 - Loss: 0.0123")
            elif step == 150:
                success(f"Epoch {epoch} completed - Avg Loss: 0.0089")
    
    print()
    warning("Training completed with minor issues")
    success("LoRA saved to output/my_character.safetensors")
    
    print()
    print_line(60, "=", "bright_green", "bold")
    cprint("🎉 Colored print demo completed!", color="bright_green", style="bold", timestamp=True)
    print_line(60, "=", "bright_green", "bold")

def demo_jupyter_friendly():
    """Demonstrate Jupyter-friendly output (with emojis instead of ANSI codes)"""
    
    print("=== Jupyter-Friendly Output Demo ===")
    print()
    
    # These will show emojis in Jupyter but colors in terminal
    success("Dataset preparation complete")
    error("Invalid file format detected")  
    warning("Low disk space remaining")
    info("Tagging 50 images with WD14 tagger")
    progress("Upload progress: 75% complete")
    
    print()
    print("This demo works in both Jupyter notebooks and terminal environments!")

if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_colored_output()
    print("\n" + "="*60 + "\n")
    demo_jupyter_friendly()