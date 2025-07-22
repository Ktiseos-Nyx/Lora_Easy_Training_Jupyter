#!/usr/bin/env python3
"""
Personal LoRA Training Calculator
Your "Stop Being a Chicken" Math Helper
"""

def main():
    print("🎯 Personal LoRA Training Calculator")
    print("=" * 40)
    
    # Get basic info
    try:
        images = int(input("How many images do you have? "))
        print(f"\nYou have {images} images...")
        
        # Dataset size assessment
        if images <= 10:
            print("🐣 TINY DATASET - Time to be brave!")
            size_category = "tiny"
        elif images <= 15:
            print("🐤 SMALL DATASET - Your peers are right!")
            size_category = "small"
        elif images <= 30:
            print("🐔 MEDIUM DATASET - Sweet spot territory")
            size_category = "medium"
        elif images <= 50:
            print("🦅 LARGE DATASET - Plenty to work with")
            size_category = "large"
        else:
            print("🦆 HUGE DATASET - Stop being a chicken, you have tons!")
            size_category = "huge"
        
        # Training type
        print("\nWhat are you training?")
        print("1. Character LoRA (person/character)")
        print("2. Style LoRA (art style)")
        print("3. Concept LoRA (objects/ideas)")
        
        choice = input("Pick (1-3): ").strip()
        training_type = {"1": "character", "2": "style", "3": "concept"}.get(choice, "character")
        
        print(f"\n📊 RECOMMENDATIONS FOR {training_type.upper()} LoRA:")
        print("=" * 50)
        
        # Calculate recommendations based on size and type
        if size_category == "tiny":  # ≤10 images
            repeats = 20
            epochs = 15
            unet_lr = "3e-4"
            te_lr = "5e-5"
            batch_size = 1
            dim_alpha = "8/4"
            
        elif size_category == "small":  # 11-15 images  
            repeats = 15
            epochs = 12
            unet_lr = "4e-4" 
            te_lr = "8e-5"
            batch_size = 2
            dim_alpha = "8/4"
            
        elif size_category == "medium":  # 16-30 images
            repeats = 10
            epochs = 10
            unet_lr = "5e-4"
            te_lr = "1e-4" 
            batch_size = 4
            dim_alpha = "8/4"
            
        elif size_category == "large":  # 31-50 images
            repeats = 8
            epochs = 8
            unet_lr = "5e-4"
            te_lr = "1e-4"
            batch_size = 4
            dim_alpha = "8/4"
            
        else:  # huge >50 images
            repeats = 6
            epochs = 6
            unet_lr = "5e-4"
            te_lr = "1e-4"
            batch_size = 4
            dim_alpha = "16/8" if training_type == "style" else "8/4"
        
        # Style LoRA adjustments
        if training_type == "style":
            unet_lr_val = float(unet_lr.replace("e-", "E-"))
            te_lr_val = float(te_lr.replace("e-", "E-"))
            unet_lr = f"{unet_lr_val * 0.6:.0e}".replace("e-0", "e-")
            te_lr = f"{te_lr_val * 0.5:.0e}".replace("e-0", "e-")
            epochs = int(epochs * 1.5)
            print("📝 Style LoRA detected - using lower learning rates & more epochs")
        
        # Calculate total steps
        total_steps = (images * repeats * epochs) // batch_size
        
        # Display recommendations
        print(f"📸 Images: {images}")
        print(f"🔄 Repeats: {repeats}")
        print(f"📅 Epochs: {epochs}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"🎛️ Network: {dim_alpha} (dim/alpha)")
        print(f"🧠 UNet Learning Rate: {unet_lr}")
        print(f"📝 Text Encoder LR: {te_lr}")
        print(f"⚡ Total Steps: {total_steps}")
        
        # Step assessment
        if total_steps < 250:
            print("⚠️  WARNING: Too few steps! Increase repeats or epochs")
        elif total_steps < 400:
            print("🔶 Low steps - might work but consider more repeats")
        elif 400 <= total_steps <= 1000:
            print("✅ PERFECT step range!")
        elif total_steps <= 1500:
            print("🟡 High steps - should work, might be overkill")
        else:
            print("🔴 Too many steps! Reduce repeats/epochs or increase batch size")
        
        # Advanced suggestions
        print(f"\n🧪 ADVANCED MODE SUGGESTIONS:")
        print("━" * 30)
        
        if size_category in ["tiny", "small"]:
            print("🚀 Optimizer: CAME (saves VRAM, gentler)")
            print("📊 Scheduler: Cosine (stable)")
            print("💾 Memory: Enable all optimizations")
            print("🦄 LyCORIS: Try DoRA if you have time")
        else:
            print("🚀 Optimizer: CAME or AdamW8bit")  
            print("📊 Scheduler: Cosine with 3 restarts")
            if training_type == "style":
                print("🦄 LyCORIS: Try (IA)³ for styles")
            else:
                print("🦄 LyCORIS: DoRA for higher quality")
        
        # Confidence booster
        print(f"\n💪 CONFIDENCE BOOSTER:")
        print("━" * 20)
        if size_category in ["tiny", "small"]:
            print("🐤 Your peers train with this few images ALL THE TIME!")
            print("🎯 Quality > Quantity - you've got this!")
            print("⏱️  Faster iteration = more experiments!")
        else:
            print("🦅 You have plenty of images - stop worrying!")
            print("🎯 This is a comfortable dataset size!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Bye! Go train some LoRAs!")
    except ValueError:
        print("❌ Please enter a valid number of images!")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")

if __name__ == "__main__":
    main()