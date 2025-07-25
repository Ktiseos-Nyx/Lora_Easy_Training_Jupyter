# core/utilities_manager.py
import subprocess
import os
from huggingface_hub import HfApi, login

class UtilitiesManager:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "sd_scripts")

    def upload_to_huggingface(self, hf_token, model_path, repo_name):
        if not hf_token or not hf_token.strip():
            print("❌ Error: Hugging Face token is required.")
            print("💡 Get your token from: https://huggingface.co/settings/tokens")
            return False
        if not model_path or not model_path.strip():
            print("❌ Error: Model file path is required.")
            return False
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at {model_path}")
            return False
        if not repo_name or not repo_name.strip():
            print("❌ Error: Repository name is required.")
            print("💡 Example: 'my-awesome-loras' (no spaces, lowercase preferred)")
            return False
        
        # Validate file extension
        if not model_path.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
            print("⚠️ Warning: File doesn't appear to be a model file (.safetensors, .ckpt, .pt, .pth)")
            print("🤔 Continuing anyway...")

        try:
            login(token=hf_token)
            api = HfApi()
            
            filename = os.path.basename(model_path)
            repo_id = f"{api.whoami()['name']}/{repo_name}"

            print(f"🚀 Uploading {filename} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=repo_id,
                commit_message=f"Upload {filename}",
                repo_type="model"
            )
            print(f"✅ Upload complete!")
            print(f"🔗 View your model at: https://huggingface.co/{repo_id}")
            print(f"📁 Direct file link: https://huggingface.co/{repo_id}/blob/main/{filename}")
            return True

        except Exception as e:
            print(f"❌ Error during Hugging Face upload: {e}")
            if "Invalid token" in str(e):
                print("💡 Check your HuggingFace token: https://huggingface.co/settings/tokens")
            elif "Repository not found" in str(e):
                print("💡 Repository will be created automatically on first upload")
            return False

    def resize_lora(self, input_path, output_path, new_dim, new_alpha):
        if not os.path.exists(input_path):
            print(f"Error: Input LoRA file not found at {input_path}")
            return False
        if not output_path:
            print("Error: Please specify an output path for the resized LoRA.")
            return False
        if not new_dim or not new_alpha:
            print("Error: Please specify new dim and alpha values.")
            return False

        # Check for venv python first, fall back to system python
        venv_python_path = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        if os.path.exists(venv_python_path):
            venv_python = venv_python_path
        else:
            venv_python = "python"  # Use system python (common in containers)
        
        resize_script = os.path.join(self.sd_scripts_dir, "networks/resize_lora.py")

        if not os.path.exists(resize_script):
            print(f"Error: LoRA resize script not found at {resize_script}")
            print("💡 Please ensure the trainer environment setup completed successfully.")
            return False

        command = [
            venv_python, resize_script,
            "--save_precision", "fp16",
            "--save_to", output_path,
            "--model", input_path,
            "--new_rank", str(new_dim),
            "--new_alpha", str(new_alpha)
        ]

        print(f"Resizing LoRA from {input_path} to {output_path} with dim={new_dim}, alpha={new_alpha}...")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.trainer_dir  # Run from trainer directory like training scripts
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

            print("\nLoRA resizing complete.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while resizing LoRA: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def count_dataset_files(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path not found at {dataset_path}")
            return False

        image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        caption_extensions = (".txt", ".caption")

        image_count = 0
        caption_count = 0
        other_files = 0

        print(f"Counting files in {dataset_path}...")
        try:
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_count += 1
                    elif file.lower().endswith(caption_extensions):
                        caption_count += 1
                    else:
                        other_files += 1
            
            print(f"\nResults for {dataset_path}:")
            print(f"  Images: {image_count}")
            print(f"  Captions: {caption_count}")
            print(f"  Other files: {other_files}")
            return True
            
        except Exception as e:
            print(f"Error counting files: {e}")
            return False