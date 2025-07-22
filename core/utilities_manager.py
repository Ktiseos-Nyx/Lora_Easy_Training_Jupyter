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
        if not hf_token:
            print("Error: Hugging Face token is required.")
            return
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        if not repo_name:
            print("Error: Repository name is required.")
            return

        try:
            login(token=hf_token)
            api = HfApi()
            
            filename = os.path.basename(model_path)
            repo_id = f"{api.whoami()['name']}/{repo_name}"

            print(f"Uploading {filename} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=repo_id,
                commit_message=f"Upload {filename}"
            )
            print(f"Upload complete. View your model at https://huggingface.co/{repo_id}/blob/main/{filename}")

        except Exception as e:
            print(f"An error occurred during Hugging Face upload: {e}")

    def resize_lora(self, input_path, output_path, new_dim, new_alpha):
        if not os.path.exists(input_path):
            print(f"Error: Input LoRA file not found at {input_path}")
            return
        if not output_path:
            print("Error: Please specify an output path for the resized LoRA.")
            return
        if not new_dim or not new_alpha:
            print("Error: Please specify new dim and alpha values.")
            return

        venv_python = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        resize_script = os.path.join(self.sd_scripts_dir, "networks/resize_lora.py")

        if not os.path.exists(resize_script):
            print(f"Error: LoRA resize script not found at {resize_script}. Please ensure your trainer backend is correctly installed.")
            return

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
                cwd=self.project_root
            )

            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

            print("\nLoRA resizing complete.")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while resizing LoRA: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def count_dataset_files(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path not found at {dataset_path}")
            return

        image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        caption_extensions = (".txt", ".caption")

        image_count = 0
        caption_count = 0
        other_files = 0

        print(f"Counting files in {dataset_path}...")
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