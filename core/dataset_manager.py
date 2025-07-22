# core/dataset_manager.py
import subprocess
import os
import zipfile

class DatasetManager:
    def __init__(self, model_manager):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_manager = model_manager
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "sd_scripts")

    def extract_dataset(self, zip_path, extract_to_dir, hf_token=""):
        if not extract_to_dir:
            print("Error: Please specify a directory to extract to.")
            return

        # Handle Hugging Face URL
        if zip_path.startswith("https://huggingface.co/"):
            print("Downloading dataset from Hugging Face...")
            downloaded_zip_path = self.model_manager.download_file(zip_path, self.project_root, hf_token)
            if not downloaded_zip_path:
                print("Failed to download the dataset.")
                return
            zip_path = downloaded_zip_path

        if not os.path.exists(zip_path):
            print(f"Error: Zip file not found at '{zip_path}'")
            return

        extract_path = os.path.join(self.project_root, extract_to_dir)
        os.makedirs(extract_path, exist_ok=True)

        print(f"Extracting dataset to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as f:
                f.extractall(extract_path)
            print("Extraction complete.\n")
        except zipfile.BadZipFile:
            print("Error: The file is not a valid zip file.")
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
        finally:
            # Clean up downloaded zip file if it was from HF
            if 'downloaded_zip_path' in locals() and os.path.exists(downloaded_zip_path):
                os.remove(downloaded_zip_path)

    def tag_images(self, dataset_dir, method, tagger_model, threshold, blacklist_tags="", caption_extension=".txt"):
        """Enhanced image tagging with more options"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return

        if method == "anime":
            self._tag_images_wd14(dataset_path, tagger_model, threshold, blacklist_tags, caption_extension)
        elif method == "photo":
            self._tag_images_blip(dataset_path, caption_extension)
        else:
            print(f"❌ Unknown tagging method: {method}")
    
    def _tag_images_wd14(self, dataset_path, tagger_model, threshold, blacklist_tags, caption_extension):
        """Tag images using WD14 tagger"""
        venv_python = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        tagger_script = os.path.join(self.sd_scripts_dir, "finetune/tag_images_by_wd14_tagger.py")

        # Build command with enhanced options
        command = [
            venv_python, tagger_script,
            dataset_path,
            "--repo_id", tagger_model,
            "--thresh", str(threshold),
            "--batch_size", "8" if "v3" in tagger_model or "swinv2" in tagger_model else "1",
            "--max_data_loader_n_workers", "2",
            "--caption_extension", caption_extension,
            "--remove_underscore",  # Convert underscores to spaces
            "--onnx"  # Use ONNX for faster inference
        ]
        
        # Add blacklisted tags if provided
        if blacklist_tags:
            command.extend(["--undesired_tags", blacklist_tags.replace(" ", "")])

        print(f"🏷️ Tagging images with {tagger_model.split('/')[-1]} (threshold: {threshold})...")
        if blacklist_tags:
            print(f"🚫 Blacklisted tags: {blacklist_tags}")
            
        self._run_subprocess(command, "Image tagging")
    
    def _tag_images_blip(self, dataset_path, caption_extension):
        """Tag images using BLIP captioning"""
        venv_python = os.path.join(self.sd_scripts_dir, "venv/bin/python")
        blip_script = os.path.join(self.sd_scripts_dir, "finetune/make_captions.py")

        command = [
            venv_python, blip_script,
            dataset_path,
            "--beam_search",
            "--max_data_loader_n_workers", "2",
            "--batch_size", "8",
            "--min_length", "10",
            "--max_length", "75",
            "--caption_extension", caption_extension
        ]

        print(f"📸 Generating BLIP captions...")
        self._run_subprocess(command, "BLIP captioning")
    
    def _run_subprocess(self, command, task_name):
        """Helper to run subprocess with consistent error handling"""
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

            print(f"\n✅ {task_name} complete.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during {task_name.lower()}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during {task_name.lower()}: {e}")
            return False

    def add_trigger_word(self, dataset_dir, trigger_word):
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset directory not found at {dataset_path}")
            return

        if not trigger_word:
            print("Error: Please specify a trigger word.")
            return

        print(f"Adding trigger word '{trigger_word}' to caption files...")
        for item in os.listdir(dataset_path):
            if item.endswith(".txt"):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r+') as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write(f"{trigger_word}, {content}")
                    print(f"  Added to: {item}")
                except Exception as e:
                    print(f"Error processing {item}: {e}")
        print("✅ Trigger word addition complete.")
    
    def remove_tags(self, dataset_dir, tags_to_remove):
        """Remove specified tags from all caption files"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return

        if not tags_to_remove:
            print("❌ Error: Please specify tags to remove.")
            return
        
        # Parse tags (handle both comma and space separated)
        tags_list = [tag.strip() for tag in tags_to_remove.replace(',', ' ').split() if tag.strip()]
        
        print(f"🗑️ Removing tags from caption files: {', '.join(tags_list)}...")
        files_processed = 0
        
        for item in os.listdir(dataset_path):
            if item.endswith((".txt", ".caption")):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Remove each tag
                    modified = False
                    for tag in tags_list:
                        # Remove tag with various separators
                        patterns = [f"{tag}, ", f", {tag}", f" {tag}, ", f", {tag} ", tag]
                        for pattern in patterns:
                            if pattern in content:
                                content = content.replace(pattern, "")
                                modified = True
                    
                    # Clean up any double commas or extra spaces
                    content = ', '.join([t.strip() for t in content.split(',') if t.strip()])
                    
                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content + "\n")
                        files_processed += 1
                        print(f"  ✓ Updated: {item}")
                        
                except Exception as e:
                    print(f"⚠️ Error processing {item}: {e}")
        
        print(f"✅ Tag removal complete. {files_processed} files updated.")