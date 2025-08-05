# core/dataset_manager.py
import subprocess
import os
import sys
import zipfile
import platform

class DatasetManager:
    def __init__(self, model_manager):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_manager = model_manager
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")

    def _detect_environment_type(self):
        """Detect the current environment to choose the best tagger strategy"""
        # Check for cloud/rental GPU environments
        if (os.path.exists("/workspace") or 
            any(key in os.environ for key in ["VAST_CONTAINERLABEL", "COLAB_GPU"]) or
            "vastai" in platform.node().lower()):
            return "cloud_rental"
        
        # Check for Colab/Jupyter environments  
        if (os.path.exists("/content") or 
            "COLAB_GPU" in os.environ or
            "jupyter" in sys.modules):
            return "jupyter_colab"
            
        # Default to local environment
        return "local"
    
    def _get_tagger_script_path(self):
        """Get the appropriate tagger script based on environment"""
        env_type = self._detect_environment_type()
        
        if env_type == "cloud_rental":
            # Use robust custom tagger for unstable rental GPU environments
            custom_tagger = os.path.join(self.project_root, "custom", "tag_images_by_wd14_tagger.py")
            if os.path.exists(custom_tagger):
                print("🛡️ Using robust custom tagger for cloud/rental environment")
                return custom_tagger
        
        elif env_type == "jupyter_colab":
            # Use HoloStrawberry's version optimized for Colab/Jupyter
            holo_tagger = os.path.join(self.project_root, "kohya-colab-main", "tag_images_by_wd14_tagger.py")
            if os.path.exists(holo_tagger):
                print("📚 Using HoloStrawberry's tagger for Jupyter/Colab environment") 
                return holo_tagger
        
        # Fallback to Derrian's backend (Kohya-based)
        derrian_tagger = os.path.join(self.sd_scripts_dir, "finetune", "tag_images_by_wd14_tagger.py")
        if os.path.exists(derrian_tagger):
            print("⚙️ Using Derrian's backend tagger (Kohya-based)")
            return derrian_tagger
            
        # Final fallback to custom if nothing else works
        custom_tagger = os.path.join(self.project_root, "custom", "tag_images_by_wd14_tagger.py")
        print("🔄 Falling back to custom tagger as last resort")
        return custom_tagger

    def extract_dataset(self, zip_path, extract_to_dir, hf_token=""):
        if not extract_to_dir:
            print("Error: Please specify a directory to extract to.")
            return False

        # Handle Hugging Face URL
        if zip_path.startswith("https://huggingface.co/"):
            print("Downloading dataset from Hugging Face...")
            downloaded_zip_path = self.model_manager.download_file(zip_path, self.project_root, hf_token)
            if not downloaded_zip_path:
                print("Failed to download the dataset.")
                return False
            zip_path = downloaded_zip_path

        if not os.path.exists(zip_path):
            print(f"Error: Zip file not found at '{zip_path}'")
            return False

        extract_path = os.path.join(self.project_root, extract_to_dir)
        os.makedirs(extract_path, exist_ok=True)

        print(f"Extracting dataset to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as f:
                f.extractall(extract_path)
            print("Extraction complete.\n")
            return True
        except zipfile.BadZipFile:
            print("Error: The file is not a valid zip file.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
            return False
        finally:
            # Clean up downloaded zip file if it was from HF
            if 'downloaded_zip_path' in locals() and os.path.exists(downloaded_zip_path):
                os.remove(downloaded_zip_path)

    def tag_images(self, dataset_dir, method, tagger_model, threshold, blacklist_tags="", caption_extension=".txt"):
        """Enhanced image tagging with more options"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        if method == "anime":
            return self._tag_images_wd14(dataset_path, tagger_model, threshold, blacklist_tags, caption_extension)
        elif method == "photo":
            return self._tag_images_blip(dataset_path, caption_extension)
        else:
            print(f"❌ Unknown tagging method: {method}")
            return False
    
    def _tag_images_wd14(self, dataset_path, tagger_model, threshold, blacklist_tags, caption_extension):
        """Tag images using WD14 tagger"""
        # Known working tagger models (fallback list)
        working_models = [
            tagger_model,  # User's choice first
            "SmilingWolf/wd-vit-large-tagger-v3",  # Reliable fallback
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2"  # Stable older model
        ]
        
        # First ensure the tagger model is available
        self._ensure_tagger_model_available(tagger_model)
        
        # Try to find appropriate Python executable (cross-platform)
        if sys.platform == "win32":
            venv_python = os.path.join(self.sd_scripts_dir, "venv", "Scripts", "python.exe")
        else:
            venv_python = os.path.join(self.sd_scripts_dir, "venv", "bin", "python")
            
        if not os.path.exists(venv_python):
            # Fallback to system python if venv doesn't exist
            venv_python = sys.executable
            print(f"⚠️ Kohya venv not found, using system Python: {venv_python}")
        
        tagger_script = self._get_tagger_script_path()
        
        # Check if tagger script exists
        if not os.path.exists(tagger_script):
            print(f"❌ Tagger script not found at: {tagger_script}")
            print("💡 This usually means the SD-scripts setup is incomplete")
            print("💡 Try running the setup widget first to install the training backend")
            return False

        # Try each model until one works
        for attempt, model_to_try in enumerate(working_models):
            print(f"🏷️ Attempting tagging with {model_to_try.split('/')[-1]} (threshold: {threshold})...")
            
            # Build command with enhanced options
            command = [
                venv_python, tagger_script,
                dataset_path,
                "--repo_id", model_to_try,
                "--thresh", str(threshold),
                "--batch_size", "8" if "v3" in model_to_try or "swinv2" in model_to_try else "1",
                "--max_data_loader_n_workers", "2",
                "--caption_extension", caption_extension,
                "--remove_underscore",  # Convert underscores to spaces
                "--character_tags_first"  # Put character tags before general tags
            ]
            
            # Force ONNX usage for modern models
            if "v3" in model_to_try:
                command.append("--onnx")
                print("🚀 Using ONNX for v3 model (faster inference)")
            else:
                # Try to add ONNX flag if available for older models
                try:
                    test_onnx_cmd = [venv_python, "-c", "import onnxruntime"]
                    subprocess.run(test_onnx_cmd, check=True, capture_output=True)
                    command.append("--onnx")
                    print("✅ ONNX available - using accelerated inference")
                except subprocess.CalledProcessError:
                    print("⚠️ ONNX not available - using standard PyTorch inference")
            
            # Add blacklisted tags if provided
            if blacklist_tags:
                command.extend(["--undesired_tags", blacklist_tags.replace(" ", "")])
                print(f"🚫 Blacklisted tags: {blacklist_tags}")
            
            # Try running the command
            if self._run_subprocess(command, f"Image tagging (attempt {attempt + 1})"):
                print(f"✅ Successfully tagged with {model_to_try.split('/')[-1]}")
                return True
            else:
                print(f"❌ Failed with {model_to_try.split('/')[-1]}")
                if attempt < len(working_models) - 1:
                    print(f"🔄 Trying fallback model...")
                    
        print("❌ All tagger models failed")
        return False
    
    def _tag_images_blip(self, dataset_path, caption_extension):
        """Tag images using BLIP captioning"""
        # Try to find appropriate Python executable (cross-platform)
        if sys.platform == "win32":
            venv_python = os.path.join(self.sd_scripts_dir, "venv", "Scripts", "python.exe")
        else:
            venv_python = os.path.join(self.sd_scripts_dir, "venv", "bin", "python")
            
        if not os.path.exists(venv_python):
            # Fallback to system python if venv doesn't exist
            venv_python = sys.executable
            print(f"⚠️ Kohya venv not found, using system Python: {venv_python}")
        
        blip_script = os.path.join(self.sd_scripts_dir, "finetune/make_captions.py")
        
        # Check if BLIP script exists
        if not os.path.exists(blip_script):
            print(f"❌ BLIP script not found at: {blip_script}")
            print("💡 This usually means the SD-scripts setup is incomplete")
            print("💡 Try running the setup widget first to install the training backend")
            return False

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
        return self._run_subprocess(command, "BLIP captioning")
    
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
            return False

        if not trigger_word:
            print("Error: Please specify a trigger word.")
            return False

        try:
            print(f"Adding trigger word '{trigger_word}' to caption files...")
            files_processed = 0
            for item in os.listdir(dataset_path):
                if item.endswith(".txt"):
                    file_path = os.path.join(dataset_path, item)
                    try:
                        with open(file_path, 'r+') as f:
                            content = f.read()
                            f.seek(0, 0)
                            f.write(f"{trigger_word}, {content}")
                        print(f"  Added to: {item}")
                        files_processed += 1
                    except Exception as e:
                        print(f"Error processing {item}: {e}")
            print(f"✅ Trigger word addition complete. Processed {files_processed} files.")
            return True
        except Exception as e:
            print(f"❌ Error adding trigger word: {e}")
            return False
    
    def remove_tags(self, dataset_dir, tags_to_remove):
        """Remove specified tags from all caption files"""
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        if not tags_to_remove:
            print("❌ Error: Please specify tags to remove.")
            return False
        
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
        return True
    
    def _ensure_tagger_model_available(self, tagger_model):
        """Ensure the tagger model is downloaded and available"""
        print(f"🔍 Checking tagger model availability: {tagger_model}")
        
        # Try to import huggingface_hub to check model availability
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            import torch
        except ImportError:
            print("⚠️ huggingface_hub or torch not available - model download may fail")
            print("💡 The tagger script will attempt to download automatically")
            return
        
        try:
            # Check if model is already cached
            model_path = snapshot_download(repo_id=tagger_model, allow_patterns=["*.onnx", "*.json"])
            print(f"✅ Tagger model found at: {model_path}")
            
        except Exception as e:
            print(f"📥 Downloading tagger model {tagger_model}...")
            try:
                # Download the model files we need
                files_to_download = [
                    "model.onnx",
                    "selected_tags.csv", 
                    "config.json"
                ]
                
                for file in files_to_download:
                    try:
                        hf_hub_download(repo_id=tagger_model, filename=file)
                        print(f"  ✅ Downloaded: {file}")
                    except Exception as file_error:
                        print(f"  ⚠️ Could not download {file}: {file_error}")
                        
                print(f"✅ Tagger model {tagger_model.split('/')[-1]} ready!")
                
            except Exception as download_error:
                print(f"⚠️ Model download failed: {download_error}")
    
    def scrape_from_gelbooru(self, tags, dataset_dir, limit=100, confirm_callback=None):
        """
        Scrape and download images from Gelbooru based on tags
        
        Args:
            tags (str): Gelbooru tags (e.g., "1girl, blue_hair, -solo")
            dataset_dir (str): Directory to save images
            limit (int): Maximum number of images to download
            confirm_callback (callable): Function to call for confirmation before download
        """
        import requests
        import os
        from urllib.parse import urlparse
        
        print(f"🔍 Searching Gelbooru for: {tags}")
        print(f"📁 Will save to: {dataset_dir}")
        print(f"📊 Max images: {limit}")
        
        # Ask for confirmation if callback provided
        if confirm_callback and not confirm_callback():
            print("❌ Download cancelled by user")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Build Gelbooru API URL
            api_url = "https://gelbooru.com/index.php"
            params = {
                "page": "dapi",
                "s": "post", 
                "q": "index",
                "json": "1",
                "tags": tags,
                "limit": min(limit, 1000)  # Gelbooru max is 1000
            }
            
            print("📡 Fetching image URLs from Gelbooru...")
            
            # Add headers to avoid 401 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 401:
                print("❌ Gelbooru API returned 401 Unauthorized")
                print("💡 This might be due to:")
                print("   - Rate limiting (wait a few minutes)")
                print("   - API changes requiring authentication")
                print("   - Blocked IP or user agent")
                print("   - Try using different tags or smaller limits")
                return False
            
            response.raise_for_status()
            
            data = response.json()
            if not data or not isinstance(data, list):
                print("❌ No images found or invalid response from Gelbooru")
                return False
                
            print(f"✅ Found {len(data)} images matching your tags")
            
            # Extract image URLs
            image_urls = []
            for post in data:
                if 'file_url' in post and post['file_url']:
                    # Filter for common image formats
                    file_url = post['file_url']
                    if any(file_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        image_urls.append(file_url)
            
            if not image_urls:
                print("❌ No valid image URLs found")
                return False
                
            print(f"📥 Starting download of {len(image_urls)} images...")
            
            # Download images using aria2c (if available) or fallback to requests
            downloaded_count = 0
            
            # Try using aria2c first (faster for multiple files)
            try:
                # Create temporary URL list file for aria2c
                url_file = os.path.join(dataset_dir, "temp_urls.txt")
                with open(url_file, 'w') as f:
                    for url in image_urls:
                        f.write(f"{url}\n")
                
                # Use aria2c for fast parallel downloads
                import subprocess
                cmd = [
                    'aria2c',
                    '-i', url_file,
                    '-d', dataset_dir,
                    '-j', '4',  # 4 parallel downloads
                    '-x', '2',  # 2 connections per download
                    '--auto-file-renaming=false',
                    '--allow-overwrite=true'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Clean up temp file
                os.remove(url_file)
                
                if result.returncode == 0:
                    downloaded_count = len([f for f in os.listdir(dataset_dir) 
                                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                    print(f"✅ Downloaded {downloaded_count} images using aria2c")
                else:
                    raise Exception("aria2c failed, falling back to requests")
                    
            except Exception as e:
                print(f"⚠️ aria2c not available or failed: {e}")
                print("📥 Falling back to slower direct downloads...")
                
                # Fallback: download using requests
                for i, url in enumerate(image_urls):
                    try:
                        # Get filename from URL
                        parsed_url = urlparse(url)
                        filename = os.path.basename(parsed_url.path)
                        if not filename or '.' not in filename:
                            filename = f"gelbooru_image_{i+1}.jpg"
                            
                        filepath = os.path.join(dataset_dir, filename)
                        
                        # Skip if file already exists
                        if os.path.exists(filepath):
                            downloaded_count += 1
                            continue
                            
                        # Download the image
                        img_response = requests.get(url, timeout=30, stream=True)
                        img_response.raise_for_status()
                        
                        with open(filepath, 'wb') as f:
                            for chunk in img_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                
                        downloaded_count += 1
                        if downloaded_count % 10 == 0:
                            print(f"📥 Downloaded {downloaded_count}/{len(image_urls)} images...")
                            
                    except Exception as img_error:
                        print(f"⚠️ Failed to download {url}: {img_error}")
                        continue
            
            print(f"🎉 Successfully downloaded {downloaded_count} images to {dataset_dir}")
            return downloaded_count > 0
            
        except Exception as e:
            print(f"❌ Gelbooru scraping failed: {e}")
            return False
    
    def search_and_replace_tags(self, dataset_dir, search_tags, replace_with="", search_mode="AND"):
        """
        Search and replace tags across all caption files in dataset
        
        Args:
            dataset_dir (str): Dataset directory path
            search_tags (str): Tags to search for (comma or space separated)
            replace_with (str): What to replace with (empty string to remove)
            search_mode (str): "AND" or "OR" - how to match multiple search tags
        """
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        if not search_tags.strip():
            print("❌ Error: Please specify tags to search for.")
            return False
        
        # Parse search tags
        search_list = [tag.strip() for tag in search_tags.replace(',', ' ').split() if tag.strip()]
        replace_text = replace_with.strip() if replace_with else ""
        
        print(f"🔍 Searching for tags: {', '.join(search_list)} (mode: {search_mode})")
        if replace_text:
            print(f"🔄 Will replace with: '{replace_text}'")
        else:
            print("🗑️ Will remove matching tags")
        
        files_processed = 0
        total_replacements = 0
        
        for item in os.listdir(dataset_path):
            if item.endswith((".txt", ".caption")):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    original_content = content
                    
                    # Split content into tags
                    tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                    
                    # Apply search and replace logic
                    new_tags = []
                    file_replacements = 0
                    
                    for tag in tags:
                        tag_matches = []
                        for search_tag in search_list:
                            if search_tag.lower() in tag.lower():
                                tag_matches.append(True)
                            else:
                                tag_matches.append(False)
                        
                        # Check if tag should be replaced based on search mode
                        should_replace = False
                        if search_mode.upper() == "AND":
                            should_replace = all(tag_matches)
                        else:  # OR mode
                            should_replace = any(tag_matches)
                        
                        if should_replace:
                            if replace_text:
                                new_tags.append(replace_text)
                            # If replace_text is empty, tag is removed (not added to new_tags)
                            file_replacements += 1
                        else:
                            new_tags.append(tag)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_tags = []
                    for tag in new_tags:
                        if tag not in seen:
                            unique_tags.append(tag)
                            seen.add(tag)
                    
                    new_content = ', '.join(unique_tags)
                    
                    if new_content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content + "\n")
                        files_processed += 1
                        total_replacements += file_replacements
                        print(f"  ✓ Updated {item}: {file_replacements} replacements")
                        
                except Exception as e:
                    print(f"⚠️ Error processing {item}: {e}")
        
        print(f"✅ Search and replace complete. {files_processed} files updated, {total_replacements} total replacements.")
        return True
    
    def sort_and_deduplicate_tags(self, dataset_dir, sort_alphabetically=True, remove_duplicates=True):
        """
        Sort tags alphabetically and/or remove duplicate tags within each caption file
        
        Args:
            dataset_dir (str): Dataset directory path
            sort_alphabetically (bool): Whether to sort tags alphabetically
            remove_duplicates (bool): Whether to remove duplicate tags
        """
        dataset_path = os.path.join(self.project_root, dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset directory not found at {dataset_path}")
            return False

        print(f"🔧 Processing caption files...")
        if sort_alphabetically:
            print("   📝 Sorting tags alphabetically")
        if remove_duplicates:
            print("   🗑️ Removing duplicate tags")
        
        files_processed = 0
        total_duplicates_removed = 0
        
        for item in os.listdir(dataset_path):
            if item.endswith((".txt", ".caption")):
                file_path = os.path.join(dataset_path, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if not content:
                        continue
                    
                    # Split content into tags
                    tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                    original_count = len(tags)
                    
                    # Remove duplicates if requested
                    if remove_duplicates:
                        # Use dict.fromkeys() to preserve order while removing duplicates
                        # Convert to lowercase for comparison but keep original case
                        seen = {}
                        unique_tags = []
                        for tag in tags:
                            tag_lower = tag.lower()
                            if tag_lower not in seen:
                                seen[tag_lower] = True
                                unique_tags.append(tag)
                        tags = unique_tags
                    
                    # Sort alphabetically if requested
                    if sort_alphabetically:
                        tags.sort(key=str.lower)
                    
                    new_content = ', '.join(tags)
                    
                    # Write back if content changed
                    if new_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content + "\n")
                        
                        duplicates_removed = original_count - len(tags)
                        total_duplicates_removed += duplicates_removed
                        files_processed += 1
                        
                        if duplicates_removed > 0:
                            print(f"  ✓ {item}: removed {duplicates_removed} duplicate(s)")
                        else:
                            print(f"  ✓ {item}: sorted")
                        
                except Exception as e:
                    print(f"⚠️ Error processing {item}: {e}")
        
        if total_duplicates_removed > 0:
            print(f"✅ Processing complete. {files_processed} files updated, {total_duplicates_removed} duplicates removed.")
        else:
            print(f"✅ Processing complete. {files_processed} files updated.")
        return True