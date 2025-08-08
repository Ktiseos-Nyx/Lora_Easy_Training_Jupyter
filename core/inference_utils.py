import os
import subprocess
from pathlib import Path
import logging
from .managers import get_venv_python_path, get_subprocess_environment

logger = logging.getLogger(__name__)

def generate_sample_images(
    lora_path: str,
    base_model_path: str,
    prompt: str,
    num_samples: int,
    output_dir: str,
    epoch: int,
    resolution: int = 512,
    seed: int = 42,
    negative_prompt: str = "bad anatomy, bad hands, missing fingers, extra fingers, fewer fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
):
    """
    Generates sample images using a trained LoRA and a base Stable Diffusion model
    via the Kohya-SS sdxl_minimal_inference.py script.

    Args:
        lora_path (str): Path to the LoRA checkpoint.
        base_model_path (str): Path to the base Stable Diffusion model (CKPT/Safetensors).
        prompt (str): The positive prompt for image generation.
        num_samples (int): Number of sample images to generate.
        output_dir (str): Directory to save the generated images.
        epoch (int): Current training epoch (for naming output files).
        resolution (int): Resolution of the generated images (e.g., 512, 768, 1024).
        seed (int): Random seed for reproducibility.
        negative_prompt (str): The negative prompt for image generation.
        guidance_scale (float): Classifier-free guidance scale.
        num_inference_steps (int): Number of inference steps.
    """
    logger.info(f"✨ Generating {num_samples} sample images for epoch {epoch} using Kohya-SS inference...")
    logger.info(f"  Prompt: '{prompt}'")
    logger.info(f"  LoRA: {lora_path}")
    logger.info(f"  Base Model: {base_model_path}")

    # Determine the project root and script path dynamically
    project_root = Path(__file__).parents[1] # Go up two levels from core/inference_utils.py
    sd_scripts_dir = project_root / "trainer" / "derrian_backend" / "sd_scripts"
    inference_script = sd_scripts_dir / "sdxl_minimal_inference.py"

    if not inference_script.exists():
        logger.error(f"❌ Inference script not found at: {inference_script}")
        logger.error("   Please ensure the training backend is correctly installed.")
        return

    # Get proper venv python and environment (fixes CAME import issues!)
    venv_python = get_venv_python_path(str(project_root))
    if not os.path.exists(venv_python):
        logger.warning(f"⚠️ Virtual environment python not found at {venv_python}, using system python")
        venv_python = "python"
    
    # Get standardized subprocess environment
    env = get_subprocess_environment(str(project_root))

    # Create a specific output directory for this epoch's samples
    epoch_sample_output_dir = Path(output_dir) / "sample_images" / f"epoch_{epoch:03d}"
    epoch_sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Base command arguments (now using proper venv python!)
    cmd_args = [
        venv_python,
        str(inference_script),
        "--ckpt_path", str(base_model_path),
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--output_dir", str(epoch_sample_output_dir),
        "--steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--target_height", str(resolution),
        "--target_width", str(resolution),
        "--original_height", str(resolution),
        "--original_width", str(resolution),
    ]

    # Add LoRA weights if provided
    if lora_path and Path(lora_path).exists():
        # sdxl_minimal_inference.py expects 'path;multiplier'
        # We'll use a default multiplier of 1.0
        cmd_args.extend(["--lora_weights", f"{lora_path};1.0"])
    else:
        logger.warning(f"⚠️ LoRA checkpoint not found at {lora_path}. Generating samples without LoRA.")

    # Loop to generate multiple samples
    for i in range(num_samples):
        current_seed = seed + i # Vary seed for different samples
        sample_cmd = cmd_args + ["--seed", str(current_seed)]
        logger.info(f"  Running inference command (sample {i+1}/{num_samples}): {' '.join(map(str, sample_cmd))}")

        try:
            # Run the subprocess with proper environment and working directory
            process = subprocess.run(
                sample_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env,
                cwd=str(sd_scripts_dir)  # Run from scripts directory for proper imports
            )
            logger.info(f"  Inference output (sample {i+1}):\n{process.stdout}")
            if process.stderr:
                logger.warning(f"  Inference stderr (sample {i+1}):\n{process.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"🚨 Error generating sample image {i+1} for epoch {epoch}: {e}")
            logger.error(f"  Stdout: {e.stdout}")
            logger.error(f"  Stderr: {e.stderr}")
            break # Stop if one sample fails
        except Exception as e:
            logger.error(f"🚨 Unexpected error during sample generation {i+1} for epoch {epoch}: {e}")
            break # Stop if one sample fails

    logger.info(f"✅ Finished generating samples for epoch {epoch}.")