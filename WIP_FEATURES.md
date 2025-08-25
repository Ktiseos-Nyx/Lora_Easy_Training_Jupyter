# 🚧 WIP Features Queue

**Queue of improvements, cleanups, and features that aren't critical bugs but would make the system better.**

## 🎯 High Priority Improvements

### UI/UX Cleanup
- **Smart conditional TOML parameter filtering** - Avoid including unused scheduler/optimizer params that break training (e.g., don't include `lr_scheduler_num_cycles` for simple cosine scheduler)
- **Remove sample generation settings box** from training config section (inference is disabled anyway)
- **Clean up "Advanced" settings section** - Remove features that are no longer advanced or are broken
- **Remove back pass functionality** - Doesn't work unless we can find the working code from another trainer
- **Move pretrained models** to `trainer/derrian_backend/sd_scripts/models/` (use Kohya-native paths instead of hardcoded `/workspace/` stuff)

## 🔮 Future Utility Ideas

### Dataset Management
- **Dataset analyzer** - Show image resolution distribution, file size stats, duplicate detection
- **Visual tag management system** - Bulk editing with thumbnails and search/filter
- **Caption editor** - Bulk find/replace operations across caption files
- **Tag frequency analyzer** - See which tags appear most/least in dataset
- **Dataset splitter** - Split large datasets into train/validation sets

### Training & Analysis
- **Training log parser** - Extract loss curves, generate graphs from training logs
- **Hyperparameter optimizer** - Suggest optimal learning rates based on hardware
- **Training time estimator** - "Your 500-image dataset will take ~2.3 hours"
- **VRAM optimizer** - Auto-calculate max batch size for user's GPU
- **Model benchmarker** - Quick quality tests on trained LoRAs

### Model Management  
- **Model merger** - Merge multiple LoRAs or blend with base models
- **Model inspector** - Show LoRA layer info, dimensions, compatibility
- **Checkpoint converter** - Convert between different formats (safetensors, ckpt, etc.)
- **LoRA naming wizard** - Generate creative names for trained models

### Fun Chaos 🎭
- **Random prompt generator** - For testing LoRAs with varied prompts
- **Training meme generator** - Generate memes based on loss curves ("This is fine" when loss explodes)

## 📋 Development Philosophy

**Be ruthless about feature removal:**
- Better to have clean, working UI than cluttered "maybe someone uses this" bloat
- If it's broken and we can't fix it easily, remove it
- If it's not actually "advanced" anymore, move it to basic settings or remove entirely
- Work WITH existing systems (like Kohya's expected paths) instead of fighting them

## 🚫 Things NOT To Do

- Don't add features just because they're technically possible
- Don't keep broken features "just in case" someone needs them
- Don't overcomplicate simple workflows
- Don't add advanced features that 99% of users won't understand or need

---

*This file tracks non-critical improvements. For actual bugs, see `CLAUDE_DEBUGGING.md`.*