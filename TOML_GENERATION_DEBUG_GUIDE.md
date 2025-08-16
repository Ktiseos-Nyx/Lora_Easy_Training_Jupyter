# TOML Generation Debug Guide

## 🚨 CRITICAL DEBUGGING PROTOCOL 🚨

**Context:** Multiple expensive sessions lost to TOML field mapping mistakes. This guide provides step-by-step debugging to prevent $10+ losses.

## Problem Pattern Recognition

### Symptom: Empty TOML Sections
```toml
[network_arguments]

[optimizer_arguments]

[training_arguments]
```

**ROOT CAUSE:** Field mapping mismatch between widget output and TOML generation code.
**NOT:** Missing data from widget.

### Symptom: Training Assertion Errors
```
AssertionError: resolution is required / resolution（解像度）指定は必須です
```

**ROOT CAUSE:** Required fields not reaching Kohya's training scripts due to field name mismatches.

## Debugging Steps (MANDATORY ORDER)

### Step 1: Capture Widget Debug Output
**ALWAYS run first** - Get actual widget field names:

```python
# In kohya_training_manager.py start_training method:
logger.info("🎭 === WIDGET CONFIG DEBUG DUMP ===")
logger.info(f"📊 Full config keys: {list(config.keys())}")
logger.info(f"📊 Config type: {type(config)}")
```

### Step 2: Verify Key Field Values
Check critical fields have actual values (not None):

```python
logger.info(f"📊 model_path: {repr(config.get('model_path'))}")
logger.info(f"📊 dataset_path: {repr(config.get('dataset_path'))}")
logger.info(f"📊 resolution: {repr(config.get('resolution'))}")
logger.info(f"📊 network_dim: {repr(config.get('network_dim'))}")
logger.info(f"📊 network_alpha: {repr(config.get('network_alpha'))}")
logger.info(f"📊 unet_lr: {repr(config.get('unet_lr'))}")
```

### Step 3: Check Git History for Working Implementation
```bash
# Find last working TOML generation
git log --oneline -10 -- core/*training_manager.py

# Compare working vs current field mappings
git show HEAD~3:core/kohya_training_manager.py | grep -A 10 "config.get"
```

### Step 4: Verify Current TOML Field Mapping
Check that TOML generation uses EXACT widget field names:

**Common Widget → TOML Mappings:**
- Widget: `'model_path'` → TOML: `pretrained_model_name_or_path`
- Widget: `'dataset_path'` → TOML: `image_dir` (in subsets)
- Widget: `'resolution'` → TOML: `resolution` (in general)
- Widget: `'network_dim'` → TOML: `network_dim`
- Widget: `'network_alpha'` → TOML: `network_alpha`
- Widget: `'unet_lr'` → TOML: `learning_rate`

## Known Working Field Mappings

### Config TOML (network/optimizer/training arguments)
```python
# WORKING PATTERN - Use widget field names directly:
toml_config = {
    "network_arguments": {
        "network_dim": config.get('network_dim'),           # Widget provides this
        "network_alpha": config.get('network_alpha'),       # Widget provides this
        "network_module": "networks.lora",                  # Static value
    },
    "optimizer_arguments": {
        "learning_rate": config.get('unet_lr'),             # Widget field name
        "optimizer_type": config.get('optimizer'),          # Widget field name
    },
    "training_arguments": {
        "pretrained_model_name_or_path": config.get('model_path'),  # Widget field name
        "max_train_epochs": config.get('epochs'),           # Widget field name
        "train_batch_size": config.get('train_batch_size'), # Widget field name
    }
}
```

### Dataset TOML (datasets/subsets/general)
```python
# WORKING PATTERN - Map widget paths correctly:
dataset_config = {
    "datasets": [{
        "subsets": [{
            "image_dir": config.get('dataset_path'),    # Widget provides dataset_path
            "num_repeats": config.get('num_repeats'),   # Widget provides this
        }]
    }],
    "general": {
        "resolution": config.get('resolution'),         # Widget provides this
        "shuffle_caption": config.get('shuffle_caption'),
        "flip_aug": config.get('flip_aug'),
    }
}
```

## Validation Checklist

Before deploying TOML generation fixes:

- [ ] Widget debug output shows all required fields have values (not None)
- [ ] TOML generation uses exact widget field names from debug output
- [ ] Generated TOML files have populated sections (not empty)
- [ ] Critical fields (model_path, dataset_path, resolution, network_dim) appear in TOML
- [ ] Test with actual widget on server before declaring "fixed"

## Emergency Rollback

If TOML generation breaks:

1. **Find last working commit:**
   ```bash
   git log --oneline -- core/kohya_training_manager.py
   ```

2. **Extract working TOML methods:**
   ```bash
   git show WORKING_COMMIT:core/kohya_training_manager.py | grep -A 50 "def create_.*_toml"
   ```

3. **Restore working field mappings** - Don't "improve" them!

## Cost Prevention Rules

1. **NEVER touch TOML generation without widget debug output first**
2. **NEVER assume field names** - Always verify with actual widget data
3. **NEVER "improve" working TOML code** without explicit user request
4. **ALWAYS test on server** before declaring success

**Remember:** Empty TOML sections = Field mapping mismatch, NOT missing widget data!