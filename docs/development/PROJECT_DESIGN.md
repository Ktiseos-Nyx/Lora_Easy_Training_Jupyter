# LoRA Training Notebook - Design Document

## Project Overview

**Goal**: Create **"Instructable Widgets"** for LoRA training in Jupyter - widgets that teach while they work, bridging the gap between intimidating command-line tools and inflexible GUI applications.

**User Profile**: Mid-range users, especially neurodivergent (DID/Autism/ADHD), with basic Python understanding who want guided but powerful training tools without complex server setups.

**Target Hardware**: 4090+ or rental GPU services (not optimizing for potato hardware)

**Positioning**: 
- **Command Line Tools** (Kohya) = Expert users, intimidating
- **GUI Applications** (OneTrainer) = Desktop users, less flexible  
- **Our Instructable Widgets** = Mid-range users who want guidance + power + learning

## Design Philosophy

### Core Principles (Borrowed from HollowStrawberry)
1. **Single-cell workflows** - Each major function in ONE cell
2. **Progressive disclosure** - Basic options visible, advanced hidden in extras
3. **Visual hierarchy** - Emoji sections for easy scanning
4. **Smart defaults** - Works out-of-box for 80% of users
5. **Forgiving interface** - Can re-run without breaking
6. **Clear validation** - Helpful error messages, not cryptic failures

### Neurodivergent-Friendly Features
- **Consistent patterns** across all widgets
- **Clear visual separation** between sections
- **Step-by-step guidance** with context
- **Error prevention** over error correction
- **Status feedback** throughout processes

### Validation from Existing Tools

**OneTrainer** validates our approach by showing:
- **GUI demand exists** - people want visual interfaces over command line
- **Tab-based organization works** - similar to our widget structure
- **VRAM optimization matters** - gradient checkpointing, layer offload
- **Dataset tools are essential** - auto-captioning, preprocessing
- **Modern model support needed** - v-pred, NoobAI compatibility

**HollowStrawberry** shows:
- **Progressive disclosure works** - basic → advanced options
- **Single-cell workflows** reduce cognitive load
- **Smart defaults** enable quick success
- **Inline documentation** teaches while configuring
- **Two-phase workflow** - Dataset Maker → Trainer separation
- **Step calculation formulas** - practical parameter guidance
- **Testing methodology** - X/Y/Z plots, epoch comparison

**Our "Instructable Widgets" advantage**:
- **More educational** than OneTrainer's GUI
- **More accessible** than Kohya's command line
- **More flexible** than Colab's limitations
- **More persistent** than web-based tools

## Widget Architecture

### 1. Setup & Models Widget 🚩
**Purpose**: Environment setup + model downloads
**Replaces**: Current cells 1-3

**Sections**:
- ▶️ Environment (auto-install toggle)
- ▶️ Base Model (dropdown + custom URL option)
- ▶️ VAE (optional, with guidance)
- ▶️ API Tokens (with help links)
- ▶️ Directories (project structure setup)

**Advanced Options** (collapsible):
- Custom model paths
- Force reinstall
- Debug mode

### 2. Dataset Manager Widget 📊
**Purpose**: Dataset upload, curation, tagging
**Inspired by**: HollowStrawberry's Dataset Maker

**Sections**:
- ▶️ Upload (zip/folder upload)
- ▶️ Curation (duplicate detection, manual review)
- ▶️ Tagging (WD14/BLIP with smart presets)
- ▶️ Caption Management (trigger words, editing)

**Advanced Options**:
- Custom taggers
- Batch operations
- Tag analysis

### 3. Training Configuration Widget ⭐
**Purpose**: All training settings in one place
**Inspired by**: HollowStrawberry's main trainer cell

**Sections**:
- ▶️ Basic Settings (resolution, epochs, batch size)
- ▶️ Learning (learning rates with presets)
- ▶️ LoRA Structure (dim/alpha with recommendations)
- ▶️ Optimization (scheduler, optimizer with explanations)

**Advanced Options**:
- Custom schedulers from Derrian Distro
- Advanced noise settings
- Experimental features

### 4. Training Execution Widget 🚀
**Purpose**: Run training with monitoring
**Features**:
- Progress tracking
- Loss visualization
- Live preview generation
- Error handling with suggestions

### 5. Utilities Widget 🔧
**Purpose**: Post-training tools
**Features**:
- LoRA resizing
- Model conversion
- Testing/validation tools

## Technical Architecture

### Dependencies Management
- **Core**: sd-scripts, accelerate, torch
- **Enhanced**: Custom schedulers from Derrian Distro
- **Widgets**: ipywidgets, matplotlib for progress
- **Validation**: Comprehensive error checking

### File Structure
```
project_root/
├── widgets/
│   ├── setup_widget.py
│   ├── dataset_widget.py
│   ├── training_widget.py
│   └── utils_widget.py
├── core/
│   ├── model_manager.py
│   ├── dataset_processor.py
│   └── training_runner.py
├── templates/
│   ├── configs/
│   └── presets/
└── Main_Notebook.ipynb
```

### Configuration System
- **TOML-based** like current system
- **Preset templates** for common scenarios
- **Widget state persistence** between runs
- **Export/import** capability for sharing configs

## User Experience Flow

### First-Time User Journey
1. **Setup Widget**: Choose model type → Auto-install → Download models
2. **Dataset Widget**: Upload images → Auto-tag → Review/edit
3. **Training Widget**: Pick preset → Adjust if needed → Start training
4. **Monitor**: Watch progress, adjust if needed
5. **Utils Widget**: Resize/optimize final LoRA

### Power User Features
- **Custom presets** for repeated workflows
- **Batch processing** for multiple LoRAs
- **Advanced scheduler options** from Derrian Distro
- **Integration** with external tools

## Implementation Phases

### Phase 1: Core Widgets
- Setup & Models widget (replaces cells 1-3)
- Basic training widget (simplified config)
- Get basic workflow working

### Phase 2: Enhanced Features
- Dataset management widget
- Advanced training options
- Progress monitoring

### Phase 3: Power Features
- Derrian Distro scheduler integration
- Batch processing
- Advanced utilities

### Phase 4: Polish
- Error handling refinement
- Documentation/tutorials
- Performance optimization

## Design Decisions & Rationale

### Why Jupyter over Colab?
- **Local control** - no Google dependency
- **GPU rental compatibility** - works with any provider
- **Persistent environment** - no re-setup each session
- **Custom widgets** - more flexibility than Colab forms

### Why Widget-Based?
- **Cognitive load reduction** - less scattered config
- **Visual clarity** - easier to scan and understand
- **Error prevention** - validation at input time
- **Consistency** - same patterns across all functions

### Why Target High-End Hardware?
- **Realistic for target users** - people training LoRAs seriously
- **Simplifies optimization** - don't need to support every edge case
- **Better UX** - can use larger batch sizes, faster training

## Success Metrics

### User Experience
- Can complete first LoRA training without reading external docs
- Errors are self-explanatory and actionable
- Widgets work consistently across sessions

### Technical
- Training results match or exceed existing tools
- Installation success rate >90% on target hardware
- Performance comparable to direct script usage

### Adoption
- Positive feedback from neurodivergent users
- Community contributions/preset sharing
- Used as reference by other projects

## Next Steps

1. **Finalize widget structure** - review this document
2. **Create Setup Widget** - start with environment + models
3. **Test basic workflow** - ensure foundation is solid
4. **Iterate based on usage** - real-world testing
5. **Add advanced features** - build up complexity gradually

---

*This document is a living design spec - update as we learn and iterate!*