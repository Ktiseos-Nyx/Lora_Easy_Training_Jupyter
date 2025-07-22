# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Goal

The primary objective is to refactor the existing Jupyter notebook (`Adapted_Easy_Training_Colab.ipynb`) into a user-friendly, widget-based interface called **"Instructable Widgets"**. The goal is to create a guided, educational, and powerful LoRA training experience within Jupyter, bridging the gap between complex command-line tools and rigid GUIs.

## User Profile & Communication

- **User**: Neurodivergent (DID system with Autism & ADHD), with a basic understanding of Python concepts but not a professional coder.
- **Communication Style**:
    - Stay focused and on-task.
    - Use clear, structured, and consistent patterns.
    - Provide step-by-step guidance.
    - Prioritize error prevention and clear feedback.

## Key Technologies

- **Frontend**: Jupyter Notebook, `ipywidgets`
- **Backend**: `sd-scripts`, `accelerate`, `torch`, `onnxruntime`
- **Configuration**: TOML files, managed via widgets.

## Proposed Architecture

The project will be refactored from a single notebook to a more modular structure, utilizing **two separate Jupyter notebooks**:

1.  **`Dataset_Maker_Widget.ipynb`**: This notebook will contain the `DatasetWidget` and all related functionality for dataset preparation (upload, tagging, caption management).
2.  **`Lora_Trainer_Widget.ipynb`**: This notebook will contain the `SetupWidget`, `TrainingWidget`, and `UtilitiesWidget` for environment setup, training configuration, and post-training utilities.

```
project_root/
├── widgets/          # Self-contained ipywidget components
│   ├── setup_widget.py
│   ├── dataset_widget.py
│   ├── training_widget.py
│   └── utilities_widget.py
├── core/             # Backend logic for widgets
│   └── managers.py   # Contains SetupManager, ModelManager, DatasetManager, TrainingManager, UtilitiesManager
├── templates/        # Config presets (future)
├── Dataset_Maker_Widget.ipynb # New notebook for dataset creation
└── Lora_Trainer_Widget.ipynb  # Renamed notebook for LoRA training
```

## Development Plan & Phases

The project will be developed iteratively.

- **Phase 1: Core Widgets (Completed)**:
    - Created `widgets/` and `core/` directories.
    - Implemented `SetupWidget` and `SetupManager`.
    - Implemented `DatasetWidget` and `DatasetManager`.
    - Implemented `TrainingWidget` and `TrainingManager`.
    - Implemented `UtilitiesWidget` and `UtilitiesManager`.

- **Refactoring to Two-Notebook Architecture (In Progress)**:
    1.  Rename `Main_Notebook.ipynb` to `Lora_Trainer_Widget.ipynb` (Completed).
    2.  Create `Dataset_Maker_Widget.ipynb`.
    3.  Move `DatasetWidget` code from `Lora_Trainer_Widget.ipynb` to `Dataset_Maker_Widget.ipynb`.

- **Subsequent Phases**:
    - **Phase 2**: Enhance with more advanced options and features in existing widgets.
    - **Phase 3**: Add power-user features like custom schedulers and batch processing.
    - **Phase 4**: Polish, refine error handling, and add documentation.

## Development Commands

- **Environment Setup**: `chmod +x ./jupyter.sh && ./jupyter.sh`
- **Run Jupyter**: `jupyter notebook Lora_Trainer_Widget.ipynb` or `jupyter notebook Dataset_Maker_Widget.ipynb`

## Guiding Principles

- **Single-cell workflows**: Each major step (setup, dataset, training) should be handled by a single widget in its own cell.
- **Progressive disclosure**: Keep the main interface simple, with advanced options hidden in collapsible sections.
- **Smart defaults**: Widgets should work "out-of-the-box" for common use cases.
- **Clear validation & feedback**: Provide helpful, context-aware error messages.
- **Neurodivergent-Friendly Design**: Maintain consistency, visual clarity, and step-by-step guidance.
- **Backend-driven**: Leverage existing, powerful backend scripts (e.g., `kohya-ss/sd-scripts`) as the core engine, with widgets acting as a configurable frontend.