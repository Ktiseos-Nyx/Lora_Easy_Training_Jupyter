# 🚀 Flux/SD3 LoRA Training (T5 Text Encoder)

Train LoRA adapters for the T5 text encoders in Flux and SD3 diffusion models with memory-optimized configurations.

## 🎯 Supported Models

- **Flux.1**: LoRA training for T5-XXL text encoder component
- **SD3/SD3.5**: LoRA training for T5-XXL text encoder component  
- **AuraFlow**: T5-XXL text encoder LoRA (community model)
- **Custom Flux variants**: Any Flux-based model with T5 text encoder

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   cd T5_Training
   pip install -r requirements.txt
   ```

2. **Prepare your dataset**:
   - Same format as standard LoRA training
   - Image files: `image1.jpg`, `image2.png`, etc.
   - Caption files: `image1.txt`, `image2.txt`, etc.
   - Can use existing Danbooru tags or BLIP captions

3. **Open the training notebook**:
   ```bash
   jupyter lab notebooks/T5_Trainer.ipynb
   ```

4. **Configure and train**:
   - Set your project name and dataset path
   - Choose model type and training preset
   - Let the system auto-detect optimal memory settings
   - Run the training cells

## 🧠 Memory Optimization

Automatic memory profiles based on your GPU:

| VRAM | Profile | Batch Size | Features |
|------|---------|------------|----------|
| 8GB | Ultra Low | 1 | AdaFactor, aggressive checkpointing |
| 12GB | Low | 2 | AdamW8bit, balanced settings |
| 16GB+ | Standard | 4 | AdamW, standard training |
| 24GB+ | High Performance | 8 | Maximum throughput |

## 📁 Project Structure

```
T5_Training/
├── notebooks/              # Training notebooks
│   └── T5_Trainer.ipynb   # Main T5 training interface
├── core/                   # Backend systems
│   ├── t5_training_manager.py
│   ├── t5_dataset_manager.py
│   └── ...
├── configs/               # Generated training configs
├── output/               # Trained models
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## 🎯 Training Presets

### Concept Learning
- **Use case**: Learn new objects, characters, concepts
- **Learning rate**: 5e-5
- **Epochs**: 3
- **Focus**: Encoder layers

### Style Adaptation  
- **Use case**: Adapt to artistic styles and aesthetics
- **Learning rate**: 1e-4
- **Epochs**: 5
- **Focus**: Full model fine-tuning

### Prompt Following
- **Use case**: Improve prompt understanding and adherence
- **Learning rate**: 3e-5
- **Epochs**: 2
- **Focus**: Decoder emphasis

## 🔧 Advanced Configuration

The system supports extensive customization:

```python
# In the notebook
CUSTOM_SETTINGS = {
    'learning_rate': 3e-5,      # Override preset learning rate
    'epochs': 5,                # Override preset epochs
    'max_samples': 1000,        # Limit dataset for testing
}
```

## 💡 Integration with Diffusion Models

Once trained, T5 encoders can be integrated with:

- **AuraFlow**: Direct T5-XXL replacement
- **HiDream**: Enhanced text understanding
- **Custom pipelines**: Use with your own diffusion implementations

## 🚨 Current Status: Work In Progress

**Completed**:
- ✅ Memory optimization profiles
- ✅ Dataset processing (reuses existing image+caption format)
- ✅ Configuration generation
- ✅ Training environment validation

**In Development**:
- 🔄 Training script integration
- 🔄 Model export and conversion
- 🔄 Inference testing and benchmarks

## 🤝 Contributing

This is part of the larger LoRA Easy Training ecosystem. The T5 training system is designed to be:

- **Modular**: Independent but integrated with main project
- **Memory-efficient**: Works on consumer GPUs
- **User-friendly**: Jupyter notebook interface
- **Extensible**: Easy to add new model architectures

## 📋 Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- 8GB+ VRAM recommended (4GB minimum with ultra-low-memory profile)

## 🎓 Learning Resources

- [T5 Paper](https://arxiv.org/abs/1910.10683) - Original T5 architecture
- [AuraFlow Documentation](https://github.com/XLabs-AI/auraflow) - AuraFlow integration
- [HuggingFace T5 Guide](https://huggingface.co/docs/transformers/model_doc/t5) - T5 usage guide

---

*Part of the LoRA Easy Training Jupyter ecosystem*  
*Focused on making advanced T5 training accessible to everyone*