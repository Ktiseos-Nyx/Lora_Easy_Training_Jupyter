# Training Parameter Deep Dive

*Comprehensive parameter reference based on community knowledge from OneTrainer and Kohya_ss*

---

This guide provides in-depth explanations of LoRA training parameters, drawing from the excellent documentation by the OneTrainer and Kohya_ss communities. **Note**: Not all parameters mentioned are available in our Jupyter system - this serves as educational reference for understanding what each setting does.

## 📚 **Reference Sources**

This guide synthesizes knowledge from:
- **[OneTrainer Wiki - Training](https://github.com/Nerogar/OneTrainer/wiki/Training)** - Comprehensive training theory and parameter explanations
- **[Kohya_ss Wiki - LoRA Training Parameters](https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters)** - Detailed parameter documentation

**Important**: These are external projects with their own installation and setup procedures. We reference their parameter knowledge but use our own Jupyter-based system for actual training.

## ⚠️ **Compatibility Notice**

**Parameter Availability**: Our Jupyter system implements a subset of parameters from these comprehensive tools. Some advanced or experimental settings may not be available. Always check the actual widget interface for current parameter availability.

**Focus**: This guide explains what parameters do conceptually, helping you understand the training process regardless of which specific parameters are exposed in any given interface.

## 🎯 **Core Training Parameters**

### Learning Rate Parameters

**UNet Learning Rate**
- **What it does**: Controls how quickly the image generation network learns
- **Typical range**: 1e-5 to 1e-3
- **OneTrainer insight**: Higher rates learn faster but risk instability
- **Kohya_ss insight**: Often the most critical parameter to tune
- **Our system**: Available in basic and advanced modes

**Text Encoder Learning Rate**
- **What it does**: Controls how the text understanding component learns
- **Typical range**: 1e-6 to 1e-4 (usually 10x lower than UNet)
- **Why lower**: Text encoder is more sensitive to changes
- **Our system**: Available when training text encoder is enabled

**Learning Rate Scheduler**
- **Constant**: Learning rate stays the same throughout training
- **Linear**: Gradual decrease over time
- **Cosine**: Smooth decrease with cosine curve
- **Cosine with Restarts**: Periodic learning rate resets
- **OneTrainer note**: Cosine often provides best results for LoRA
- **Our system**: Multiple scheduler options available

### Network Architecture

**Network Dimension (Rank)**
- **What it controls**: The "size" or capacity of the LoRA adaptation
- **Range**: Typically 4-128, commonly 8-32
- **Trade-offs**: Higher = more detail capture, larger files, slower training
- **OneTrainer guidance**: Start with 8-16 for most use cases
- **Kohya_ss guidance**: Increase for complex styles or characters
- **Our system**: Adjustable in network settings

**Network Alpha**
- **What it does**: Scaling factor for LoRA weights
- **Common values**: Often set to half of network dimension
- **Effect**: Higher alpha = stronger LoRA influence
- **Mathematical relationship**: alpha/rank ratio affects learning dynamics
- **Our system**: Paired with network dimension setting

**Dropout**
- **Purpose**: Regularization technique to prevent overfitting
- **Range**: 0.0 (no dropout) to 0.5 (aggressive)
- **OneTrainer use**: Helpful for larger datasets
- **When to use**: Large datasets or signs of overfitting
- **Our system**: Available in advanced options

### Training Schedule

**Max Epochs**
- **Definition**: How many times to go through entire dataset
- **Balancing act**: Too few = undertrained, too many = overtrained
- **Dataset relationship**: Larger datasets need fewer epochs
- **Monitoring**: Watch loss curves rather than targeting specific numbers
- **Our system**: Configurable with calculator widget preview

**Batch Size**
- **What it affects**: How many images processed simultaneously
- **Memory impact**: Larger batches use more VRAM
- **Training impact**: Larger batches = more stable gradients
- **Kohya_ss note**: Batch size affects learning rate effectiveness
- **Hardware constraint**: Limited by available VRAM
- **Our system**: Adjustable based on hardware capabilities

**Gradient Accumulation Steps**
- **Purpose**: Simulate larger batch sizes with limited memory
- **How it works**: Accumulates gradients over multiple mini-batches
- **Memory benefit**: Achieve large effective batch size with small VRAM
- **OneTrainer implementation**: Sophisticated gradient handling
- **Our system**: Available for memory optimization

## 🧠 **Advanced Optimization**

### Optimizer Selection

**AdamW (Standard)**
- **Characteristics**: Reliable, well-tested optimizer
- **Memory usage**: Higher memory requirements
- **Learning rate**: Works well with standard learning rates
- **Kohya_ss default**: Traditional choice for stability
- **Our system**: Primary optimizer option

**AdamW8Bit**
- **Purpose**: Memory-optimized version of AdamW
- **Memory savings**: Significant reduction in VRAM usage
- **Performance**: Minimal quality difference from full AdamW
- **Use case**: When memory is constrained
- **Our system**: Available for memory optimization

**CAME**
- **Innovation**: Confidence-guided adaptive optimizer
- **Memory efficiency**: Very low memory overhead
- **Learning rate**: Often works with higher learning rates
- **Stability**: Good for difficult training scenarios
- **Our system**: Available as advanced optimizer

**Prodigy**
- **Unique feature**: Learning rate free optimization
- **How it works**: Automatically adjusts learning rate during training
- **Experimental**: Newer, less tested but promising
- **OneTrainer experience**: Good results when it works
- **Our system**: Available for experimentation

### Advanced Schedulers

**REX (Exponential Annealing)**
- **Concept**: Exponential learning rate decay with restarts
- **Benefits**: Can escape local minima
- **Complexity**: More parameters to configure
- **Research basis**: Based on recent optimization research
- **Our system**: Available in advanced scheduler options

**Schedule-Free**
- **Philosophy**: Automatic schedule adjustment
- **Implementation**: No manual schedule configuration needed
- **Experimental**: Cutting-edge research implementation
- **Use case**: When you don't want to tune schedules manually

### Precision and Memory

**Mixed Precision Training**
- **fp16**: Half precision, saves memory, slightly less accurate
- **bf16**: Brain float 16, good balance of speed and accuracy
- **fp32**: Full precision, highest accuracy, most memory
- **Kohya_ss recommendation**: fp16 for most use cases
- **Hardware dependency**: Requires compatible GPU
- **Our system**: Selectable precision options

**Gradient Checkpointing**
- **Trade-off**: Memory for computation time
- **How it works**: Recomputes activations instead of storing them
- **Memory savings**: Significant VRAM reduction
- **Speed cost**: 20-30% slower training
- **OneTrainer insight**: Essential for large models on limited hardware
- **Our system**: Available for memory optimization

## 🎨 **Quality and Stability Settings**

### Noise and Conditioning

**Noise Offset**
- **Purpose**: Helps model learn darker and brighter images
- **Typical range**: 0.0 to 0.1
- **Effect**: Expands contrast range in generated images
- **OneTrainer guidance**: Often beneficial for art styles
- **Visual impact**: Better handling of lighting extremes
- **Our system**: Available in advanced options

**Min SNR Gamma**
- **Technical purpose**: Stabilizes training with noise scheduling
- **Typical value**: 5.0
- **Research basis**: Based on recent diffusion model research
- **Stability benefit**: Reduces training instability
- **Kohya_ss implementation**: Proven to improve training consistency
- **Our system**: Available with recommended default

**Clip Skip**
- **What it affects**: Which CLIP text encoder layer to use
- **Common values**: 1 (default) or 2
- **Model dependency**: Some models trained with clip skip 2
- **Effect**: Changes how text prompts are interpreted
- **Compatibility**: Should match base model's training
- **Our system**: Configurable based on base model

### Validation and Monitoring

**V-Prediction**
- **Purpose**: Parameterization method for noise prediction
- **When to use**: Models specifically trained with v-prediction
- **Visual effect**: Prevents washed-out colors in certain models
- **Compatibility**: Must match base model's training method
- **Detection**: Check model documentation or test outputs
- **Our system**: Checkbox for v-prediction models

**Sample Generation**
- **Purpose**: Generate test images during training
- **Monitoring benefit**: Visual progress tracking
- **Frequency**: Every few epochs to avoid slowing training
- **Prompt consistency**: Use same prompts to track progress
- **OneTrainer feature**: Sophisticated sampling options
- **Our system**: Configurable sample generation

## 📊 **Dataset and Bucketing**

### Resolution and Bucketing

**Training Resolution**
- **Standard sizes**: 512x512 (SD1.5), 1024x1024 (SDXL)
- **Memory impact**: Higher resolution = more VRAM needed
- **Quality trade-off**: Higher resolution can capture more detail
- **Compatibility**: Should match base model's training resolution
- **Our system**: Selectable resolution options

**Bucket Resolution**
- **Purpose**: Handle various aspect ratios efficiently
- **How it works**: Groups similar aspect ratios together
- **Benefit**: Less image distortion from resizing
- **Kohya_ss innovation**: Advanced bucketing algorithms
- **Memory consideration**: More buckets = more memory usage
- **Our system**: Automatic bucketing implementation

**Bucket No Upscale**
- **Purpose**: Prevents upscaling smaller images
- **Quality benefit**: Maintains original image quality
- **Use case**: When dataset has mixed resolutions
- **Alternative**: Manual preprocessing to consistent size
- **Our system**: Available as option

### Caption and Token Handling

**Shuffle Caption**
- **Purpose**: Randomizes tag order in captions
- **Benefit**: Prevents position bias in tag learning
- **Implementation**: Shuffles tags randomly each epoch
- **Effect**: More robust tag understanding
- **OneTrainer approach**: Sophisticated caption handling
- **Our system**: Available as training option

**Keep Tokens**
- **Purpose**: Keeps certain tokens at beginning of captions
- **Use case**: Preserving trigger words or important descriptors
- **Number**: How many tokens to keep fixed
- **Balance**: Fixed important terms, shuffle the rest
- **Our system**: Configurable token preservation

## 🔬 **Experimental and Research Features**

### Advanced Network Types

**DoRA (Weight-Decomposed LoRA)**
- **Innovation**: Decomposes weight updates more effectively
- **Quality benefit**: Often superior results to standard LoRA
- **Cost**: Slower training and inference
- **Research**: Based on recent LoRA improvement research
- **OneTrainer support**: Implemented with full feature set
- **Our system**: Available as network type option

**LyCORIS Methods (LoKr, LoHa, etc.)**
- **LoKr**: Kronecker product decomposition
- **LoHa**: Hadamard product adaptation
- **(IA)³**: Implicit attention adaptation
- **Research basis**: Various approaches to efficient adaptation
- **Use cases**: Different methods excel in different scenarios
- **Experimental**: Less standardized than regular LoRA
- **Our system**: Selection of LyCORIS methods available

### Cutting-Edge Optimizers

**Lion**
- **Research**: Google Research optimizer
- **Characteristics**: Very different from Adam family
- **Memory**: Lower memory than AdamW
- **Learning rate**: Requires different learning rate scaling
- **Experimental**: Promising but requires careful tuning

**ADOPT**
- **Innovation**: Adaptive gradient clipping
- **Purpose**: Handles gradient explosion more elegantly
- **Research status**: Recent optimization research
- **Use case**: When training is unstable with standard optimizers

## 🎯 **Practical Application Guidelines**

### Parameter Interaction Effects

**Learning Rate × Batch Size**
- **Relationship**: Larger batches often need higher learning rates
- **Scaling**: Some research suggests linear scaling
- **Practical**: Test both parameters together
- **OneTrainer guidance**: Sophisticated parameter relationship handling

**Network Dimension × Alpha**
- **Common ratios**: alpha = dim/2 is traditional
- **Experimentation**: Some prefer alpha = dim or alpha = 1
- **Effect**: Different ratios change learning dynamics
- **Research**: Ongoing investigation into optimal ratios

**Optimizer × Learning Rate**
- **AdamW**: Standard learning rate ranges
- **CAME**: Often tolerates higher rates
- **Prodigy**: Learning rate free operation
- **Adaptation**: Each optimizer has different optimal ranges

### Quality vs Efficiency Trade-offs

**Training Speed Optimizations**
- Lower resolution training
- Smaller batch sizes
- fp16 mixed precision
- Gradient checkpointing (slower but less memory)
- Efficient optimizers (CAME, AdamW8Bit)

**Quality Maximization**
- Higher resolution training
- Larger network dimensions
- Full precision training
- DoRA network type
- Careful learning rate tuning
- Multiple training runs with different seeds

### Debugging and Troubleshooting

**Training Instability**
- **Symptoms**: Loss going to NaN, wild fluctuations
- **Solutions**: Lower learning rate, check data quality, try different optimizer
- **OneTrainer tools**: Advanced monitoring and debugging features
- **Kohya_ss approach**: Conservative defaults with proven stability

**Overfitting Detection**
- **Monitoring**: Compare training vs validation loss
- **Visual**: Sample generation shows repetitive outputs
- **Solutions**: Lower learning rate, fewer epochs, more regularization
- **Prevention**: Proper dataset curation and validation sets

**Memory Issues**
- **CUDA OOM**: Reduce batch size, enable gradient checkpointing, lower resolution
- **Slow training**: Check for memory thrashing, optimize batch size
- **Monitoring**: Track GPU memory usage throughout training

## 🔗 **Community Resources and Further Learning**

### Essential Reading
- **[OneTrainer Wiki](https://github.com/Nerogar/OneTrainer/wiki)**: Comprehensive training documentation
- **[Kohya_ss Documentation](https://github.com/bmaltais/kohya_ss/wiki)**: Parameter references and guides
- **Research Papers**: Recent diffusion model and LoRA research
- **Community Forums**: Civitai, Reddit discussions on parameter tuning

### Experimentation Framework
1. **Start Conservative**: Use proven parameter combinations
2. **Change One Thing**: Isolate parameter effects
3. **Document Results**: Track what works for your use cases
4. **Share Knowledge**: Contribute findings to community
5. **Stay Updated**: Parameter understanding evolves with research

---

## 🚨 **Important Disclaimers**

### Parameter Availability
- **System Specific**: Not all parameters discussed are available in every system
- **Implementation Differences**: Same parameter names may have different implementations
- **Version Changes**: Parameter availability and behavior evolves over time
- **Hardware Constraints**: Some parameters require specific hardware capabilities

### Research Evolution
- **Cutting Edge**: Some parameters represent recent research
- **Stability**: Newer parameters may be less stable or well-tested
- **Community Testing**: Parameter effectiveness often depends on community validation
- **Best Practices**: Optimal parameter combinations continue to evolve

### Credit and Attribution
This guide synthesizes knowledge from:
- **OneTrainer Team**: Advanced training system implementation
- **Kohya_ss Community**: Comprehensive parameter documentation
- **Research Community**: Academic papers and innovations
- **LoRA Training Community**: Practical experience and validation

*Remember: Understanding parameters conceptually helps you make better training decisions regardless of which specific interface you're using. The goal is training better LoRAs through informed parameter choices!*

---

*"The best parameter settings are the ones that work for your specific use case and dataset. Use this knowledge as a foundation for your own experimentation." - Community Wisdom*