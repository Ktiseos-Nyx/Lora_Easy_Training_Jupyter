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
- **Range**: Typically 4-128, commonly 8-32 for regular LoRA, 16-32 for LyCORIS
- **Trade-offs**: Higher = more detail capture, larger files, slower training
- **OneTrainer guidance**: Start with 8-16 for most use cases
- **Kohya_ss guidance**: Increase for complex styles or characters
- **LyCORIS consideration**: Higher capacity needed due to more stable training
- **Our system**: Adjustable in network settings (defaults: 16 for LyCORIS, 8 for regular LoRA)

**Network Alpha**
- **What it does**: Scaling factor for LoRA weights during training
- **Common values**: Often set to half of network dimension (dim/2 ratio)
- **Effect**: Higher alpha = stronger LoRA influence during training
- **Mathematical relationship**: alpha/rank ratio affects learning dynamics and final strength
- **LyCORIS behavior**: Can handle higher alpha values due to improved stability
- **Capacity vs Stability**: Lower alpha/dim ratios (like 0.25-0.5) provide more stable training
- **Our system**: Paired with network dimension setting (defaults: 8 for LyCORIS, 4 for regular LoRA)

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

## 🧠 **Understanding Optimizers vs Schedulers**

**OPTIMIZERS** control **HOW** the model learns from gradients:
- **What they do**: Update model weights based on calculated gradients
- **Key function**: Decide how much to change weights each step
- **Think of it as**: The "learning strategy" - cautious vs aggressive weight updates
- **Examples**: AdamW, CAME, Adafactor, SGD

**SCHEDULERS** control **WHEN** and **HOW MUCH** learning happens:
- **What they do**: Adjust the learning rate over time during training
- **Key function**: Start high/low, increase/decrease, restart, etc.
- **Think of it as**: The "learning timeline" - when to learn fast vs slow
- **Examples**: Cosine, Linear, Constant, Cosine with Restarts

**The Relationship:**
```
Learning Rate Schedule (when/how much) + Optimizer (how) = Training Strategy
    ↓                                        ↓                    ↓
Cosine (start high, decay smooth)    +    AdamW (adaptive)  = Stable training
Constant (same rate always)         +    CAME (efficient)  = Fast convergence  
```

### Optimizer Selection

**AdamW (Standard)**
- **How it works**: Adaptive learning with momentum and weight decay
- **Characteristics**: Reliable, well-tested optimizer
- **Memory usage**: Higher memory requirements (~2-3GB extra)
- **Learning rate**: Works well with standard learning rates (1e-4 to 1e-6)
- **Best for**: General purpose, proven stability
- **Scheduler pairs**: Works with all schedulers
- **Our system**: Primary optimizer option

**AdamW8Bit** ⚠️ 
- **How it works**: Quantized version of AdamW using 8-bit precision
- **Purpose**: Memory-optimized version of AdamW
- **Memory savings**: Significant reduction in VRAM usage (~1-2GB saved)
- **Performance**: Minimal quality difference from full AdamW
- **CRITICAL ISSUE**: **Requires Triton compiler** - often broken in Docker/VastAI containers
- **Use case**: When memory is constrained AND Triton works
- **VastAI/Docker**: **Likely to fail** - see VastAI section below

**CAME** ✅
- **How it works**: Confidence-guided Adaptive Memory Efficient optimizer
- **Innovation**: Uses confidence measures to guide updates more intelligently
- **Memory efficiency**: Very low memory overhead (~500MB vs AdamW's 2-3GB)
- **Learning rate**: Often works with higher learning rates than AdamW
- **Stability**: Excellent for difficult training scenarios (v-pred models)
- **Special feature**: **Uses Huber loss automatically** for better robustness
- **Best for**: Memory-constrained systems, unstable models (NoobAI-XL, v-pred)
- **Our system**: Advanced optimizer with automatic Huber loss + REX scheduling

**Adafactor** ✅
- **How it works**: Adaptive learning with factorized second moments (memory efficient)
- **Innovation**: Reduces memory by factorizing the optimizer state
- **Memory efficiency**: Very low memory usage, scales well with model size  
- **Learning rate**: Can work with higher rates, has adaptive scaling options
- **Stability**: More stable than AdamW for large models
- **Best for**: Large models, memory-constrained training, SDXL fine-tuning
- **Scheduler pairs**: Works best with constant or constant_with_warmup
- **Our system**: Available as standard optimizer

**Prodigy/Prodigy Plus**
- **How it works**: Learning rate-free optimization - automatically finds optimal rates
- **Unique feature**: No manual learning rate tuning required
- **Innovation**: Estimates optimal learning rate based on gradient statistics
- **Experimental**: Newer, less tested but very promising results
- **Memory**: Similar to AdamW but with automatic rate adjustment
- **Learning rate**: Set to 1.0 and let Prodigy handle the rest
- **Best for**: When you don't want to tune learning rates manually
- **Our system**: Available for experimentation with schedule-free option

### Learning Rate Schedulers Explained

**Constant**
- **How it works**: Learning rate stays the same throughout training
- **When to use**: Simple training, when you found a good learning rate
- **Pros**: Predictable, easy to understand
- **Cons**: May not be optimal for long training runs
- **Best with**: Adafactor, Prodigy (which handles rate internally)

**Linear**
- **How it works**: Learning rate decreases linearly from start to end
- **When to use**: When you want gradual slowdown
- **Pros**: Smooth transition, predictable
- **Cons**: May decrease too aggressively early on
- **Best with**: AdamW for simple decay

**Cosine**
- **How it works**: Learning rate follows smooth cosine curve (high→low)
- **When to use**: Most LoRA training - excellent default choice
- **Pros**: Smooth decay, good for convergence
- **Cons**: Single decay cycle only
- **Best with**: AdamW, CAME - most popular combination

**Cosine with Restarts**
- **How it works**: Cosine decay but restarts to high learning rate periodically
- **When to use**: Long training, escaping local minima
- **Pros**: Can find better solutions through restarts
- **Cons**: More complex, needs tuning of restart frequency
- **Best with**: AdamW, good for difficult datasets

**Constant with Warmup**
- **How it works**: Starts low, warms up to target rate, then stays constant
- **When to use**: Large models, Adafactor training
- **Pros**: Stable start, prevents early instability
- **Cons**: May not decay when needed
- **Best with**: Adafactor (often recommended), large model training

### Advanced Schedulers

**REX (Exponential Annealing)**
- **How it works**: Exponential learning rate decay with warm restarts
- **Innovation**: More sophisticated than cosine restarts
- **Benefits**: Can escape local minima better than standard schedulers
- **Complexity**: More parameters to configure (gamma, d values)
- **Research basis**: Based on recent optimization research
- **Best with**: CAME optimizer (our default pairing)
- **Our system**: Available in advanced scheduler options with CAME

**Schedule-Free**
- **How it works**: Optimizer handles scheduling internally - no external scheduler
- **Philosophy**: Automatic schedule adjustment based on gradient statistics
- **Implementation**: No manual schedule configuration needed
- **Innovation**: Cutting-edge research implementation
- **Use case**: When you don't want to tune schedules manually
- **Best with**: Prodigy Plus (our implementation)
- **Our system**: Available with Prodigy Plus optimizer

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

**Network Dimension × Alpha - The Capacity vs Stability Trade-off**

**Understanding the Relationship:**
- **Alpha/Dim ratio**: Controls learning strength vs stability balance
- **0.25 ratio (dim=16, alpha=4)**: Very stable, may underfit with small datasets
- **0.5 ratio (dim=16, alpha=8)**: Balanced approach, our new default for LyCORIS
- **1.0 ratio (dim=16, alpha=16)**: Aggressive learning, higher burning risk

**Architecture-Specific Behavior:**
- **Regular LoRA**: More sensitive to high alpha/dim ratios, burns easily
- **LyCORIS methods**: More stable, can handle higher ratios and dimensions
- **DoRA**: Even more stable, can push ratios higher without burning

**Dataset Size Considerations:**
- **Small datasets (10-30 images)**: Higher dim/alpha needed for sufficient learning
- **Medium datasets (50-200 images)**: Standard ratios work well
- **Large datasets (500+ images)**: Lower ratios prevent overfitting

**Practical Guidelines from Analysis:**
- **Regular LoRA**: 8/4 (0.5 ratio) - proven stable
- **LyCORIS**: 16/8 (0.5 ratio) - higher capacity, same stability
- **Experimental**: Try 32/16 for complex characters with good datasets
- **Conservative**: Drop to 16/4 (0.25 ratio) if experiencing burning

**Optimizer × Learning Rate**
- **AdamW**: Standard learning rate ranges
- **CAME**: Often tolerates higher rates
- **Prodigy**: Learning rate free operation
- **Adaptation**: Each optimizer has different optimal ranges

## 🐳 **VastAI / Docker Container Specific Guide**

### **The Triton Problem**

**If you're on VastAI or Docker containers, you're likely stuck in "Triton Hell":**

**Symptoms:**
```
TRITON NOT FOUND
RuntimeError: triton not compatible  
bitsandbytes compilation failed
AdamW8bit requires Triton
```

**Root Cause:**
- **VastAI containers** have pre-built Python environments
- **Triton compiler** needs specific CUDA toolkit versions
- **Docker layers** make Triton installation extremely difficult
- **Our auto-fixes try** but container restrictions often prevent success

### **VastAI Recommended Optimizers (Triton-Free)**

**🥇 CAME (Best Choice)**
- **Why**: No Triton dependency, memory efficient, includes Huber loss
- **Settings**: Use default with REX scheduler
- **Learning Rate**: Can use slightly higher rates (1e-4 to 3e-4)
- **Best for**: All training types, especially v-pred models

**🥈 Adafactor (SDXL Champion)**  
- **Why**: No Triton dependency, excellent for large models
- **Settings**: Use with constant_with_warmup scheduler
- **Learning Rate**: Can use higher rates, has adaptive scaling
- **Best for**: SDXL fine-tuning, large models, memory-constrained training

**🥉 Regular AdamW (Reliable Fallback)**
- **Why**: Always works, well-tested, no dependencies
- **Settings**: Use with cosine or cosine_with_restarts
- **Learning Rate**: Standard rates (1e-4 to 1e-6)
- **Best for**: When you want proven stability

**❌ Avoid on VastAI:**
- **AdamW8bit** - Requires Triton (will likely fail)
- **Lion** - May have compilation issues
- **Any optimizer** labeled as requiring Triton compilation

### **VastAI Training Strategy**

**Recommended Combination:**
```
Optimizer: CAME
Scheduler: REX (included with CAME)
Loss: Huber (automatic with CAME)
Learning Rate: 1e-4 to 3e-4
```

**Alternative for SDXL:**
```
Optimizer: Adafactor  
Scheduler: constant_with_warmup
Learning Rate: 4e-7 (SDXL original rate) or higher
Warmup Steps: 100-200
```

**Memory Optimization for VastAI:**
- **Cache latents**: Always enable
- **Cache text encoder outputs**: For memory savings
- **Gradient checkpointing**: Essential for large models
- **fp16 precision**: Saves memory without Triton

### Quality vs Efficiency Trade-offs

**Training Speed Optimizations (VastAI-Safe)**
- Lower resolution training
- Smaller batch sizes  
- fp16 mixed precision
- Gradient checkpointing (slower but less memory)
- Efficient optimizers (CAME, Adafactor) - **NO AdamW8bit**

**Quality Maximization (VastAI-Safe)**
- Higher resolution training
- Larger network dimensions
- DoRA network type (if memory allows)
- CAME optimizer with Huber loss
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

## 📋 **Quick Reference: Optimizer + Scheduler Combinations**

### **Recommended Pairings**

**For Beginners (Most Reliable):**
```
AdamW + Cosine
Learning Rate: 1e-4 to 1e-6
```

**For VastAI/Docker (Triton-Free):**  
```
CAME + REX (automatic)
Learning Rate: 1e-4 to 3e-4
```

**For SDXL Training:**
```
Adafactor + constant_with_warmup  
Learning Rate: 4e-7 to 1e-6
Warmup: 100-200 steps
```

**For Memory-Constrained:**
```
CAME + REX (uses ~500MB vs AdamW's 2-3GB)
Enable: cache_latents, gradient_checkpointing, fp16
```

**For Experimentation:**
```
Prodigy Plus + schedule_free (automatic)
Learning Rate: 1.0 (let Prodigy handle it)
```

### **Troubleshooting Quick Fix**

**If AdamW8bit fails:** → Switch to CAME
**If training is unstable:** → Lower learning rate or try CAME with Huber loss  
**If out of memory:** → CAME + gradient checkpointing + fp16
**If on VastAI/Docker:** → Avoid anything requiring Triton, use CAME or Adafactor

---

*Remember: Understanding parameters conceptually helps you make better training decisions regardless of which specific interface you're using. The goal is training better LoRAs through informed parameter choices!*

---

*"The best parameter settings are the ones that work for your specific use case and dataset. Use this knowledge as a foundation for your own experimentation." - Community Wisdom*