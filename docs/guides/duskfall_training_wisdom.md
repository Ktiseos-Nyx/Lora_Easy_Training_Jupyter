# DuskFall's Opinionated LoRA Training Guide (2025 Edition)

*Based on DuskFall's Civitai article: [Opinionated Guide to All LoRA Training 2025 Update](https://civitai.com/articles/1716/opinionated-guide-to-all-lora-training-2025-update)*

---

## 🚨 Important Disclaimer

**This guide is NOT a bible!** This is DuskFall's neurodivergent approach to LoRA training based on personal experimentation and collaboration with the AI art community. These are opinions, observations, and findings that work for them - your mileage may vary!

*Training is as much art as science - embrace the experimentation!*

---

## 🎯 DuskFall's 2025 Training Philosophy

### Quality Over Quantity
- **Focus on curation**: Better to have fewer high-quality images than tons of mediocre ones
- **Resolution matters**: Aim for images above 512 pixels, ideally 1024px for SDXL
- **Remove duplicates**: Similar poses waste precious training steps
- **Synthetic data is valid**: Nijijourney and Midjourney outputs work great as training data

### The Art of Experimentation
DuskFall emphasizes that LoRA training is constantly evolving. What worked for SD 1.5 in 2023 may not apply to SDXL models in 2025. Stay flexible and test everything!

## 🎨 DuskFall's 2025 Model Recommendations

### Primary Models (Actually Used)
**Main Go-To Models:**
- **Pony XL**: DuskFall's frequent choice for versatile training
- **Illustrious**: High-quality artistic outputs
- **NoobAI XL**: Currently experimenting with this
- **Animagine SDXL**: Reliable for anime-style content

**Occasional Experiments:**
- **Flux models**: Testing newer architectures
- **SD 1.5 variants**: Still useful for specific cases

### Model Selection Strategy
**Don't blindly follow old guides!** Test your concept with the base model first - if it already knows your character/style somewhat, training will be easier.

## ⚙️ DuskFall's Current Parameter Setup

*These reflect DuskFall's actual 2025 experiments, not generic advice*

### Learning Rates (The Critical Setting)
```
Base Learning Rate: 5e4 (5e-4)
Text Encoder: Around 1e4 (1e-4)
```

**DuskFall's reasoning:**
- These rates work well with current SDXL models
- Text encoder should be lower to prevent prompt contamination
- Adjust based on your specific model and dataset

### Network Architecture
```
Network Dimension: 32
Network Alpha: 16-32 (experimenting with different ratios)
```

**Why 32 dim:**
- Higher than traditional 8-16 recommendations
- Captures more detail for complex styles
- File size trade-off worth it for quality

### Optimizer Experiments
**Current Favorites:**
- **Adafactor**: Memory efficient, stable results
- **AdamW8Bit**: Good balance of performance and VRAM usage
- **Prodigy**: Experimenting with learning-rate-free training

**Batch Size Strategy:**
- **2-4 for most training**: Cost optimization on Civitai
- Higher batch sizes when VRAM allows

### Training Schedule Insights
```
Clip Skip: 1-2 (depends on base model)
Steps: Above 3600 for style consistency
```

**DuskFall's Step Philosophy:**
Going above 3600 steps often improves style consistency, contrary to "quick and dirty" approaches.

## 📊 DuskFall's Dataset Approach

### Dataset Quality Focus
- **Check for duplicates**: Wastes training on repeated content
- **Resolution consistency**: Mixed sizes can confuse training
- **Content variety**: But not at the expense of quality
- **Source diversity**: Synthetic data (Midjourney/Nijijourney) is perfectly valid

### Training Time Considerations
DuskFall optimizes for cost-effectiveness on platforms like Civitai while maintaining quality. This means finding the sweet spot between training time and results.

## 🔬 DuskFall's Experimental Mindset

### Neurodivergent Approach to Training
DuskFall's approach embraces the chaotic nature of LoRA training. Rather than following rigid rules, they emphasize:

- **Constant experimentation**: What works today might not work tomorrow
- **Community collaboration**: Learning from peers in the AI art space
- **Adaptation over adherence**: Updating methods based on new findings
- **Personal documentation**: Tracking what works for specific use cases

### 2025 Reality Check
**Don't blindly follow 2023 guides!** The landscape has changed significantly:
- SDXL models behave differently than SD 1.5
- New optimizers and techniques are available
- Hardware and software have evolved
- Community knowledge has expanded

## 🎨 Practical Application in Jupyter System

### DuskFall's Workflow Adaptation
These settings translate to the Jupyter training system:

**Dataset Preparation:**
- Use the dataset widget for curation
- Focus on quality over quantity
- Leverage WD14 tagging for anime content
- Manual review of captions is crucial

**Training Configuration:**
- Set network dimension to 32 (higher than traditional)
- Use Adafactor or AdamW8Bit optimizers
- Target 3600+ steps for style consistency
- Batch size 2-4 for cost optimization

**Monitoring Approach:**
- Watch loss curves closely
- Generate test samples during training
- Iterate based on results, not rigid schedules

## 🎯 DuskFall's Training Philosophy

### Embrace the Experimental Nature
"Training is as much art as science" - DuskFall approaches each project as an experiment:

- **No universal solutions**: Each dataset and concept is unique
- **Iterative improvement**: Build on previous successes and failures
- **Community wisdom**: Learn from others but adapt to your needs
- **Cost-effectiveness**: Balance quality with practical constraints

### Quality Over Speed
Rather than rushing to quick results, DuskFall emphasizes:
- **Thorough dataset curation**
- **Proper parameter testing**
- **Adequate training time** (above 3600 steps when needed)
- **Post-training evaluation**

## 🚀 Key Takeaways for Your Training

### What Makes DuskFall's Approach Work:
1. **Neurodivergent perspective**: Embracing non-linear experimentation
2. **2025 model focus**: Adapting to current SDXL landscape
3. **Cost optimization**: Practical training on platforms like Civitai
4. **Community collaboration**: Learning from and sharing with others
5. **Quality focus**: Better datasets > perfect parameters

### Adaptation Strategy:
- **Start with DuskFall's parameters** as a baseline
- **Adapt based on your specific model and dataset**
- **Document your results** for future reference
- **Share findings** with the community
- **Stay flexible** as techniques evolve

## 🎉 Final Thoughts

This guide reflects DuskFall's personal journey through LoRA training in 2025. It's not a universal solution but rather one person's documented approach to navigating the evolving landscape of AI model training.

**Remember:**
- Your results may vary (and that's normal!)
- Experimentation is key to finding what works for you
- The community learns together through shared experiences
- Training techniques will continue to evolve

### The Real Wisdom
Success in LoRA training comes from:
1. **Quality dataset curation** (DuskFall's #1 priority)
2. **Methodical experimentation** with parameters
3. **Community learning** and knowledge sharing
4. **Adaptation** to new models and techniques
5. **Patience** with the inherently chaotic process

---

*"These are just tips I've picked up to manage my own training sessions. Your results may vary, and that's perfectly normal!" - DuskFall*

## 💸 Duskfall's 2025 GPU Rental Research
*Real money, real results - documented findings from RTX 5090 training sessions*

### CAME + DoRA Fighting Game Style Experiments

**Current Test Pipeline:**
- **Marvel vs Capcom** (73 images + CVS crossover data)
- **Capcom vs SNK** (62 images, pure style consistency)  
- **Marvel Rivals** (282 images, largest dataset test)

**Configuration Being Tested:**
```toml
# Network Setup
network_dim = 16
network_alpha = 8
network_module = "lycoris.kohya"
network_args = ["algo=lora", "use_dora=True", "conv_dim=16", "conv_alpha=8"]

# Optimizer (CAME)
learning_rate = 1e-4 (0.0001)
text_encoder_lr = 0.0 (style training focus)
optimizer_type = "LoraEasyCustomOptimizer.came.CAME"
optimizer_args = ["weight_decay=0.1"]
lr_scheduler = "cosine"

# Training Parameters
train_batch_size = 4 (RTX 5090 power usage)
max_train_epochs = 10
min_snr_gamma = 8.0
keep_tokens = 1
```

**Hypothesis Being Tested:**
- CAME's adaptive learning should handle 1410 steps (282 images × 2 repeats × 10 epochs ÷ 4 batch) without overfitting
- DoRA should provide better style expressiveness than standard LoRA at same dimensions
- Style training benefits from aggressive regularization (weight_decay=0.1)

**Learning Rate Discoveries:**
- **1e-4 with CAME**: Expected sweet spot for style training
- **5e-4 with CAME**: "Endsinger mode" - abstract nightmare fuel (avoid!)
- **6e-5**: Too conservative for 282-image style datasets

**Environment Optimizations Applied:**
- Custom tagger with robust error handling (no more dependency crashes)
- Environment-based fallback system (cloud/rental GPU optimizations)
- Fixed widget cache issues for smooth dataset management

**Cost Efficiency Notes:**
- RTX 5090 rental @ ~$0.50/hour
- Batch size 4 maximizes GPU utilization
- Training time: ~20-30 minutes per style LoRA

**Training Results & Discoveries:**

### CAME Dataset Size Threshold Discovery
**Confirmed through RTX 4090/5090 testing:**

**✅ CAME Success Cases:**
- **Tyler Kincade (14 images):** PERFECT character learning
  - 6e-5 LR, 8/4 dimensions, 336 steps
  - Result: Slightly overbaked but highly versatile
  - Style inheritance actually beneficial for comic characters
  - Users can dial down to 0.5 strength for style flexibility

- **Wuthering Waves (125 images):** Style learning with base model dependency
  - 1e-4 LR, DoRA 16/8, 630 steps, text encoder disabled
  - Result: Success varies by base model - gacha-focused models perform better
  - Not as thick as original versions but more focused due to curated dataset
  - Won't do characters perfectly (by design with TE disabled)
  - **Base model alignment crucial** - Chinese gacha styles work better on gacha-trained models

**⚠️ CAME Partial Success:**
- **Marvel vs Capcom (73 images):** Weak but functional style learning
  - 6e-5 LR, regular LoRA 16/8, 730 steps
  - Result: Generic anime-ish style, recognizable but not strong
  - Works better when prompted with specific MvC characters
  - Benefited from base model's existing knowledge of similar styles (Persona 5-adjacent)
  - **Learning:** Even in CAME's comfort zone, style training needs higher LR than characters

**❌ CAME Failure Case:**
- **Marvel Rivals (282 images):** Failed style learning
  - 1e-4 LR, regular LoRA 16/8, 1410 steps  
  - Result: Wimpy texture learning only, missed actual style

**✅ AdamW Success Case:**
- **Marvel Rivals (282 images):** B+ style learning - underbaked but functional
  - 5e-4 LR, regular LoRA 16/8, AdamW optimizer
  - Result: Actually learning the style, some artifacts but recognizable
  - Higher epochs showing improvement trend
  - **Base model bias:** SDXL defaults to Captain America/Superman for superhero prompts
  - **Negative prompting helps** reduce default superhero references
  - **Conclusion:** AdamW can handle novel styles that CAME cannot

### Key Threshold Insights:
- **<50 images:** CAME perfect for characters (6e-5 LR ideal)
- **50-100 images:** CAME weak for styles (needs 1e-4+ LR, benefits from base model knowledge)
- **100-150 images:** CAME viable for styles with DoRA assistance (1e-4 LR minimum)  
- **>200 images:** CAME too conservative, switch to AdamW + aggressive LR (5e-4+)

### Style vs Character Training with CAME:
- **Character LoRAs:** Conservative LR (6e-5) works perfectly
- **Style LoRAs:** Need higher LR (1e-4+) even in CAME's comfort zone
- **Style novelty matters:** Familiar styles (MvC) easier than unique styles (Marvel Rivals)
- **Base model knowledge:** Significant advantage for style learning efficiency
- **Base model alignment:** Critical factor - gacha styles work better on gacha-focused models

### Testing Methodology Notes:
**Professional evaluation standards applied:**
- **ADetailer used** for face/detail enhancement during generation
- **Hires fix applied** for upscaling and quality assessment  
- **Proper generation pipeline** - not basic/default settings
- **Quality over quantity approach** - realistic user conditions
- **Honest limitation reporting** - no overselling of results

### The "Overbaked but Controllable" Philosophy:
**Better slightly overbaked than underbaked** - discovered through Tyler results:
- Strong learning at 1.0 strength ensures concept is properly captured
- Users can dial down to 0.5-0.7 for style flexibility  
- Comic/anime characters SHOULD have stylistic inheritance
- Provides reliability over perfect calibration

### What NOT to Do (Expensive Lessons Learned)
- **Never use 5e-4 LR with CAME** - turns art into abstract void
- **Don't skip system dependencies** - missing onnx/xformers kills training
- **Avoid mixing different tagging methods** without cleanup
- **Don't trust widget defaults** - always verify keep_tokens settings

### Future Research Directions
- Compare DoRA vs LoHa/LoCon performance on same datasets
- Test CAME vs AdamW8bit on larger style datasets (500+ images)  
- Investigate optimal repeat strategies for different dataset sizes
- Document training time vs quality trade-offs on rental hardware

---

## 🔗 Credits and References

- **Original Article**: [Duskfall's Opinionated Guide to All LoRA Training 2025 Update](https://civitai.com/articles/1716/opinionated-guide-to-all-lora-training-2025-update)
- **Author**: Duskfall of Ktiseos-Nyx
- **Community Wisdom**: HoloStrawberry, JustTNP, and the broader LoRA training community
- **System Integration**: Adapted for LoRA Easy Training Jupyter notebook system
- **Research Funding**: Personal GPU rental expenses (the real MVP 💸)

*This guide builds on the collective wisdom of the LoRA training community. Experiment, share, and help others learn!*