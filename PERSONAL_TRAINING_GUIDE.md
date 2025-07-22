# Personal LoRA Training Guide 🎯
*Your "Don't Mess Up the Big Kid Shoes" Cheat Sheet*

## 🚨 GOLDEN RULES (Don't Break These!)

### Dataset Size Sweet Spots
```
Character LoRAs: 15-50 images (sweet spot: 20-30)
Style LoRAs: 50-200 images (more consistency needed)  
Concept LoRAs: 20-100 images (depends on complexity)

TINY DATASETS (Stop Being a Chicken Edition):
10-15 images: TOTALLY DOABLE (your peers are right!)
5-10 images: Possible with careful settings
```

### The Magic Formula (Holostrawberry Approved™)
```
Target Steps: 250-1000 (aim for 500-750)
Formula: Images × Repeats × Epochs ÷ Batch Size = Total Steps

Example: 25 images × 10 repeats × 10 epochs ÷ 4 batch = 625 steps ✅
```

## 🎛️ PROVEN SETTINGS (Your Safety Net)

### Character LoRA (99% Success Rate)
```
Network: 8 dim / 4 alpha (the holy grail!)
UNet LR: 5e-4 (don't touch this!)
Text Encoder LR: 1e-4 (half of UNet, always!)
Scheduler: Cosine with 3 restarts
Optimizer: AdamW8bit (safe) or CAME (fancy but reliable)
Resolution: 1024x1024 (SDXL) or 768x768 (SD 1.5)
Batch Size: 4 (or 2 if VRAM crying)
```

### Style LoRA (When Character Settings Fail)
```
Network: LoCon (instead of regular LoRA)
Method: (IA)³ - Implicit Attention (new fancy stuff!)
UNet LR: 3e-4 (slightly lower)
Text Encoder LR: 5e-5 (much lower)
More epochs, lower learning rates
```

## 🧪 NEW TOYS TO PLAY WITH (Advanced Mode)

### Memory Savers (When VRAM is Sad)
- **CAME Optimizer**: Saves ~2GB VRAM, works great
- **Fused Back Pass**: VRAM optimization, needs batch size = 1
- **Gradient Checkpointing**: Trade speed for memory

### Quality Boosters (When You Want Excellence)
- **DoRA**: Higher quality, 2-3x slower, same VRAM
- **REX Scheduler**: Smart restarts, pairs with CAME
- **Prodigy Plus**: Learning rate free (experimental but cool)

### LyCORIS Methods (The Fancy Stuff)
```
DoRA: Best quality, slower training
LoKr: Memory efficient 
LoHa: Good balance
(IA)³: Perfect for styles
BOFT: Experimental butterfly magic
GLoRA: Generalized everything
```

## ⚠️ DANGER ZONES (Learn From My Mistakes)

### Don't Do These Things
- **Learning rates > 1e-3**: Recipe for disaster
- **Steps < 200 or > 2000**: Either undercook or burn
- **Batch size > 8**: VRAM go boom (unless you're rich)
- **Too many epochs with high LR**: Overfitting city
- **Mixing incompatible optimizers/schedulers**: The widgets will warn you!

### Red Flags During Training
- **Loss not decreasing after 100 steps**: Lower learning rate
- **Loss oscillating wildly**: Batch size too high or LR too high  
- **VRAM errors**: Reduce batch size, enable optimizations
- **Training super fast (<30 min)**: Probably too few steps

## 🎯 YOUR PERSONAL WORKFLOW

### Phase 1: Dataset (Don't Skip!)
1. **20-40 good quality images** (variety is key) - OR **10-15 if you're brave!**
2. **WD14 tagging** for anime/art, **BLIP** for photos
3. **Clean up tags** - remove weird stuff
4. **Add trigger word** to ALL captions
5. **Check consistency** - same character = same tags

### TINY DATASET SPECIAL RULES (10-15 Images)
- **Higher repeats**: 15-25 instead of 10
- **More epochs**: 15-25 instead of 10
- **Lower learning rates**: UNet 3e-4, TE 5e-5
- **Perfect quality images**: Every single one counts!
- **More careful tagging**: Can't afford bad captions

### Phase 2: Training Setup
1. **Start with proven settings** (character LoRA defaults)
2. **Calculate steps** using the formula
3. **Check VRAM** - enable optimizations if needed
4. **Pick your adventure**: Basic mode (safe) or Advanced (fun)

### Phase 3: Advanced Experiments (When Feeling Brave)
1. **Try CAME optimizer** (almost always better)
2. **Experiment with DoRA** if you have time
3. **Play with LyCORIS methods** for styles
4. **Use REX scheduler** with CAME

## 🐔 STOP BEING A CHICKEN! (Small Dataset Confidence Boost)

### Your Peers Are Right - 10 Images CAN Work!
```
Math Check:
10 images × 20 repeats × 15 epochs ÷ 2 batch = 1500 steps
(That's actually more steps than your 30-image sets!)

Why You Use 200 Images (Chicken Mode):
- "More data = safer" (not always true!)
- Fear of failure (but you're experienced!)
- Overthinking it (classic you!)

Why 10-15 Images Actually Works:
- Forces you to pick ONLY the best images
- Less noise in the training data
- Faster iteration and testing
- Your peers aren't lying to you!
```

### Small Dataset Success Formula
```
10-15 Perfect Images Strategy:
- Repeats: 15-25 (milk every image!)
- Epochs: 15-20 (more passes needed)
- UNet LR: 3e-4 (be gentler)
- TE LR: 5e-5 (way gentler)
- Batch Size: 1-2 (focus on quality)
- Target Steps: 800-1200 (longer training)
```

## 💡 TROUBLESHOOTING YOUR COMMON ISSUES

### "Training is Too Slow"
- Switch to CAME optimizer
- Enable Fused Back Pass
- Lower resolution if desperate

### "Results Look Bad"
- Check your captions (garbage in = garbage out)
- Try DoRA method
- Lower learning rate, train longer
- More variety in dataset

### "Out of VRAM" 
- Batch size = 1
- Enable all memory optimizations
- Use CAME optimizer
- Lower resolution

### "Training Finishes Too Fast"
- Increase epochs or repeats
- Target 500+ steps minimum
- Check your math on step calculation

## 🏆 SUCCESS METRICS

### Good Signs
- **Loss steadily decreasing** for first 200-300 steps
- **Training takes 1-3 hours** (not minutes!)
- **Total steps between 400-800**
- **No VRAM errors or crashes**

### Victory Conditions
- **LoRA generates your subject** with trigger word
- **Style is consistent** with training data
- **File size ~20-100MB** (reasonable dims)
- **Works at strength 0.7-1.0** without artifacts

---

## 🚀 QUICK START CHECKLIST

**Before You Start:**
- [ ] Dataset has 20+ good images
- [ ] All images tagged and have trigger word
- [ ] You know what you're training (character/style/concept)

**Safe First Run:**
- [ ] Use Character LoRA defaults (8 dim/4 alpha)
- [ ] UNet LR: 5e-4, TE LR: 1e-4
- [ ] Cosine scheduler, AdamW8bit optimizer
- [ ] Calculate steps (aim for 500-750)

**Feeling Fancy:**
- [ ] Enable advanced mode
- [ ] Try CAME optimizer
- [ ] Experiment with DoRA
- [ ] Use REX scheduler

---

*Remember: "Either gonna work or blow up!" - Start safe, then get fancy!* 🎉

**P.S.** The widgets have smart warnings and recommendations - TRUST THEM! They'll stop you from doing dumb stuff. 😄