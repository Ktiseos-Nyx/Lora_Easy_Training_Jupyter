# Post Tagging


##  <img src="assets/doro_fubuki.png" width="32" height="32"> Dataset Preparation Checklist

Before moving to training, ensure you have:

### ✅ Dataset Structure
- [ ] Images are in a single folder
- [ ] All images have corresponding .txt caption files
- [ ] No corrupted or unreadable images
- [ ] Consistent image format (jpg/png)

### ✅ Caption Quality
- [ ] All captions contain your trigger word
- [ ] Tags are accurate and relevant
- [ ] No unwanted or problematic tags
- [ ] Caption length is reasonable (50-200 tokens)

### ✅ Content Verification
- [ ] Images represent what you want to train
- [ ] Sufficient variety in poses/angles
- [ ] Consistent quality across dataset
- [ ] No duplicate or near-duplicate images

---

##  <img src="assets/OTNANGELDOROFIX.png" width="32" height="32"> Next Steps

Once your dataset is prepared:

1. **Note your dataset path** - you'll need it for training
2. **Remember your trigger word** - important for generation
3. **Open** `Lora_Trainer_Widget.ipynb` for training setup
4. **Run the Setup widget** first in the training notebook

---

## <img src="assets/OTNEARTHFIXDORO.png" width="32" height="32"> Troubleshooting

### Common Issues

**"No images found":**
- Check ZIP file structure (images should be in root or single folder)
- Verify image formats (jpg, png, webp supported)
- Ensure files aren't corrupted

**"Tagging failed":**
- Check internet connection for model downloads
- Verify sufficient disk space (2-3GB for tagger models)
- Try different tagger model

**"Captions too long/short":**
- Adjust tag threshold settings
- Use tag filtering to remove excess tags
- Consider manual editing for important images

**"Missing trigger words":**
- Use bulk edit to add trigger words
- Check trigger word injection settings
- Verify trigger word isn't being filtered out

---

*Ready to create amazing LoRAs? Let's go! 🚀*
