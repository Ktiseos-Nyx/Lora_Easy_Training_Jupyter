# Contributing to LoRA Easy Training 🚀

**Welcome, brave soul!** Thanks for wanting to make LoRA training less painful for everyone. Whether you're here to fix bugs, add features, or just make things prettier, we're excited to have you aboard! 

## 🎯 Ways to Contribute

### 🐛 **Bug Reports** 
Found something broken? Don't suffer in silence!

- **Search existing issues first** (maybe someone beat you to it)
- **Use our templates** or just tell us what went wrong
- **Include your setup**: GPU, OS, Python version, what you were trying to train
- **Steps to reproduce**: Help us break things the same way you did
- **Screenshots/logs**: A picture (or error traceback) is worth a thousand "it doesn't work"

### ✨ **Feature Requests**
Got ideas? We love ideas!

- **Check if it exists** in issues/discussions first
- **Explain the "why"**: What problem does this solve?
- **Be specific**: "Better UI" vs "Add dark mode toggle in settings"
- **Consider scope**: Small improvements > major rewrites

### 🔧 **Code Contributions**
Ready to get your hands dirty? Awesome!

#### **What We Need Help With:**
- 🎨 **Widget improvements**: Make the UI more intuitive
- 🧮 **Calculator enhancements**: Better step estimation, learning rate suggestions
- 📊 **Dataset tools**: Upload improvements, better tagging workflows
- 🚀 **Training optimizations**: Memory usage, speed improvements
- 📝 **Documentation**: Code comments, user guides, examples
- 🐧 **Platform support**: Linux testing, AMD GPU compatibility
- 🧪 **Advanced features**: New optimizers, schedulers, LoRA types

#### **Getting Started:**
1. **Fork the repo** and clone it locally
2. **Create a branch**: `git checkout -b feature/amazing-improvement`
3. **Set up your environment**: `python installer.py` 
4. **Make your changes**: Follow our code style (it's pretty relaxed)
5. **Test your stuff**: Make sure it doesn't break existing features
6. **Commit with style**: Clear messages like "Add bulk dataset validation"
7. **Push and PR**: We'll review faster than you can say "LoRA"

## 🎨 Code Style & Standards

### **Python Guidelines:**
- **Follow PEP 8-ish**: We're not nazis about it, but readable code is happy code
- **Type hints welcome**: `def train_lora(epochs: int) -> bool:`
- **Docstrings for public functions**: Help future contributors (including future you)
- **Error handling**: Fail gracefully with helpful messages
- **No hardcoded paths**: Use `os.path.join()` and respect different platforms

### **Widget Development:**
- **Keep it simple**: Complex UIs confuse users (and maintainers)
- **Consistent styling**: Match existing widgets' look and feel
- **Clear labels**: "Network Dim" not "nd", "Learning Rate" not "lr"
- **Status feedback**: Show users what's happening with progress indicators
- **Error messages**: Helpful, not scary technical jargon

### **Training Code:**
- **Safety first**: Validate inputs before passing to training scripts
- **Memory awareness**: Not everyone has a 4090
- **Cross-platform**: Windows, Linux, macOS should all work
- **Backwards compatibility**: Don't break existing configs without good reason

## 🧪 Testing

### **Before You Submit:**
- **Test the happy path**: Does your feature work as intended?
- **Test edge cases**: What happens with 0 images? Invalid learning rates?
- **Test on different setups**: Different GPUs, Python versions if possible
- **Check existing functionality**: Did you break anything?

### **Manual Testing Basics:**
- Install fresh using `installer.py`
- Try the widget workflows: setup → dataset → training
- Test with small datasets (faster iteration)
- Check error handling with invalid inputs

## 🗣️ Communication

### **Issue Discussions:**
- **Be respectful**: We're all here to learn and improve
- **Stay on topic**: Keep discussions focused on the issue at hand
- **Share context**: What's your use case? Training anime characters? Realistic portraits?
- **Be patient**: We're volunteers with day jobs and lives

### **Pull Request Reviews:**
- **Explain your changes**: What does this PR do and why?
- **Link related issues**: `Fixes #123` or `Related to #456`
- **Be open to feedback**: Code review makes everyone better
- **Update documentation**: If you change behavior, update the docs

## 🚫 What We DON'T Want

- **Malicious code**: Obviously
- **Copyright violations**: Don't copy-paste licensed code without permission
- **Breaking changes without discussion**: Talk to us first for major changes
- **Spam or self-promotion**: Keep it relevant to LoRA training
- **Duplicate efforts**: Check existing PRs before starting work

## 🎉 Recognition

**Contributors get:**
- Your name in the contributors list (if you want it)
- Eternal gratitude from the community
- The satisfaction of making LoRA training better for everyone
- Bragging rights when your feature helps someone train their first successful LoRA

## 🤔 Questions?

**Not sure about something?**
- **Open a discussion** for general questions
- **Check existing issues** for similar problems
- **Ask in PRs/issues** for specific technical questions
- **Be specific** about what you need help with

## 📜 Legal Stuff

By contributing, you agree that:
- Your contributions will be licensed under the same MIT license
- You have the right to submit your contributions
- You're not submitting copyrighted code without permission

---

**"Either gonna work or blow up!"** - Thanks for helping make it work! 🎯

*P.S. If you're reading this far, you're definitely the kind of contributor we want. Welcome to the team! 🤝*