# Security Policy 🛡️

**TL;DR: Don't be evil, and if you find evil things, tell us privately.**

Look, we get it. Security policies are usually boring corporate-speak that nobody reads. But since you're here training AI models and potentially running code that downloads things from the internet, let's have a real talk about keeping your setup secure.

## 🎯 What We Actually Care About

### **Real Security Issues:**
- **Code execution vulnerabilities**: Can someone run arbitrary commands on your machine?
- **Malicious model downloads**: Are we properly validating downloaded files?
- **Data exfiltration**: Could someone steal your training datasets or API keys?
- **Dependency hijacking**: Are we pulling in sketchy packages?
- **File system access**: Can the tool write/read files it shouldn't?

### **Things We Don't Lose Sleep Over:**
- Theoretical academic attacks that require physical access to your GPU
- "Vulnerabilities" that require you to manually edit code to be malicious
- Issues that only affect people who intentionally bypass safety checks
- DoS attacks against your local Jupyter server (just restart it)

## 🚨 Reporting Security Issues

**Found something sketchy? Here's what to do:**

### **For Real Security Issues:**
1. **Don't open a public issue** (that's like announcing "hey everyone, here's how to hack this!")
2. **Email us privately**: [Create a security advisory](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/security/advisories/new) on GitHub
3. **Include details**: What's the issue? How did you find it? Can you reproduce it?
4. **Give us time**: We'll try to respond within 48 hours, fix within a week

### **For "Maybe Security Issues":**
- Not sure if it's a security problem? Just email us anyway
- Better safe than sorry
- We won't judge you for false alarms

## 🔒 What We Do to Stay Secure

### **Code Practices:**
- **Input validation**: We sanitize file paths, validate URLs, check file extensions
- **Safe defaults**: No auto-execution of downloaded scripts, no privileged operations
- **Dependency management**: We pin versions and regularly update dependencies
- **Error handling**: No stack traces that leak sensitive paths or data

### **Download Safety:**
- **URL validation**: We check that download URLs point to expected domains
- **File type checking**: We verify file extensions and magic bytes
- **Size limits**: No accidentally downloading 50GB "models" 
- **Virus scanning**: We recommend running downloads through your AV (we can't do this for you)

### **API Key Handling:**
- **No logging**: API keys don't get written to log files or console output
- **Environment variables**: Store tokens in env vars, not hardcoded
- **Scope limiting**: Use the most restricted permissions possible

## ⚠️ Stuff You Should Know

### **We Download Things From the Internet:**
- **Models from HuggingFace**: Generally safe, but verify the source
- **Models from Civitai**: Also generally safe, but again, verify
- **Training scripts from GitHub**: We clone from known-good repositories
- **Python packages**: Via pip from PyPI (supply chain attacks are a thing)

### **We Execute Code:**
- **Training scripts**: We run Python scripts that do intensive GPU work
- **Jupyter notebooks**: By design, these can execute arbitrary code
- **Shell commands**: For git operations, file extraction, system checks

### **Your Responsibilities:**
- **Don't run this on production servers**: It's a development/training tool
- **Use appropriate network isolation**: Consider firewalls if you're paranoid
- **Keep your system updated**: OS patches, Python updates, etc.
- **Verify your downloads**: Check model sources, don't blindly trust URLs
- **Use strong API keys**: Enable 2FA on your HuggingFace/GitHub accounts

## 🛠️ Security Best Practices for Users

### **Environment Setup:**
```bash
# Use virtual environments (the installer does this for you)
python -m venv lora_training
source lora_training/bin/activate

# Don't run as root (seriously, why would you?)
whoami  # Should NOT return 'root'

# Keep your system updated
sudo apt update && sudo apt upgrade  # Linux
brew update && brew upgrade          # macOS
```

### **API Token Management:**
```bash
# Store tokens in environment variables
export HF_TOKEN="your_token_here"
export CIVITAI_TOKEN="your_other_token"

# Or use .env files (not committed to git)
echo "HF_TOKEN=your_token_here" >> .env
```

### **Network Security:**
- **Use HTTPS**: All our default URLs use HTTPS
- **VPN if paranoid**: Especially on public networks
- **Firewall rules**: Block unnecessary inbound connections to Jupyter

### **File System Hygiene:**
- **Separate training directory**: Don't mix with important personal files
- **Regular backups**: Your trained LoRAs are valuable!
- **Clean up temp files**: The tool tries to, but check occasionally

## 🔍 Supported Versions

We provide security updates for:
- **Latest release**: Always gets security fixes
- **Previous release**: Gets critical security fixes for 3 months
- **Older versions**: You're on your own, please upgrade

## 🎭 Threat Model

**What we protect against:**
- Malicious models that could exploit training scripts
- Network-based attacks during downloads
- Local privilege escalation through file operations
- Data leakage through logs or error messages

**What we DON'T protect against:**
- Nation-state actors (if the NSA wants your waifu LoRAs, we can't help)
- Physical access attacks (lock your computer)
- Social engineering (don't give strangers your API keys)
- Quantum computers (not a thing yet, probably)

## 🚀 Security Roadmap

**Stuff we're working on:**
- Better input sanitization for all user inputs
- Cryptographic verification of downloaded models
- Sandboxing for training processes
- Automated dependency vulnerability scanning

**Stuff we might do eventually:**
- Integration with hardware security modules
- Zero-knowledge training (probably overkill)
- Blockchain-based model verification (probably a bad idea)

## 📞 Contact

**Security issues**: Use GitHub Security Advisories (preferred)
**General questions**: Open a regular GitHub issue
**Urgent stuff**: Find us on Discord (link in README)

---

**Remember**: Perfect security is impossible, but good security is achievable. We're trying to hit that sweet spot between "reasonably secure" and "actually usable for training LoRAs."

*Stay safe out there, and may your models converge quickly! 🎯*

---

**Last updated**: Check the git commit history
**Next review**: When we add major new features or someone finds something scary