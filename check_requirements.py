#!/usr/bin/env python3
"""
Requirements checker for LoRA Easy Training
Validates that all core dependencies are working properly
"""

import sys
import importlib
import subprocess

def check_package(package_name, import_name=None, version_check=None):
    """Check if a package is installed and working"""
    import_name = import_name or package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Get version if available
        version = None
        if hasattr(module, '__version__'):
            version = module.__version__
        elif version_check:
            version = version_check(module)
        
        version_str = f" ({version})" if version else ""
        print(f"✅ {package_name}{version_str}")
        return True
        
    except ImportError as e:
        print(f"❌ {package_name} - {e}")
        return False

def check_pillow():
    """Special check for Pillow with functionality test"""
    try:
        from PIL import Image, ImageDraw
        
        # Test basic functionality
        img = Image.new('RGB', (100, 100), color='red')
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 50, 50], fill='blue')
        
        print(f"✅ Pillow ({Image.__version__}) - Full functionality")
        return True
        
    except ImportError as e:
        print(f"❌ Pillow not installed - {e}")
        return False
    except Exception as e:
        print(f"⚠️  Pillow installed but not working properly - {e}")
        return False

def main():
    print("🔍 LoRA Easy Training - Requirements Check")
    print("=" * 45)
    
    # Core packages needed for the notebook system
    packages = [
        ('Jupyter', 'jupyter'),
        ('IPython', 'IPython'),
        ('ipywidgets', 'ipywidgets'),
        ('toml', 'toml'),
        ('requests', 'requests'),
        ('tqdm', 'tqdm'),
        ('numpy', 'numpy'),
    ]
    
    print("📦 Checking core packages...")
    working = 0
    total = len(packages) + 1  # +1 for Pillow
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            working += 1
    
    # Special check for Pillow
    print("🖼️  Checking Pillow (with functionality test)...")
    if check_pillow():
        working += 1
    
    print(f"\n📊 Results: {working}/{total} packages working")
    
    if working == total:
        print("🎉 All core requirements are working!")
        return 0
    else:
        print(f"⚠️  {total - working} packages need attention")
        print("\n💡 To fix issues:")
        print("1. Run the main installer: python installer.py")
        print("2. Or install missing packages manually with pip")
        return 1

if __name__ == "__main__":
    sys.exit(main())