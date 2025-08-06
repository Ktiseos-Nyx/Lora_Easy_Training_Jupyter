#!/usr/bin/env python3
"""
Wrapper script that intercepts and fixes -e (editable) install issues
during Kohya-ss/Derrian backend installation.

This script monitors pip install commands and fixes requirements.txt files
that have problematic -e flags at the end.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import time
from pathlib import Path

class PipInstallFixer:
    def __init__(self):
        self.original_pip = None
        self.find_original_pip()
        
    def find_original_pip(self):
        """Find the real pip executable"""
        # Look for pip in common locations
        pip_candidates = [
            shutil.which('pip'),
            shutil.which('pip3'),
            sys.executable.replace('python', 'pip') if 'python' in sys.executable else None,
            os.path.join(os.path.dirname(sys.executable), 'pip'),
            os.path.join(os.path.dirname(sys.executable), 'pip3'),
        ]
        
        for candidate in pip_candidates:
            if candidate and os.path.exists(candidate):
                self.original_pip = candidate
                break
        
        if not self.original_pip:
            self.original_pip = 'pip'  # Fallback
    
    def fix_requirements_file(self, req_file):
        """Fix a requirements.txt file by removing problematic -e and file:// installs"""
        if not os.path.exists(req_file):
            return req_file
        
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            regular_reqs = []
            editable_reqs = []
            skipped_reqs = []
            has_issues = False
            
            for line_num, line in enumerate(lines, 1):
                original_line = line
                line = line.strip()
                
                if line and not line.startswith('#'):
                    # Check for problematic patterns
                    if (line.startswith('-e ') or line.startswith('--editable ') or 
                        line.startswith('file://') or 
                        ('file://' in line and not line.startswith('http'))):
                        
                        # Skip if it's a file:// URL pointing to the wrong project directory
                        if ('file://' in line and 
                            ('Lora_Easy_Training_Jupyter' in line or 
                             'file:///' in line)):
                            print(f"⚠️  Skipping misdirected file:// path (line {line_num}): {line}")
                            print(f"     💡 '-e .' is being resolved to wrong directory")
                            skipped_reqs.append(line)
                            has_issues = True
                            continue
                        else:
                            # Allow legitimate editable installs
                            print(f"📝 Keeping editable install (line {line_num}): {line}")
                            editable_reqs.append(line)
                            has_issues = True
                    else:
                        regular_reqs.append(line)
                elif line:  # Keep comments and empty lines
                    regular_reqs.append(line)
            
            if not has_issues:
                return req_file  # Nothing to fix
            
            # Create a temporary requirements file without problematic entries
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_file.write('\n'.join(regular_reqs))
            temp_file.close()
            
            print(f"🔧 Fixed requirements file: {req_file}")
            print(f"   📦 Regular packages: {len(regular_reqs)}")
            print(f"   📝 Editable packages to install separately: {len(editable_reqs)}")
            print(f"   🚫 Skipped problematic packages: {len(skipped_reqs)}")
            
            # Store editable requirements for later installation
            self.pending_editable = editable_reqs
            self.skipped_requirements = skipped_reqs
            
            return temp_file.name
            
        except Exception as e:
            print(f"⚠️  Error fixing {req_file}: {e}")
            return req_file
    
    def install_editable_packages(self, venv_pip=None):
        """Install any pending editable packages"""
        if not hasattr(self, 'pending_editable') or not self.pending_editable:
            return
        
        pip_cmd = venv_pip or self.original_pip
        
        for editable_req in self.pending_editable:
            print(f"🔄 Installing editable package: {editable_req}")
            try:
                # Try with -U first
                subprocess.run([pip_cmd, 'install', '-U'] + editable_req.split()[1:], 
                             check=True, capture_output=True)
                print(f"   ✅ Installed: {editable_req}")
            except subprocess.CalledProcessError:
                try:
                    # Try without -U
                    subprocess.run([pip_cmd, 'install'] + editable_req.split()[1:], 
                                 check=True, capture_output=True)
                    print(f"   ✅ Installed (no upgrade): {editable_req}")
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Failed to install {editable_req}: {e}")
        
        # Clear pending list
        self.pending_editable = []
    
    def run_pip_with_fix(self, args):
        """Run pip command, intercepting and fixing -r requirements.txt calls"""
        # Check if this is a requirements file install
        if '-r' in args:
            try:
                r_index = args.index('-r')
                if r_index + 1 < len(args):
                    req_file = args[r_index + 1]
                    
                    # Fix the requirements file
                    fixed_req_file = self.fix_requirements_file(req_file)
                    
                    # Replace the requirements file in args
                    new_args = args.copy()
                    new_args[r_index + 1] = fixed_req_file
                    
                    # Run pip with fixed requirements
                    result = subprocess.run([self.original_pip] + new_args[1:])
                    
                    # Install editable packages separately
                    self.install_editable_packages(args[0] if args[0] != 'pip' else None)
                    
                    # Clean up temp file
                    if fixed_req_file != req_file and os.path.exists(fixed_req_file):
                        os.unlink(fixed_req_file)
                    
                    return result.returncode
            except (ValueError, IndexError):
                pass
        
        # For non-requirements installs, just pass through
        return subprocess.run([self.original_pip] + args[1:]).returncode

def create_pip_wrapper():
    """Create a pip wrapper script that uses our fixer"""
    wrapper_script = """#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_editable_installs import PipInstallFixer

fixer = PipInstallFixer()
sys.exit(fixer.run_pip_with_fix(sys.argv))
"""
    
    # Create wrapper in a temp location
    wrapper_path = os.path.join(tempfile.gettempdir(), 'pip_wrapper.py')
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_script)
    os.chmod(wrapper_path, 0o755)
    
    return wrapper_path

def setup_pip_interceptor():
    """Set up pip interception for the current process"""
    print("🔧 Setting up pip install interceptor for -e flag fixes...")
    
    # Create the fixer
    fixer = PipInstallFixer()
    
    # Monkey patch subprocess to intercept pip calls
    original_run = subprocess.run
    original_check_call = subprocess.check_call
    
    def patched_run(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)) and len(args[0]) > 0:
            cmd = args[0]
            if 'pip' in str(cmd[0]) and 'install' in cmd and '-r' in cmd:
                print(f"🎯 Intercepting pip command: {' '.join(map(str, cmd))}")
                return_code = fixer.run_pip_with_fix(cmd)
                # Create a mock result object
                class MockResult:
                    def __init__(self, returncode):
                        self.returncode = returncode
                return MockResult(return_code)
        return original_run(*args, **kwargs)
    
    def patched_check_call(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)) and len(args[0]) > 0:
            cmd = args[0]
            if 'pip' in str(cmd[0]) and 'install' in cmd and '-r' in cmd:
                print(f"🎯 Intercepting pip check_call: {' '.join(map(str, cmd))}")
                return_code = fixer.run_pip_with_fix(cmd)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
                return
        elif isinstance(args[0], str) and 'pip' in args[0] and 'install' in args[0] and '-r' in args[0]:
            print(f"🎯 Intercepting pip shell command: {args[0]}")
            # Parse shell command
            import shlex
            cmd_parts = shlex.split(args[0])
            return_code = fixer.run_pip_with_fix(cmd_parts)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, args[0])
            return
        return original_check_call(*args, **kwargs)
    
    subprocess.run = patched_run
    subprocess.check_call = patched_check_call
    
    print("✅ Pip interceptor installed!")

if __name__ == "__main__":
    # If called directly, act as a pip wrapper
    fixer = PipInstallFixer()
    sys.exit(fixer.run_pip_with_fix(sys.argv))