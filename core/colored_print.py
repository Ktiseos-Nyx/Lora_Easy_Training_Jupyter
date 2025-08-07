# core/colored_print.py
# Enhanced console printing with colors and styles
# Inspired by Linaqruf's colablib colored_print module
# https://github.com/Linaqruf/colablib

import datetime
import sys

# ANSI Color codes
color_codes = {
    "default"      : "\033[0m",
    "black"        : "\033[0;30m",
    "red"          : "\033[0;31m",
    "green"        : "\033[0;32m",
    "yellow"       : "\033[0;33m",
    "blue"         : "\033[0;34m",
    "purple"       : "\033[0;35m",
    "cyan"         : "\033[0;36m",
    "white"        : "\033[0;37m",
    "bright_black" : "\033[1;30m",
    "bright_red"   : "\033[1;31m",
    "bright_green" : "\033[1;32m",
    "bright_yellow": "\033[1;33m",
    "bright_blue"  : "\033[1;34m",
    "bright_purple": "\033[1;35m",
    "bright_cyan"  : "\033[1;36m",
    "bright_white" : "\033[1;37m",
    # Flat colors (RGB)
    "flat_red"     : "\033[38;2;204;102;102m",
    "flat_yellow"  : "\033[38;2;255;204;0m",
    "flat_blue"    : "\033[38;2;0;102;204m",
    "flat_purple"  : "\033[38;2;153;51;255m",
    "flat_orange"  : "\033[38;2;255;153;0m",
    "flat_green"   : "\033[38;2;0;204;102m",
    "flat_gray"    : "\033[38;2;128;128;128m",
    "flat_cyan"    : "\033[38;2;0;255;255m",
    "flat_pink"    : "\033[38;2;255;0;255m",
}

# Background color codes
bg_color_codes = {
    "bg_black"     : "\033[40m",
    "bg_red"       : "\033[41m",
    "bg_green"     : "\033[42m",
    "bg_yellow"    : "\033[43m",
    "bg_blue"      : "\033[44m",
    "bg_purple"    : "\033[45m",
    "bg_cyan"      : "\033[46m",
    "bg_white"     : "\033[47m",
}

# Style codes
style_codes = {
    "normal"       : "\033[0m",
    "bold"         : "\033[1m",
    "dim"          : "\033[2m",
    "italic"       : "\033[3m",
    "underline"    : "\033[4m",
    "blink"        : "\033[5m",
    "reverse"      : "\033[7m",
    "strikethrough": "\033[9m",
}

def is_jupyter():
    """Check if running in Jupyter notebook environment"""
    try:
        return 'ipykernel' in sys.modules or 'IPython' in sys.modules
    except:
        return False

def cprint(*args, color="default", style="normal", bg_color=None, reset=True, 
          timestamp=False, prefix="", suffix="", sep=" ", end="\n", 
          file=None, flush=False):
    """
    Enhanced print function with color and style support
    
    Args:
        *args: Objects to print
        color: Text color (str) - see color_codes keys
        style: Text style (str) - see style_codes keys  
        bg_color: Background color (str) - see bg_color_codes keys
        reset: Whether to reset formatting after print (bool)
        timestamp: Whether to add timestamp prefix (bool)
        prefix: Text to add before the main content (str)
        suffix: Text to add after the main content (str)
        sep: String inserted between values (str)
        end: String appended after the last value (str)
        file: File object to write to (defaults to sys.stdout)
        flush: Whether to forcibly flush the stream (bool)
    """
    if file is None:
        file = sys.stdout
    
    # Build the message
    message_parts = []
    
    # Add timestamp if requested
    if timestamp:
        now = datetime.datetime.now()
        ts = now.strftime("[%H:%M:%S]")
        message_parts.append(ts)
    
    # Add prefix
    if prefix:
        message_parts.append(prefix)
    
    # Add main content
    if args:
        content = sep.join(str(arg) for arg in args)
        message_parts.append(content)
    
    # Add suffix
    if suffix:
        message_parts.append(suffix)
    
    # Join all parts
    full_message = " ".join(message_parts)
    
    # If in Jupyter, colors might not work well, so we'll be more conservative
    if is_jupyter():
        # For Jupyter, we'll use a simpler approach with emojis and formatting
        emoji_map = {
            "red": "❌",
            "green": "✅", 
            "yellow": "⚠️",
            "blue": "ℹ️",
            "purple": "🔮",
            "cyan": "💎",
            "flat_orange": "🔶",
            "flat_pink": "💖"
        }
        
        if color in emoji_map:
            full_message = f"{emoji_map[color]} {full_message}"
        
        print(full_message, end=end, file=file, flush=flush)
        return
    
    # For terminal output, use full ANSI colors
    output_parts = []
    
    # Apply color
    if color in color_codes:
        output_parts.append(color_codes[color])
    
    # Apply background color
    if bg_color and f"bg_{bg_color}" in bg_color_codes:
        output_parts.append(bg_color_codes[f"bg_{bg_color}"])
    
    # Apply style
    if style in style_codes:
        output_parts.append(style_codes[style])
    
    # Add the message
    output_parts.append(full_message)
    
    # Reset formatting if requested
    if reset:
        output_parts.append(color_codes["default"])
    
    # Print the formatted message
    formatted_message = "".join(output_parts)
    print(formatted_message, end=end, file=file, flush=flush)

def print_line(length=50, char="=", color="default", style="normal"):
    """
    Print a line of characters with optional color and style
    
    Args:
        length: Number of characters to print (int)
        char: Character to repeat (str)
        color: Line color (str)
        style: Line style (str)
    """
    line = char * length
    cprint(line, color=color, style=style)

def print_header(text, length=60, char="=", color="bright_blue", style="bold"):
    """
    Print a formatted header with surrounding lines
    
    Args:
        text: Header text (str)
        length: Total width (int)
        char: Border character (str)
        color: Header color (str)
        style: Header style (str)
    """
    print_line(length, char, color, style)
    # Center the text
    padding = (length - len(text)) // 2
    centered_text = " " * padding + text + " " * padding
    if len(centered_text) < length:
        centered_text += " "  # Add one more space if needed
    cprint(centered_text, color=color, style=style)
    print_line(length, char, color, style)

def print_status(status_type, message, timestamp=True):
    """
    Print a status message with appropriate color coding
    
    Args:
        status_type: Type of status ('success', 'error', 'warning', 'info', 'debug')
        message: Status message (str)
        timestamp: Whether to include timestamp (bool)
    """
    status_config = {
        'success': {'color': 'green', 'prefix': '✅ SUCCESS:'},
        'error': {'color': 'red', 'prefix': '❌ ERROR:'},
        'warning': {'color': 'yellow', 'prefix': '⚠️  WARNING:'},
        'info': {'color': 'blue', 'prefix': 'ℹ️  INFO:'},
        'debug': {'color': 'purple', 'prefix': '🔍 DEBUG:'},
        'progress': {'color': 'cyan', 'prefix': '🚀 PROGRESS:'}
    }
    
    if status_type in status_config:
        config = status_config[status_type]
        cprint(message, 
               color=config['color'], 
               style="bold", 
               prefix=config['prefix'],
               timestamp=timestamp)
    else:
        cprint(message, timestamp=timestamp)

# Convenience functions for common use cases
def success(message, timestamp=True):
    """Print success message"""
    print_status('success', message, timestamp)

def error(message, timestamp=True):
    """Print error message"""
    print_status('error', message, timestamp)

def warning(message, timestamp=True):
    """Print warning message"""
    print_status('warning', message, timestamp)

def info(message, timestamp=True):
    """Print info message"""
    print_status('info', message, timestamp)

def debug(message, timestamp=True):
    """Print debug message"""
    print_status('debug', message, timestamp)

def progress(message, timestamp=True):
    """Print progress message"""
    print_status('progress', message, timestamp)

# Export main functions
__all__ = [
    'cprint', 'print_line', 'print_header', 'print_status',
    'success', 'error', 'warning', 'info', 'debug', 'progress',
    'color_codes', 'style_codes', 'bg_color_codes'
]