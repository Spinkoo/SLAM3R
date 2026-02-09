#!/usr/bin/env python3
"""
Helper script to fix CUDA compatibility issues by installing the correct PyTorch version.
"""

import subprocess
import sys
import platform

def check_cuda_version():
    """Check system CUDA version."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    version_str = line.split('release')[1].strip().split(',')[0]
                    return version_str
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    version_str = line.split('CUDA Version:')[1].strip().split()[0]
                    return version_str
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None

def check_pytorch_cuda():
    """Check PyTorch CUDA version."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
        return None
    except ImportError:
        return None

def get_pytorch_install_command(cuda_version_str):
    """Get the appropriate PyTorch install command based on CUDA version."""
    if cuda_version_str is None:
        return None
    
    # Parse version
    try:
        major, minor = map(int, cuda_version_str.split('.')[:2])
    except:
        return None
    
    # CUDA 13.x -> use CUDA 12.1 (backward compatible)
    if major >= 13:
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    
    # CUDA 12.x -> use CUDA 12.1
    if major == 12:
        if minor >= 1:
            return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        else:
            return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    
    # CUDA 11.x -> use CUDA 11.8
    if major == 11:
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    
    # Default: try CUDA 12.1
    return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"

def main():
    print("=" * 60)
    print("PyTorch CUDA Compatibility Checker")
    print("=" * 60)
    
    system_cuda = check_cuda_version()
    pytorch_cuda = check_pytorch_cuda()
    
    print(f"\nSystem CUDA version: {system_cuda or 'Not detected'}")
    print(f"PyTorch CUDA version: {pytorch_cuda or 'Not available'}")
    
    if system_cuda and pytorch_cuda:
        try:
            sys_major = int(system_cuda.split('.')[0])
            pytorch_major = int(pytorch_cuda.split('.')[0])
            
            if sys_major != pytorch_major and sys_major >= 12:
                print(f"\n⚠️  Version mismatch detected!")
                print(f"   System: CUDA {system_cuda}")
                print(f"   PyTorch: CUDA {pytorch_cuda}")
                print(f"\n   This can cause 'no kernel image' errors.")
                
                install_cmd = get_pytorch_install_command(system_cuda)
                if install_cmd:
                    print(f"\n   Recommended fix:")
                    print(f"   {install_cmd}")
                    print(f"\n   Would you like to install this now? (y/n): ", end='')
                    
                    response = input().strip().lower()
                    if response == 'y':
                        print(f"\n   Installing PyTorch...")
                        subprocess.run(install_cmd.split(), check=True)
                        print(f"\n   ✅ Installation complete!")
                        print(f"   Please restart your Python session and try again.")
                    else:
                        print(f"\n   Skipped. Run the command manually when ready.")
                else:
                    print(f"\n   Could not determine appropriate PyTorch version.")
            else:
                print(f"\n✅ Versions appear compatible.")
        except Exception as e:
            print(f"\n   Error checking versions: {e}")
    elif not pytorch_cuda:
        print(f"\n⚠️  PyTorch CUDA not available.")
        print(f"   Install PyTorch with CUDA support:")
        if system_cuda:
            install_cmd = get_pytorch_install_command(system_cuda)
            if install_cmd:
                print(f"   {install_cmd}")
        else:
            print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"\n✅ CUDA setup looks good!")

if __name__ == "__main__":
    main()
