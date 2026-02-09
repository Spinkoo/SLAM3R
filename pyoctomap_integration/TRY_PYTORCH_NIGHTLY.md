# Try PyTorch Nightly for Blackwell GPU Support

Your NVIDIA RTX PRO 2000 Blackwell GPU (compute capability 12.0) is very new. 
PyTorch nightly builds may already have experimental support.

## Quick Test:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Install PyTorch nightly (may have Blackwell support)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121

# Test if it works
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); x=torch.zeros(1).cuda(); print('CUDA works!' if x.is_cuda else 'CUDA failed')"
```

## If Nightly Works:

Run your script normally with `--device cuda`:
```bash
python -m pyoctomap_integration.recon_with_octomap_incremental --device cuda --test_name my_test --dataset data/room0/ ...
```

## If Nightly Doesn't Work:

1. **Use CPU mode** (works but slower):
   ```bash
   python -m pyoctomap_integration.recon_with_octomap_incremental --device cpu ...
   ```

2. **Wait for official PyTorch release** - Check https://pytorch.org/get-started/locally/

3. **Build PyTorch from source** (advanced, requires compiling with sm_120 support)

## Check Current Support:

```bash
python -c "import torch; print('Supported compute capabilities:', torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else 'N/A')"
```
